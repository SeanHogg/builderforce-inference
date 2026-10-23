//! Longest-prefix KV reuse across in-process generate calls.
//!
//! An LRU bounded by BYTES (not entry count). Each entry is a token sequence plus the
//! model state after it. Lookup is O(prompt length): one rolling hash over the prompt,
//! probed against marks at every [`BLOCK`]-token boundary and at every entry's end, then
//! an exact token comparison of the deepest hit (hash collisions can never restore the
//! wrong state). A share shorter than one block is found by comparing the first block of
//! each entry, O(entries × BLOCK).
//!
//! KV-only models (qwen3) are REWINDABLE: an entry for `prompt + generated` also serves
//! any shorter prefix of it, by restoring with a smaller `n` (rows past `n` are simply
//! overwritten). Models with recurrent state (qwen35) cannot rewind, so their entries
//! only restore at exactly their own length.
//!
//! Copy costs that remain, by design:
//! - Insert from a borrowed state (the prompt state at prefill, while decode keeps
//!   going) copies the USED K/V rows (`n × n_kv × head_dim` per layer, not the full
//!   `n_ctx` capacity) plus the recurrent tensors.
//! - Insert from an owned state (on completion) moves the buffers and only truncates
//!   them to the used rows — no copy of the data.
//! - Restore allocates full-capacity K/V and copies the used rows in, plus clones the
//!   recurrent tensors: the entry stays cached for the next caller.

use std::collections::HashMap;

use bfi_core::{Storage, Tensor};

/// Prefix-hash probe granularity for rewindable entries.
pub const BLOCK: usize = 64;

/// Where a cached prefix can resume a prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hit {
    pub id: u64,
    /// Tokens of the prompt the entry agrees with.
    pub matched: usize,
    /// Tokens to restore. Equals `matched` except when the match ends inside a
    /// rewindable entry at the prompt's end: the logits are only known at the entry's end,
    /// so one prompt token is left to forward.
    pub restore: usize,
}

/// One K or V layer, kept as its used rows when the storage allows it.
enum Kept {
    Rows { shape: Vec<usize>, used: Vec<f32> },
    Whole(Tensor),
}

impl Kept {
    fn bytes(&self) -> usize {
        match self {
            Kept::Rows { used, .. } => used.len() * 4,
            Kept::Whole(t) => tensor_bytes(t),
        }
    }

    fn row_len(shape: &[usize]) -> usize {
        shape.iter().skip(1).product::<usize>().max(1)
    }

    fn from_owned(t: Tensor, n: usize) -> Self {
        match t.storage {
            Storage::F32(mut data) if !t.shape.is_empty() => {
                data.truncate(n * Self::row_len(&t.shape));
                data.shrink_to_fit();
                Kept::Rows { shape: t.shape, used: data }
            }
            storage => Kept::Whole(Tensor { shape: t.shape, dtype: t.dtype, storage }),
        }
    }

    fn from_borrowed(t: &Tensor, n: usize) -> Self {
        match &t.storage {
            Storage::F32(data) if !t.shape.is_empty() => {
                let used = (n * Self::row_len(&t.shape)).min(data.len());
                Kept::Rows { shape: t.shape.clone(), used: data[..used].to_vec() }
            }
            _ => Kept::Whole(t.clone()),
        }
    }

    fn restore(&self) -> Tensor {
        match self {
            Kept::Rows { shape, used } => {
                let mut data = vec![0.0f32; shape.iter().product()];
                data[..used.len()].copy_from_slice(used);
                Tensor::f32(shape.clone(), data)
            }
            Kept::Whole(t) => t.clone(),
        }
    }
}

fn tensor_bytes(t: &Tensor) -> usize {
    match &t.storage {
        Storage::F32(v) => v.len() * 4,
        Storage::Quant { bytes, .. } => bytes.len(),
    }
}

struct Entry {
    tokens: Vec<u32>,
    k: Vec<Kept>,
    v: Vec<Kept>,
    recurrent: Vec<Option<Tensor>>,
    logits: Vec<f32>,
    rewindable: bool,
    bytes: usize,
    last_used: u64,
    /// Prefix hashes this entry is registered under in [`PrefixCache::marks`].
    marks: Vec<u64>,
}

pub struct PrefixCache {
    entries: HashMap<u64, Entry>,
    /// Prefix hash → entries with that prefix at a block boundary or at their end.
    marks: HashMap<u64, Vec<u64>>,
    budget_bytes: usize,
    used_bytes: usize,
    next_id: u64,
    tick: u64,
    forwarded: u64,
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self::new(super::options::DEFAULT_PREFIX_CACHE_BYTES)
    }
}

impl PrefixCache {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            marks: HashMap::new(),
            budget_bytes,
            used_bytes: 0,
            next_id: 0,
            tick: 0,
            forwarded: 0,
        }
    }

    pub fn forwarded(&self) -> u64 {
        self.forwarded
    }

    pub fn note_forward(&mut self, n: usize) {
        self.forwarded += n as u64;
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.marks.clear();
        self.used_bytes = 0;
    }

    /// The best cached resume point for `ids`, or `None`.
    pub fn lookup(&self, ids: &[u32]) -> Option<Hit> {
        // Deepest probe first: walk once, remember every (len, entry) the hashes name.
        let mut probes: Vec<(usize, u64)> = Vec::new();
        let mut h = HASH_SEED;
        for (i, &t) in ids.iter().enumerate() {
            h = mix(h, t);
            if let Some(list) = self.marks.get(&h) {
                for &id in list {
                    probes.push((i + 1, id));
                }
            }
        }
        let mut best: Option<(usize, u64)> = None;
        for &(len, id) in probes.iter().rev() {
            if best.is_some_and(|(b, _)| b >= len + BLOCK) {
                break; // nothing shallower can extend past the best match
            }
            let Some(e) = self.entries.get(&id) else { continue };
            if e.tokens.len() < len || e.tokens[..len] != ids[..len] {
                continue; // hash collision
            }
            let matched = if e.rewindable {
                len + common_prefix(&e.tokens[len..], &ids[len..])
            } else if len == e.tokens.len() {
                len
            } else {
                continue; // a recurrent state cannot resume mid-entry
            };
            if best.map_or(true, |(b, _)| matched > b) {
                best = Some((matched, id));
            }
        }
        // A share shorter than one block reaches no boundary mark: compare the first
        // block of every rewindable entry directly (bounded by entries × BLOCK).
        if best.map_or(true, |(b, _)| b < BLOCK) {
            for (&id, e) in self.entries.iter().filter(|(_, e)| e.rewindable) {
                let matched = common_prefix(&e.tokens, ids);
                if matched > 0 && best.map_or(true, |(b, _)| matched > b) {
                    best = Some((matched, id));
                }
            }
        }
        let (matched, id) = best?;
        let e = &self.entries[&id];
        let restore = if matched == e.tokens.len() { matched } else { matched.min(ids.len().saturating_sub(1)) };
        (restore > 0).then_some(Hit { id, matched, restore })
    }

    /// Rebuild the state of `hit` (and its logits when it resumes at the entry's end).
    pub fn restore(&mut self, hit: Hit) -> Option<(bfi_models::SeqState, Option<bfi_models::Logits>)> {
        self.tick += 1;
        let tick = self.tick;
        let e = self.entries.get_mut(&hit.id)?;
        e.last_used = tick;
        let state = bfi_models::SeqState {
            n: hit.restore,
            k: e.k.iter().map(Kept::restore).collect(),
            v: e.v.iter().map(Kept::restore).collect(),
            recurrent: e.recurrent.clone(),
        };
        let logits = (hit.restore == e.tokens.len()).then(|| bfi_models::Logits { values: e.logits.clone() });
        Some((state, logits))
    }

    /// Cache a state the caller keeps using (copies its used rows).
    pub fn insert_copy(&mut self, tokens: &[u32], state: &bfi_models::SeqState, logits: &bfi_models::Logits) {
        if tokens.is_empty() || state.n != tokens.len() {
            return;
        }
        let n = state.n;
        self.insert_entry(
            tokens.to_vec(),
            state.k.iter().map(|t| Kept::from_borrowed(t, n)).collect(),
            state.v.iter().map(|t| Kept::from_borrowed(t, n)).collect(),
            state.recurrent.clone(),
            logits.values.clone(),
        );
    }

    /// Cache a finished sequence's state, taking ownership (no data copy).
    pub fn insert_owned(&mut self, tokens: Vec<u32>, state: bfi_models::SeqState, logits: bfi_models::Logits) {
        if tokens.is_empty() || state.n != tokens.len() {
            return;
        }
        let n = state.n;
        self.insert_entry(
            tokens,
            state.k.into_iter().map(|t| Kept::from_owned(t, n)).collect(),
            state.v.into_iter().map(|t| Kept::from_owned(t, n)).collect(),
            state.recurrent,
            logits.values,
        );
    }

    fn insert_entry(
        &mut self,
        tokens: Vec<u32>,
        k: Vec<Kept>,
        v: Vec<Kept>,
        recurrent: Vec<Option<Tensor>>,
        logits: Vec<f32>,
    ) {
        let rewindable = recurrent.iter().all(Option::is_none);
        let bytes = tokens.len() * 4
            + logits.len() * 4
            + k.iter().chain(&v).map(Kept::bytes).sum::<usize>()
            + recurrent.iter().flatten().map(tensor_bytes).sum::<usize>();
        if bytes > self.budget_bytes {
            return;
        }
        self.drop_superseded(&tokens, rewindable);
        while self.used_bytes + bytes > self.budget_bytes {
            if !self.evict_lru() {
                break;
            }
        }

        // Marks: every BLOCK boundary (rewindable entries resume anywhere) and the end.
        let mut marks = Vec::new();
        let mut h = HASH_SEED;
        for (i, &t) in tokens.iter().enumerate() {
            h = mix(h, t);
            let len = i + 1;
            if len == tokens.len() || (rewindable && len % BLOCK == 0) {
                marks.push(h);
            }
        }
        self.tick += 1;
        let id = self.next_id;
        self.next_id += 1;
        for &m in &marks {
            self.marks.entry(m).or_default().push(id);
        }
        self.used_bytes += bytes;
        self.entries.insert(
            id,
            Entry { tokens, k, v, recurrent, logits, rewindable, bytes, last_used: self.tick, marks },
        );
    }

    /// An identical sequence, or (rewindable) any cached prefix of the new one, is
    /// redundant once the new entry exists.
    fn drop_superseded(&mut self, tokens: &[u32], rewindable: bool) {
        let mut doomed = Vec::new();
        for (&id, e) in &self.entries {
            let same = e.tokens.as_slice() == tokens;
            let covered = rewindable && e.rewindable && tokens.starts_with(&e.tokens);
            if same || covered {
                doomed.push(id);
            }
        }
        for id in doomed {
            self.remove(id);
        }
    }

    fn evict_lru(&mut self) -> bool {
        let oldest = self.entries.iter().min_by_key(|(_, e)| e.last_used).map(|(&id, _)| id);
        match oldest {
            Some(id) => {
                self.remove(id);
                true
            }
            None => false,
        }
    }

    fn remove(&mut self, id: u64) {
        let Some(e) = self.entries.remove(&id) else { return };
        self.used_bytes -= e.bytes;
        for m in e.marks {
            if let Some(list) = self.marks.get_mut(&m) {
                list.retain(|&x| x != id);
                if list.is_empty() {
                    self.marks.remove(&m);
                }
            }
        }
    }
}

const HASH_SEED: u64 = 0xcbf2_9ce4_8422_2325;

/// Rolling prefix hash: `h(ids[..n+1]) = mix(h(ids[..n]), ids[n])`.
fn mix(h: u64, t: u32) -> u64 {
    (h.rotate_left(5) ^ t as u64).wrapping_mul(0x5851_f42d_4c95_7f2d) ^ 0x9e37_79b9
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A KV-only (rewindable) state of `n` tokens: 1 layer, 2 floats per row, capacity 512.
    fn kv_state(n: usize) -> bfi_models::SeqState {
        let mut data = vec![0.0f32; 512 * 2];
        for (i, x) in data.iter_mut().enumerate().take(n * 2) {
            *x = i as f32 + 1.0;
        }
        bfi_models::SeqState {
            n,
            k: vec![Tensor::f32(vec![512, 1, 2], data.clone())],
            v: vec![Tensor::f32(vec![512, 1, 2], data)],
            recurrent: vec![None],
        }
    }

    fn recurrent_state(n: usize) -> bfi_models::SeqState {
        let mut s = kv_state(n);
        s.recurrent = vec![Some(Tensor::f32(vec![1, 2, 2], vec![0.5; 4]))];
        s
    }

    fn logits() -> bfi_models::Logits {
        bfi_models::Logits { values: vec![0.1, 0.2] }
    }

    fn seq(n: usize, salt: u32) -> Vec<u32> {
        (0..n as u32).map(|i| i * 7 + salt).collect()
    }

    #[test]
    fn exact_hit_restores_logits_and_compact_rows() {
        let mut c = PrefixCache::new(1 << 20);
        let ids = seq(5, 0);
        c.insert_owned(ids.clone(), kv_state(5), logits());
        let hit = c.lookup(&ids).unwrap();
        assert_eq!((hit.matched, hit.restore), (5, 5));
        let (state, l) = c.restore(hit).unwrap();
        assert_eq!(state.n, 5);
        assert_eq!(l.unwrap().values, vec![0.1, 0.2]);
        let k = state.k[0].as_f32().unwrap();
        assert_eq!(k.len(), 1024, "restored at full capacity");
        assert_eq!(&k[..10], &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
        assert!(k[10..].iter().all(|&x| x == 0.0));
        // Only the used rows are held: 5 rows × 2 floats × (k + v), not 512 rows.
        assert!(c.used_bytes() < 200, "{}", c.used_bytes());
    }

    #[test]
    fn longest_match_rewinds_inside_a_generated_entry() {
        let mut c = PrefixCache::new(1 << 20);
        // prompt + generated, longer than one block.
        let cached = seq(150, 0);
        c.insert_owned(cached.clone(), kv_state(150), logits());

        // A new prompt shares 130 tokens, then diverges.
        let mut prompt = cached[..130].to_vec();
        prompt.extend([9999, 9998]);
        let hit = c.lookup(&prompt).unwrap();
        assert_eq!(hit.matched, 130);
        assert_eq!(hit.restore, 130);
        let (state, l) = c.restore(hit).unwrap();
        assert_eq!(state.n, 130);
        assert!(l.is_none(), "logits are only known at the entry's end");

        // The whole prompt is inside the entry: leave one token to forward for logits.
        let inside = cached[..100].to_vec();
        let hit = c.lookup(&inside).unwrap();
        assert_eq!((hit.matched, hit.restore), (100, 99));

        // Nothing shared → no hit.
        assert!(c.lookup(&[4242, 4243]).is_none());
    }

    #[test]
    fn recurrent_entries_only_resume_at_their_end() {
        let mut c = PrefixCache::new(1 << 20);
        let cached = seq(80, 1);
        c.insert_owned(cached.clone(), recurrent_state(80), logits());
        // Shares 70 tokens: cannot rewind a recurrent state to 70.
        let mut p = cached[..70].to_vec();
        p.push(1_000_000);
        assert!(c.lookup(&p).is_none());
        // Extends the entry: resumes at 80.
        let mut longer = cached.clone();
        longer.extend([5, 6]);
        let hit = c.lookup(&longer).unwrap();
        assert_eq!((hit.matched, hit.restore), (80, 80));
        let (state, l) = c.restore(hit).unwrap();
        assert!(state.recurrent[0].is_some());
        assert!(l.is_some());
    }

    #[test]
    fn evicts_least_recently_used_by_bytes() {
        // Each 4-token entry: 16 B tokens + 8 B logits + 2 × 32 B rows = 88 B.
        let mut c = PrefixCache::new(200);
        let (a, b, d) = (seq(4, 100), seq(4, 200), seq(4, 300));
        c.insert_owned(a.clone(), kv_state(4), logits());
        c.insert_owned(b.clone(), kv_state(4), logits());
        assert_eq!(c.len(), 2);
        // Touch `a`: `b` becomes the least recently used.
        let hit = c.lookup(&a).unwrap();
        c.restore(hit);
        c.insert_owned(d.clone(), kv_state(4), logits());
        assert_eq!(c.len(), 2);
        assert!(c.used_bytes() <= 200);
        assert!(c.lookup(&a).is_some(), "recently used survives");
        assert!(c.lookup(&b).is_none(), "LRU evicted");
        assert!(c.lookup(&d).is_some());

        // An entry larger than the whole budget is not cached (and evicts nothing).
        c.insert_owned(seq(100, 7), kv_state(100), logits());
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn extending_entry_supersedes_its_prefix() {
        let mut c = PrefixCache::new(1 << 20);
        let prompt = seq(10, 3);
        let s = kv_state(10);
        c.insert_copy(&prompt, &s, &logits());
        let mut done = prompt.clone();
        done.extend([1, 2, 3]);
        c.insert_owned(done.clone(), kv_state(13), logits());
        assert_eq!(c.len(), 1, "the prompt entry is a rewind of the completion entry");
        let hit = c.lookup(&prompt).unwrap();
        assert_eq!((hit.matched, hit.restore), (10, 9));
    }

    #[test]
    fn forwarded_counter() {
        let mut c = PrefixCache::default();
        c.note_forward(3);
        c.note_forward(1);
        assert_eq!(c.forwarded(), 4);
    }
}
