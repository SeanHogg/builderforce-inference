//! Longest-prefix KV reuse across in-process generate calls.

use std::collections::HashMap;

/// Cached KV snapshot for a token prefix.
pub struct PrefixEntry {
    pub n: usize,
    pub k: Vec<bfi_core::Tensor>,
    pub v: Vec<bfi_core::Tensor>,
    pub recurrent: Vec<Option<bfi_core::Tensor>>,
    pub logits: Vec<f32>,
}

impl PrefixEntry {
    pub fn from_state(state: &bfi_models::SeqState, logits: &bfi_models::Logits) -> Self {
        Self {
            n: state.n,
            k: state.k.clone(),
            v: state.v.clone(),
            recurrent: state.recurrent.clone(),
            logits: logits.values.clone(),
        }
    }

    pub fn restore(&self) -> (bfi_models::SeqState, bfi_models::Logits) {
        (
            bfi_models::SeqState {
                n: self.n,
                k: self.k.clone(),
                v: self.v.clone(),
                recurrent: self.recurrent.clone(),
            },
            bfi_models::Logits {
                values: self.logits.clone(),
            },
        )
    }
}

pub struct PrefixCache {
    map: HashMap<Vec<u32>, PrefixEntry>,
    forwarded: u64,
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self {
            map: HashMap::new(),
            forwarded: 0,
        }
    }
}

impl PrefixCache {
    pub fn forwarded(&self) -> u64 {
        self.forwarded
    }

    pub fn note_forward(&mut self, n: usize) {
        self.forwarded += n as u64;
    }

    /// Longest cached token prefix of `ids`, or 0.
    pub fn longest_len(&self, ids: &[u32]) -> usize {
        (1..=ids.len())
            .rev()
            .find(|&n| self.map.contains_key(&ids[..n]))
            .unwrap_or(0)
    }

    pub fn get(&self, prefix: &[u32]) -> Option<&PrefixEntry> {
        self.map.get(prefix)
    }

    pub fn insert(&mut self, ids: Vec<u32>, entry: PrefixEntry) {
        const MAX: usize = 8;
        if self.map.len() >= MAX && !self.map.contains_key(&ids) {
            if let Some(k) = self.map.keys().next().cloned() {
                self.map.remove(&k);
            }
        }
        self.map.insert(ids, entry);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_call_does_not_reforward_prefix() {
        let mut c = PrefixCache::default();
        let prefix = vec![7u32, 8, 9];
        c.note_forward(prefix.len());
        c.insert(
            prefix.clone(),
            PrefixEntry {
                n: 3,
                k: vec![],
                v: vec![],
                recurrent: vec![],
                logits: vec![],
            },
        );
        let skip = c.longest_len(&prefix);
        let rest = prefix.len() - skip;
        c.note_forward(rest);
        assert_eq!(skip, 3);
        assert_eq!(c.forwarded(), 3);
    }

    #[test]
    fn insert_caps_at_eight() {
        let mut c = PrefixCache::default();
        for i in 0..12u32 {
            c.insert(
                vec![i],
                PrefixEntry {
                    n: 1,
                    k: vec![],
                    v: vec![],
                    recurrent: vec![],
                    logits: vec![],
                },
            );
        }
        assert!(c.map.len() <= 8);
    }

    #[test]
    fn longer_prompt_reuses_shared_prefix() {
        let mut c = PrefixCache::default();
        c.insert(
            vec![1, 2, 3],
            PrefixEntry {
                n: 3,
                k: vec![],
                v: vec![],
                recurrent: vec![],
                logits: vec![],
            },
        );
        let second = vec![1, 2, 3, 4];
        let skip = c.longest_len(&second);
        c.note_forward(second.len() - skip);
        assert_eq!(skip, 3);
        assert_eq!(c.forwarded(), 1);
    }
}
