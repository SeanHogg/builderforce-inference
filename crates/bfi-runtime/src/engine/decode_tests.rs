//! Decode-loop tests on a toy `Model` (no GGUF): speculative decoding must reproduce
//! plain greedy decode exactly, roll back rejected drafts on both state kinds, and the
//! prefix cache must resume prompts it has seen.

use std::sync::Arc;

use anyhow::Result;
use bfi_core::{Backend, CpuBackend, Tensor};
use bfi_models::{Logits, Model, SeqState, StateLayout};

use super::*;

const V: usize = 6;
const CTX: usize = 128;

/// Next token = a function of the WHOLE history the state holds, so a wrong rollback
/// shows up as divergent output. KV toy: history lives in K rows (rewound by `n`).
/// Recurrent toy: additionally folds every token into a running sum.
struct Toy {
    recurrent: bool,
}

fn rule(history: &[u32], acc: f32) -> u32 {
    let n = history.len();
    let last = *history.last().unwrap_or(&0) as usize;
    // A 1→2→3→4→1 cycle that prompt lookup drafts well, broken every 5th position
    // (and, for the recurrent toy, by the running sum) so some drafts are rejected.
    let cycle = [1, 2, 3, 4, 1, 1];
    let mut next = if n % 5 == 0 { (last + 2) % V } else { cycle[last] };
    if acc as usize % 7 == 3 {
        next = (next + 1) % V;
    }
    next as u32
}

impl Model for Toy {
    fn architecture(&self) -> &'static str {
        "toy"
    }
    fn n_layer(&self) -> usize {
        1
    }
    fn n_embd(&self) -> usize {
        1
    }
    fn n_vocab(&self) -> usize {
        V
    }
    fn n_ctx(&self) -> usize {
        CTX
    }
    fn n_head(&self) -> usize {
        1
    }
    fn n_head_kv(&self) -> usize {
        1
    }
    fn head_dim(&self) -> usize {
        1
    }
    fn forward(&self, tokens: &[u32], _pos: &[i32], state: &mut SeqState) -> Result<Logits> {
        anyhow::ensure!(state.n + tokens.len() <= CTX, "toy KV overflow");
        let mut acc = 0.0;
        for &t in tokens {
            let n = state.n;
            match &mut state.k[0].storage {
                bfi_core::Storage::F32(rows) => rows[n] = t as f32,
                _ => unreachable!(),
            }
            state.n += 1;
            if self.recurrent {
                if let Some(Tensor { storage: bfi_core::Storage::F32(r), .. }) = state.recurrent[0].as_mut() {
                    r[0] += t as f32;
                    acc = r[0];
                }
            }
        }
        let history: Vec<u32> = state.k[0].as_f32()?[..state.n].iter().map(|&x| x as u32).collect();
        let mut values = vec![0.0; V];
        values[rule(&history, acc) as usize] = 10.0;
        Ok(Logits { values })
    }
    fn state_layout(&self) -> StateLayout {
        StateLayout {
            n_layer: 1,
            n_ctx: CTX,
            n_head_kv: 1,
            head_dim: 1,
            n_recurrent: usize::from(self.recurrent),
            recurrent_shape: if self.recurrent { [1, 1, 1] } else { [0, 0, 0] },
        }
    }
}

fn engine(recurrent: bool, spec: Option<SpecLookup>) -> Engine {
    let tokens: Vec<String> = ["a", "b", "c", "d", "e", "f"].into_iter().map(String::from).collect();
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend);
    Engine {
        model: Box::new(Toy { recurrent }),
        tokenizer: Tokenizer::from_parts(tokens, Vec::new()),
        _backend: backend,
        prefix_cache: PrefixCache::new(DEFAULT_PREFIX_CACHE_BYTES),
        token_embd: None,
        last_batch_decode_steps: 0,
        options: EngineOptions { spec_lookup: spec, ..EngineOptions::default() },
        spec_stats: SpecStats::default(),
        last_generated: 0,
    }
}

fn run(e: &mut Engine, prompt: &[u32], params: &SampleParams) -> Vec<u32> {
    let mut slot = e.prefill_ids(prompt.to_vec(), params).unwrap();
    while !slot.done {
        e.step(&mut slot, params).unwrap();
    }
    e.finish(slot)
}

fn greedy(max_tokens: u32) -> SampleParams {
    SampleParams { max_tokens, temperature: 0.0, ..SampleParams::default() }
}

const PROMPT: [u32; 9] = [1, 2, 3, 4, 1, 2, 3, 4, 1];

#[test]
fn spec_lookup_greedy_matches_plain_decode_kv_state() {
    let params = greedy(40);
    let plain = run(&mut engine(false, None), &PROMPT, &params);
    let mut spec = engine(false, Some(SpecLookup::default()));
    let fast = run(&mut spec, &PROMPT, &params);
    assert_eq!(plain.len(), 40);
    assert_eq!(fast, plain, "speculative greedy must be token-identical");
    let st = spec.spec_stats();
    assert!(st.verify_passes > 0 && st.accepted > 0, "{st:?}");
    assert!(st.accepted < st.drafted, "the toy breaks the cycle, so some drafts are rejected: {st:?}");
    // Fewer model passes than tokens: that is the speedup.
    assert!(st.verify_passes < 40);
}

#[test]
fn spec_lookup_rolls_back_recurrent_state() {
    let params = greedy(40);
    let plain = run(&mut engine(true, None), &PROMPT, &params);
    let mut spec = engine(true, Some(SpecLookup::default()));
    let fast = run(&mut spec, &PROMPT, &params);
    assert_eq!(fast, plain, "recurrent rollback must restore the snapshot exactly");
    assert!(spec.spec_stats().accepted > 0);
}

#[test]
fn spec_lookup_respects_max_tokens_and_context() {
    for max in [1u32, 2, 3, 7] {
        let mut e = engine(false, Some(SpecLookup { draft_len: 8, ..SpecLookup::default() }));
        assert_eq!(run(&mut e, &PROMPT, &greedy(max)).len(), max as usize);
    }
    // Generating to the context limit never overflows the KV.
    let mut e = engine(false, Some(SpecLookup::default()));
    let out = run(&mut e, &PROMPT, &greedy((CTX - PROMPT.len()) as u32 - 1));
    assert_eq!(out.len(), CTX - PROMPT.len() - 1);
}

#[test]
fn spec_lookup_sampling_runs_to_length() {
    let params = SampleParams { max_tokens: 30, temperature: 0.8, seed: 7, ..SampleParams::default() };
    let mut e = engine(false, Some(SpecLookup::default()));
    let out = run(&mut e, &PROMPT, &params);
    assert_eq!(out.len(), 30);
    assert!(out.iter().all(|&t| (t as usize) < V));
}

#[test]
fn completion_is_cached_and_resumes_the_next_prompt() {
    // `forwarded_tokens` counts PREFILL forwards (cache misses), not decode steps.
    let params = greedy(10);
    let mut e = engine(false, None);
    let first = run(&mut e, &PROMPT, &params);
    assert_eq!(e.forwarded_tokens() as usize, PROMPT.len());

    // Next turn = prompt + answer + new text: only the new text is forwarded, because
    // the finished sequence (prompt + generated) was cached on completion.
    let mut turn2 = PROMPT.to_vec();
    turn2.extend(&first);
    turn2.extend([5, 0]);
    let before = e.forwarded_tokens();
    let _ = run(&mut e, &turn2, &params);
    assert_eq!(e.forwarded_tokens() - before, 2);

    // The first prompt again: rewinds the longer entry, forwarding one token for logits.
    let before = e.forwarded_tokens();
    let again = run(&mut e, &PROMPT, &params);
    assert_eq!(again, first);
    assert_eq!(e.forwarded_tokens() - before, 1);
}
