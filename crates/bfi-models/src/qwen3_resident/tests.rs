//! Resident qwen3 vs the CPU reference, on native wgpu (skipped without an adapter).

use super::*;
use std::sync::Arc;

use bfi_core::resident::{block_on, open_native, Gpu};
use bfi_core::{max_rel_err, CpuBackend};
use bfi_gguf::{GgufFile, TensorStream};

use crate::kv_plan::KvShape;
use crate::test_fixtures::bonsai_toy;
use crate::SeqState;

#[test]
fn context_budget_is_bounded_by_kv_binding_and_training() {
    let Some(gpu) = bfi_core::gpu_or_skip(open_native(), "the resident qwen3 tests") else { return };
    let bytes = bonsai_toy(false);
    let mut stream = TensorStream::new();
    stream.push(&bytes).unwrap();
    stream.flush_header().unwrap();
    let loader = Qwen3ResidentLoader::new(gpu, stream.header().unwrap()).unwrap();
    // toy: 2 layers, 1 kv head × 64; trained ctx 32.
    let shape = loader.kv_shape();
    assert_eq!(shape, KvShape { n_layer: 2, n_head_kv: 1, head_dim: 64, trained_ctx: 32 });
    assert_eq!(crate::kv_plan::kv_bytes_per_token(&shape, false), 2 * 2 * 64 * 4);
    assert_eq!(crate::served_context(&shape, false, u64::MAX, u64::MAX), 32);
    assert_eq!(crate::served_context(&shape, false, 10 * 1024, u64::MAX), 10);
    assert_eq!(crate::served_context(&shape, true, 10 * 1024, u64::MAX), 20);
}

#[test]
fn resident_matches_cpu_qwen3_prompt_then_decode_f32_kv() {
    resident_vs_cpu(false, 2e-3, u64::MAX);
}

#[test]
fn resident_matches_cpu_qwen3_prompt_then_decode_f16_kv() {
    resident_vs_cpu(true, 1e-2, u64::MAX);
}

#[test]
fn small_binding_shrinks_prefill_chunk_not_results() {
    // 2 heads · 32 ctx · 4 B = 256 B per score row → a 512 B binding forces
    // 2-token prefill chunks; logits must not change.
    resident_vs_cpu(false, 2e-3, 512);
}

/// Stream the toy GGUF into a loader in 4 KiB chunks, as the browser does.
fn toy_loader(gpu: &Gpu, bytes: &[u8]) -> Qwen3ResidentLoader {
    let mut stream = TensorStream::new();
    stream.push(bytes).unwrap();
    stream.flush_header().unwrap();
    let mut loader = Qwen3ResidentLoader::new(gpu.clone(), stream.header().unwrap()).unwrap();
    let mut s2 = TensorStream::new();
    for c in bytes.chunks(4096) {
        for t in s2.push(c).unwrap() {
            loader.accept(t).unwrap();
        }
    }
    for t in s2.flush_header().unwrap() {
        loader.accept(t).unwrap();
    }
    s2.finish().unwrap();
    loader
}

#[test]
fn prefill_then_forward_matches_one_forward() {
    let Some(gpu) = bfi_core::gpu_or_skip(open_native(), "the resident qwen3 tests") else { return };
    let bytes = bonsai_toy(false);
    let prompt = [1u32, 5, 9, 3, 14];
    let mut whole = toy_loader(&gpu, &bytes).finish(32, false, u64::MAX).unwrap();
    let want = block_on(gpu.read_f32(&whole.forward(&prompt).unwrap())).unwrap();

    let mut split = toy_loader(&gpu, &bytes).finish(32, false, u64::MAX).unwrap();
    split.prefill(&prompt[..3]).unwrap();
    block_on(gpu.drain()).unwrap();
    assert_eq!(split.n_past(), 3);
    let got = block_on(gpu.read_f32(&split.forward(&prompt[3..]).unwrap())).unwrap();
    let err = max_rel_err(&got, &want);
    assert!(err <= 1e-4, "split prefill rel err {err}");
    assert!(split.prefill_yield_tokens() >= PREFILL_YIELD_CHUNKS);
}

#[test]
fn assemble_then_allocate_kv_retries_without_reloading() {
    let Some(gpu) = bfi_core::gpu_or_skip(open_native(), "the resident qwen3 tests") else { return };
    let bytes = bonsai_toy(false);
    let mut model = toy_loader(&gpu, &bytes).assemble(false, u64::MAX).unwrap();
    assert_eq!(model.n_ctx(), 0);
    let none = model.forward(&[1]).err().unwrap().to_string();
    assert!(none.starts_with(CONTEXT_EXCEEDED), "{none}");
    // load's retry: a larger context first (clamped to the trained 32), then a smaller one.
    model.allocate_kv(1_000);
    assert_eq!(model.n_ctx(), 32);
    model.forward(&[1, 2]).unwrap();
    model.allocate_kv(8);
    assert_eq!((model.n_ctx(), model.n_past()), (8, 0), "reallocation resets the sequence");
    model.forward(&[1, 2]).unwrap();
    model.release_kv();
    assert_eq!((model.n_ctx(), model.n_past()), (0, 0));
}

fn resident_vs_cpu(kv_f16: bool, tol: f32, max_binding: u64) {
    let Some(gpu) = bfi_core::gpu_or_skip(open_native(), "the resident qwen3 tests") else { return };
    let bytes = bonsai_toy(false);

    let file = GgufFile::from_bytes("toy.gguf", bytes.clone()).unwrap();
    let cpu_model = crate::load_model(&file, Arc::new(CpuBackend)).unwrap();
    let mut state = SeqState::allocate(&cpu_model.state_layout());
    let prompt = [1u32, 5, 9, 3, 14];
    let pos: Vec<i32> = (0..prompt.len() as i32).collect();
    let want_prompt = cpu_model.forward(&prompt, &pos, &mut state).unwrap();
    let want_step = cpu_model.forward(&[7], &[prompt.len() as i32], &mut state).unwrap();

    let mut model = toy_loader(&gpu, &bytes).finish(32, kv_f16, max_binding).unwrap();
    assert_eq!(model.kv_f16(), kv_f16);
    assert_eq!(model.n_vocab(), 16);

    let got = model.forward(&prompt).unwrap();
    let got = block_on(gpu.read_f32(&got)).unwrap();
    let err = max_rel_err(&got, want_prompt.as_slice());
    assert!(err <= tol, "prompt logits rel err {err}\n{got:?}\n{:?}", want_prompt.as_slice());

    let got = model.forward(&[7]).unwrap();
    let got = block_on(gpu.read_f32(&got)).unwrap();
    let err = max_rel_err(&got, want_step.as_slice());
    assert!(err <= tol, "decode logits rel err {err}");

    let over = model.forward(&vec![1u32; 32]).err().unwrap().to_string();
    assert!(over.starts_with(CONTEXT_EXCEEDED), "{over}");
}
