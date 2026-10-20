//! Resident qwen3 vs the CPU reference, on native wgpu (skipped without an adapter).

use super::*;
use std::collections::BTreeMap;
use std::sync::Arc;

use bfi_core::resident::{block_on, open_native, Gpu};
use bfi_core::{max_rel_err, CpuBackend};
use bfi_gguf::writer::{write_gguf, TensorWrite};
use bfi_gguf::{lookup_type, quantize_row_q1_0, GgufFile, MetaValue, TensorStream, TYPE_F32, TYPE_Q1_0};

use crate::kv_plan::KvShape;
use crate::SeqState;

fn pattern(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((((i * 7919 + seed * 104_729) % 2003) as f32) / 1001.5 - 1.0) * 0.5)
        .collect()
}

fn q1(name: &str, n_in: u64, n_out: u64, seed: usize) -> TensorWrite {
    let w = pattern((n_in * n_out) as usize, seed);
    let mut data = vec![0u8; w.len() / 128 * 18];
    for (b, c) in w.chunks(128).enumerate() {
        quantize_row_q1_0(c, &mut data[b * 18..b * 18 + 18]);
    }
    TensorWrite {
        name: name.into(),
        dims: vec![n_in, n_out],
        ty: lookup_type(TYPE_Q1_0).unwrap(),
        data,
    }
}

fn norm(name: &str, n: u64, seed: usize) -> TensorWrite {
    let data = pattern(n as usize, seed)
        .iter()
        .flat_map(|v| (1.0 + v * 0.2).to_le_bytes())
        .collect();
    TensorWrite {
        name: name.into(),
        dims: vec![n],
        ty: lookup_type(TYPE_F32).unwrap(),
        data,
    }
}

/// 1-bit Bonsai-shaped toy: every matrix Q1_0, untied lm_head, GQA 2q/1kv,
/// head_dim 64, two layers.
fn bonsai_toy() -> Vec<u8> {
    let (e, d, ff, vocab) = (128u64, 64u64, 256u64, 16u64);
    let mut meta = BTreeMap::new();
    let s = |v: &str| MetaValue::String(v.into());
    meta.insert("general.architecture".into(), s("qwen3"));
    meta.insert("qwen3.block_count".into(), MetaValue::U32(2));
    meta.insert("qwen3.embedding_length".into(), MetaValue::U32(e as u32));
    meta.insert("qwen3.feed_forward_length".into(), MetaValue::U32(ff as u32));
    meta.insert("qwen3.context_length".into(), MetaValue::U32(32));
    meta.insert("qwen3.attention.head_count".into(), MetaValue::U32(2));
    meta.insert("qwen3.attention.head_count_kv".into(), MetaValue::U32(1));
    meta.insert("qwen3.attention.key_length".into(), MetaValue::U32(d as u32));
    meta.insert("qwen3.attention.layer_norm_rms_epsilon".into(), MetaValue::F32(1e-6));
    meta.insert("qwen3.rope.freq_base".into(), MetaValue::F32(1_000_000.0));
    let mut t = vec![
        q1("token_embd.weight", e, vocab, 1),
        norm("output_norm.weight", e, 2),
        q1("output.weight", e, vocab, 3),
    ];
    for i in 0..2usize {
        let p = format!("blk.{i}");
        let k = 10 + i * 20;
        t.extend([
            norm(&format!("{p}.attn_norm.weight"), e, k),
            q1(&format!("{p}.attn_q.weight"), e, 2 * d, k + 1),
            q1(&format!("{p}.attn_k.weight"), e, d, k + 2),
            q1(&format!("{p}.attn_v.weight"), e, d, k + 3),
            q1(&format!("{p}.attn_output.weight"), 2 * d, e, k + 4),
            norm(&format!("{p}.attn_q_norm.weight"), d, k + 5),
            norm(&format!("{p}.attn_k_norm.weight"), d, k + 6),
            norm(&format!("{p}.ffn_norm.weight"), e, k + 7),
            q1(&format!("{p}.ffn_gate.weight"), e, ff, k + 8),
            q1(&format!("{p}.ffn_up.weight"), e, ff, k + 9),
            q1(&format!("{p}.ffn_down.weight"), ff, e, k + 10),
        ]);
    }
    write_gguf(&meta, &t).unwrap()
}

#[test]
fn context_budget_is_bounded_by_kv_binding_and_training() {
    let Some(gpu) = bfi_core::gpu_or_skip(open_native(), "the resident qwen3 tests") else { return };
    let bytes = bonsai_toy();
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
    let bytes = bonsai_toy();
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
    let bytes = bonsai_toy();
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
    let bytes = bonsai_toy();

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
