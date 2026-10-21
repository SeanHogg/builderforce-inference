//! GGUF fixtures shared by the model tests: a 1-bit Bonsai-shaped toy qwen3.

use std::collections::BTreeMap;

use bfi_gguf::writer::{write_gguf, TensorWrite};
use bfi_gguf::{lookup_type, quantize_row_q1_0, MetaValue, TYPE_F32, TYPE_Q1_0};

pub fn pattern(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((((i * 7919 + seed * 104_729) % 2003) as f32) / 1001.5 - 1.0) * 0.5)
        .collect()
}

pub fn q1(name: &str, n_in: u64, n_out: u64, seed: usize) -> TensorWrite {
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

pub fn norm(name: &str, n: u64, seed: usize) -> TensorWrite {
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

/// 1-bit Bonsai-shaped toy: every matrix Q1_0, GQA 2q/1kv, head_dim 64, two layers,
/// trained context 32. `tied` drops `output.weight` (the lm_head is then the token
/// embedding, as in Bonsai 1.7B / 4B); untied is Bonsai 8B's layout.
pub fn bonsai_toy(tied: bool) -> Vec<u8> {
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
    let mut t = vec![q1("token_embd.weight", e, vocab, 1), norm("output_norm.weight", e, 2)];
    if !tied {
        t.push(q1("output.weight", e, vocab, 3));
    }
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
