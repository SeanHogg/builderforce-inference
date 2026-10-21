//! Architecture registry for bfi.
//!
//! Architectures self-register through [`inventory`] ([`ArchReg`]). A new model
//! is one module plus an `inventory::submit!` — the engine is not edited.
//!
//! M2: `qwen3` (dense, all full attention, Q/K-norm per head, GQA, SwiGLU).
//! M3: `qwen35` (Bonsai): hybrid gated-delta + full attention, multi-section RoPE.
//! Feature `webgpu`: `Qwen3Resident`, the GPU-resident qwen3 for the browser
//! (1-bit Bonsai `Q1_0`), loaded tensor-by-tensor from a stream.
//! Always: `Qwen3StreamLoader` → [`Session`], the same qwen3 streamed onto a host-tensor
//! `Backend` (the browser engine's CPU backend, when there is no WebGPU).

mod arch;
pub mod kv_plan;
mod qwen3;
#[cfg(feature = "webgpu")]
mod qwen3_resident;
mod qwen3_stream;
mod qwen35;
mod session;
mod state;
#[cfg(test)]
mod test_fixtures;
mod tutil;

pub use arch::{architecture_for, load_model, ArchReg, Architecture, Model};
pub use qwen3::{Qwen3Arch, ARCH_NAME as QWEN3_ARCH, QWEN3};
#[cfg(feature = "webgpu")]
pub use qwen3_resident::{Qwen3Resident, Qwen3ResidentLoader, PREFILL_CHUNK, PREFILL_YIELD_CHUNKS};
pub use qwen3_stream::Qwen3StreamLoader;
pub use session::{Session, CONTEXT_EXCEEDED};
pub use kv_plan::{served_context, KvShape};
pub use qwen35::{Qwen35Arch, ARCH_NAME as QWEN35_ARCH, QWEN35};
pub use state::{Logits, SeqState, StateLayout};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use bfi_core::{Backend, CpuBackend};
    use bfi_gguf::writer::{write_gguf, TensorWrite};
    use bfi_gguf::{
        lookup_type, GgufFile, MetaValue, TYPE_F32, TYPE_Q4_K, TYPE_Q6_K, TYPE_Q8_0,
    };

    fn meta_str(s: &str) -> MetaValue {
        MetaValue::String(s.to_string())
    }

    fn f32_write(name: &str, dims: Vec<u64>, fill: f32) -> TensorWrite {
        let n: usize = dims.iter().map(|d| *d as usize).product();
        let mut data = Vec::with_capacity(n * 4);
        for i in 0..n {
            let v = fill + (i as f32) * 0.0001;
            data.extend_from_slice(&v.to_le_bytes());
        }
        TensorWrite {
            name: name.into(),
            dims,
            ty: lookup_type(TYPE_F32).expect("F32 type"),
            data,
        }
    }

    fn ones_write(name: &str, dims: Vec<u64>) -> TensorWrite {
        let n: usize = dims.iter().map(|d| *d as usize).product();
        let mut data = Vec::with_capacity(n * 4);
        for _ in 0..n {
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }
        TensorWrite {
            name: name.into(),
            dims,
            ty: lookup_type(TYPE_F32).expect("F32 type"),
            data,
        }
    }

    /// Tiny dense qwen3: n_embd=8, n_head=2, n_kv=1, head_dim=4, n_ff=16, vocab=4.
    fn synthetic_qwen3() -> Vec<u8> {
        let mut meta = BTreeMap::new();
        meta.insert("general.architecture".into(), meta_str("qwen3"));
        meta.insert("general.name".into(), meta_str("synth"));
        meta.insert("qwen3.block_count".into(), MetaValue::U32(1));
        meta.insert("qwen3.embedding_length".into(), MetaValue::U32(8));
        meta.insert("qwen3.feed_forward_length".into(), MetaValue::U32(16));
        meta.insert("qwen3.context_length".into(), MetaValue::U32(8));
        meta.insert("qwen3.attention.head_count".into(), MetaValue::U32(2));
        meta.insert("qwen3.attention.head_count_kv".into(), MetaValue::U32(1));
        meta.insert("qwen3.attention.key_length".into(), MetaValue::U32(4));
        meta.insert("qwen3.attention.value_length".into(), MetaValue::U32(4));
        meta.insert(
            "qwen3.attention.layer_norm_rms_epsilon".into(),
            MetaValue::F32(1e-6),
        );
        meta.insert("qwen3.rope.freq_base".into(), MetaValue::F32(10_000.0));
        meta.insert("qwen3.rope.dimension_count".into(), MetaValue::U32(4));

        // GGUF dims are innermost-first (ne[0] fastest).
        let tensors = vec![
            f32_write("token_embd.weight", vec![8, 4], 0.02),
            ones_write("output_norm.weight", vec![8]),
            f32_write("output.weight", vec![8, 4], 0.01),
            ones_write("blk.0.attn_norm.weight", vec![8]),
            f32_write("blk.0.attn_q.weight", vec![8, 8], 0.03),
            f32_write("blk.0.attn_k.weight", vec![8, 4], 0.03),
            f32_write("blk.0.attn_v.weight", vec![8, 4], 0.03),
            f32_write("blk.0.attn_output.weight", vec![8, 8], 0.03),
            ones_write("blk.0.attn_q_norm.weight", vec![4]),
            ones_write("blk.0.attn_k_norm.weight", vec![4]),
            ones_write("blk.0.ffn_norm.weight", vec![8]),
            f32_write("blk.0.ffn_gate.weight", vec![8, 16], 0.02),
            f32_write("blk.0.ffn_up.weight", vec![8, 16], 0.02),
            f32_write("blk.0.ffn_down.weight", vec![16, 8], 0.02),
        ];
        write_gguf(&meta, &tensors).expect("write synthetic qwen3 gguf")
    }

    /// Tiny hybrid qwen35: 2 layers (linear + full), same toy dims as qwen3.
    /// Includes an MTP tensor that must load and not run.
    fn synthetic_qwen35() -> Vec<u8> {
        let mut meta = BTreeMap::new();
        meta.insert("general.architecture".into(), meta_str("qwen35"));
        meta.insert("general.name".into(), meta_str("synth-bonsai"));
        meta.insert("qwen35.block_count".into(), MetaValue::U32(2));
        meta.insert("qwen35.embedding_length".into(), MetaValue::U32(8));
        meta.insert("qwen35.feed_forward_length".into(), MetaValue::U32(16));
        meta.insert("qwen35.context_length".into(), MetaValue::U32(8));
        meta.insert("qwen35.attention.head_count".into(), MetaValue::U32(2));
        meta.insert("qwen35.attention.head_count_kv".into(), MetaValue::U32(1));
        meta.insert("qwen35.attention.key_length".into(), MetaValue::U32(4));
        meta.insert("qwen35.attention.value_length".into(), MetaValue::U32(4));
        meta.insert(
            "qwen35.attention.layer_norm_rms_epsilon".into(),
            MetaValue::F32(1e-6),
        );
        meta.insert("qwen35.rope.freq_base".into(), MetaValue::F32(10_000.0));
        meta.insert("qwen35.rope.dimension_count".into(), MetaValue::U32(4));
        meta.insert(
            "qwen35.rope.dimension_sections".into(),
            MetaValue::Array(vec![
                MetaValue::U32(2),
                MetaValue::U32(2),
                MetaValue::U32(0),
                MetaValue::U32(0),
            ]),
        );
        meta.insert("qwen35.ssm.conv_kernel".into(), MetaValue::U32(2));
        meta.insert(
            "qwen35.attention.layer_types".into(),
            MetaValue::Array(vec![
                meta_str("linear_attention"),
                meta_str("full_attention"),
            ]),
        );

        let mut tensors = vec![
            f32_write("token_embd.weight", vec![8, 4], 0.02),
            ones_write("output_norm.weight", vec![8]),
            f32_write("output.weight", vec![8, 4], 0.01),
            ones_write("mtp.0.output_norm.weight", vec![8]),
        ];
        // Layer 0: gated-delta (linear). Mixed QKV channels = 8+4+4 = 16.
        tensors.extend([
            ones_write("blk.0.attn_norm.weight", vec![8]),
            f32_write("blk.0.attn_q.weight", vec![8, 8], 0.03),
            f32_write("blk.0.attn_k.weight", vec![8, 4], 0.03),
            f32_write("blk.0.attn_v.weight", vec![8, 4], 0.03),
            f32_write("blk.0.attn_output.weight", vec![8, 8], 0.03),
            f32_write("blk.0.ssm_conv1d.weight", vec![2, 16], 0.1),
            f32_write("blk.0.attn_gate.weight", vec![8, 1], 0.02),
            f32_write("blk.0.ssm_dt.weight", vec![8, 1], 0.02),
            ones_write("blk.0.ffn_norm.weight", vec![8]),
            f32_write("blk.0.ffn_gate.weight", vec![8, 16], 0.02),
            f32_write("blk.0.ffn_up.weight", vec![8, 16], 0.02),
            f32_write("blk.0.ffn_down.weight", vec![16, 8], 0.02),
        ]);
        // Layer 1: full attention + multi-section RoPE.
        tensors.extend([
            ones_write("blk.1.attn_norm.weight", vec![8]),
            f32_write("blk.1.attn_q.weight", vec![8, 8], 0.03),
            f32_write("blk.1.attn_k.weight", vec![8, 4], 0.03),
            f32_write("blk.1.attn_v.weight", vec![8, 4], 0.03),
            f32_write("blk.1.attn_output.weight", vec![8, 8], 0.03),
            ones_write("blk.1.attn_q_norm.weight", vec![4]),
            ones_write("blk.1.attn_k_norm.weight", vec![4]),
            ones_write("blk.1.ffn_norm.weight", vec![8]),
            f32_write("blk.1.ffn_gate.weight", vec![8, 16], 0.02),
            f32_write("blk.1.ffn_up.weight", vec![8, 16], 0.02),
            f32_write("blk.1.ffn_down.weight", vec![16, 8], 0.02),
        ]);
        write_gguf(&meta, &tensors).expect("write synthetic qwen35 gguf")
    }

    fn open_bytes(bytes: &[u8]) -> (tempfile::TempDir, GgufFile) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.gguf");
        std::fs::write(&path, bytes).unwrap();
        let file = GgufFile::open(&path).unwrap();
        (dir, file)
    }

    fn qwen3_toy_meta(n_layer: u32, n_ff: u32) -> BTreeMap<String, MetaValue> {
        let mut meta = BTreeMap::new();
        meta.insert("general.architecture".into(), meta_str("qwen3"));
        meta.insert("general.name".into(), meta_str("synth"));
        meta.insert("qwen3.block_count".into(), MetaValue::U32(n_layer));
        meta.insert("qwen3.embedding_length".into(), MetaValue::U32(8));
        meta.insert("qwen3.feed_forward_length".into(), MetaValue::U32(n_ff));
        meta.insert("qwen3.context_length".into(), MetaValue::U32(8));
        meta.insert("qwen3.attention.head_count".into(), MetaValue::U32(2));
        meta.insert("qwen3.attention.head_count_kv".into(), MetaValue::U32(1));
        meta.insert("qwen3.attention.key_length".into(), MetaValue::U32(4));
        meta.insert("qwen3.attention.value_length".into(), MetaValue::U32(4));
        meta.insert(
            "qwen3.attention.layer_norm_rms_epsilon".into(),
            MetaValue::F32(1e-6),
        );
        meta.insert("qwen3.rope.freq_base".into(), MetaValue::F32(10_000.0));
        meta.insert("qwen3.rope.dimension_count".into(), MetaValue::U32(4));
        meta
    }

    fn qwen3_dense_layer_f32(i: usize, n_ff: u64) -> Vec<TensorWrite> {
        let p = format!("blk.{i}");
        vec![
            ones_write(&format!("{p}.attn_norm.weight"), vec![8]),
            f32_write(&format!("{p}.attn_q.weight"), vec![8, 8], 0.03),
            f32_write(&format!("{p}.attn_k.weight"), vec![8, 4], 0.03),
            f32_write(&format!("{p}.attn_v.weight"), vec![8, 4], 0.03),
            f32_write(&format!("{p}.attn_output.weight"), vec![8, 8], 0.03),
            ones_write(&format!("{p}.attn_q_norm.weight"), vec![4]),
            ones_write(&format!("{p}.attn_k_norm.weight"), vec![4]),
            ones_write(&format!("{p}.ffn_norm.weight"), vec![8]),
            f32_write(&format!("{p}.ffn_gate.weight"), vec![8, n_ff], 0.02),
            f32_write(&format!("{p}.ffn_up.weight"), vec![8, n_ff], 0.02),
            f32_write(&format!("{p}.ffn_down.weight"), vec![n_ff, 8], 0.02),
        ]
    }

    /// Two-layer dense qwen3 (same toy dims as [`synthetic_qwen3`]).
    fn synthetic_qwen3_two_layer() -> Vec<u8> {
        let meta = qwen3_toy_meta(2, 16);
        let mut tensors = vec![
            f32_write("token_embd.weight", vec![8, 4], 0.02),
            ones_write("output_norm.weight", vec![8]),
            f32_write("output.weight", vec![8, 4], 0.01),
        ];
        tensors.extend(qwen3_dense_layer_f32(0, 16));
        tensors.extend(qwen3_dense_layer_f32(1, 16));
        write_gguf(&meta, &tensors).expect("write two-layer qwen3 gguf")
    }

    /// Qwen3-0.6B-style tied output: no `output.weight`, lm_head = token_embd.
    fn synthetic_qwen3_tied_output() -> Vec<u8> {
        let meta = qwen3_toy_meta(1, 16);
        let mut tensors = vec![
            f32_write("token_embd.weight", vec![8, 4], 0.02),
            ones_write("output_norm.weight", vec![8]),
        ];
        tensors.extend(qwen3_dense_layer_f32(0, 16));
        write_gguf(&meta, &tensors).expect("write tied-output qwen3 gguf")
    }

    fn quant_write(name: &str, dims: Vec<u64>, ty_id: u32) -> TensorWrite {
        let ty = lookup_type(ty_id).unwrap_or_else(|| panic!("unknown ggml type {ty_id}"));
        let n: u64 = dims.iter().copied().product();
        let nbytes = ty.bytes_for_elems(n).unwrap_or_else(|| {
            panic!(
                "{name}: {n} elems not aligned to {} block {}",
                ty.name, ty.block_elems
            )
        });
        TensorWrite {
            name: name.into(),
            dims,
            ty,
            data: vec![0u8; nbytes as usize],
        }
    }

    /// Mixed-quant toy qwen3: Q8_0 attn, Q4_K/Q6_K FFN (n_ff=32 so K-quants align).
    /// Norms and embeddings stay F32 (real GGUF layout).
    fn synthetic_qwen3_quant_mix() -> Vec<u8> {
        let meta = qwen3_toy_meta(1, 32);
        let tensors = vec![
            f32_write("token_embd.weight", vec![8, 4], 0.02),
            ones_write("output_norm.weight", vec![8]),
            f32_write("output.weight", vec![8, 4], 0.01),
            ones_write("blk.0.attn_norm.weight", vec![8]),
            quant_write("blk.0.attn_q.weight", vec![8, 8], TYPE_Q8_0),
            quant_write("blk.0.attn_k.weight", vec![8, 4], TYPE_Q8_0),
            quant_write("blk.0.attn_v.weight", vec![8, 4], TYPE_Q8_0),
            quant_write("blk.0.attn_output.weight", vec![8, 8], TYPE_Q8_0),
            ones_write("blk.0.attn_q_norm.weight", vec![4]),
            ones_write("blk.0.attn_k_norm.weight", vec![4]),
            ones_write("blk.0.ffn_norm.weight", vec![8]),
            quant_write("blk.0.ffn_gate.weight", vec![8, 32], TYPE_Q4_K),
            quant_write("blk.0.ffn_up.weight", vec![8, 32], TYPE_Q6_K),
            quant_write("blk.0.ffn_down.weight", vec![32, 8], TYPE_Q4_K),
        ];
        write_gguf(&meta, &tensors).expect("write quantized qwen3 gguf")
    }

    fn assert_finite_logits(logits: &Logits, n_vocab: usize) {
        assert_eq!(logits.n_vocab(), n_vocab);
        assert!(
            logits.as_slice().iter().all(|v| v.is_finite()),
            "logits not finite: {:?}",
            logits.as_slice()
        );
    }

    #[test]
    fn registry_lookup_by_architecture_name() {
        let arch = architecture_for("qwen3").expect("qwen3 must self-register via inventory");
        assert_eq!(arch.name(), "qwen3");
        assert_eq!(QWEN3.name(), "qwen3");
        assert_eq!(QWEN3_ARCH, "qwen3");
        assert_eq!(
            architecture_for(QWEN3_ARCH).map(|a| a.name()),
            Some("qwen3"),
            "QWEN3_ARCH alias must resolve through architecture_for"
        );
        assert_eq!(architecture_for(QWEN3.name()).map(|a| a.name()), Some("qwen3"));
        let q35 = architecture_for("qwen35").expect("qwen35 must self-register via inventory");
        assert_eq!(q35.name(), "qwen35");
        assert_eq!(QWEN35.name(), "qwen35");
        assert!(architecture_for("llama").is_none());
        assert!(architecture_for("").is_none());
        assert!(architecture_for("qwen").is_none());
        assert!(architecture_for("Qwen3").is_none());
        let mut known: Vec<&str> = inventory::iter::<ArchReg>
            .into_iter()
            .map(|r| r.0.name())
            .collect();
        known.sort();
        assert_eq!(known, vec!["qwen3", "qwen35"]);
    }

    #[test]
    fn qwen3_from_gguf_synthetic_header() {
        let bytes = synthetic_qwen3();
        let (_dir, file) = open_bytes(&bytes);
        assert_eq!(file.header().architecture(), Some("qwen3"));

        let backend: Arc<dyn Backend> = Arc::new(CpuBackend);
        let model = architecture_for("qwen3")
            .unwrap()
            .from_gguf(&file, backend.clone())
            .expect("qwen3 from_gguf on synthetic header");
        assert_eq!(model.architecture(), "qwen3");
        assert_eq!(model.n_layer(), 1);
        assert_eq!(model.n_embd(), 8);
        assert_eq!(model.n_vocab(), 4);
        assert_eq!(model.n_ctx(), 8);
        assert_eq!(model.n_head(), 2);
        assert_eq!(model.n_head_kv(), 1);
        assert_eq!(model.head_dim(), 4);

        let layout = model.state_layout();
        assert_eq!(layout.n_layer, 1);
        assert_eq!(layout.n_ctx, 8);
        assert_eq!(layout.n_head_kv, 1);
        assert_eq!(layout.head_dim, 4);
        assert_eq!(layout.n_recurrent, 0);

        let loaded = load_model(&file, backend).expect("load_model dispatches on general.architecture");
        assert_eq!(loaded.architecture(), "qwen3");
    }

    #[test]
    fn qwen3_forward_prompt_then_decode() {
        let bytes = synthetic_qwen3();
        let (_dir, file) = open_bytes(&bytes);
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend);
        let model = load_model(&file, backend).unwrap();
        let mut state = SeqState::allocate(&model.state_layout());

        let logits1 = model
            .forward(&[1, 2], &[0, 1], &mut state)
            .expect("prompt forward");
        assert_eq!(state.n, 2);
        assert_eq!(logits1.n_vocab(), 4);
        assert!(logits1.as_slice().iter().all(|v| v.is_finite()));

        let logits2 = model
            .forward(&[3], &[2], &mut state)
            .expect("decode forward with KV cache");
        assert_eq!(state.n, 3);
        assert_eq!(logits2.n_vocab(), 4);
        assert!(logits2.as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn load_model_rejects_unregistered_architecture() {
        let mut meta = BTreeMap::new();
        meta.insert("general.architecture".into(), meta_str("llama"));
        let bytes = write_gguf(&meta, &[]).unwrap();
        let (_dir, file) = open_bytes(&bytes);
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend);
        let err = match load_model(&file, backend) {
            Ok(_) => panic!("expected unregistered architecture to fail"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("llama"), "{err}");
        assert!(err.contains("qwen3"), "{err}");
        assert!(err.contains("qwen35"), "{err}");
    }

    #[test]
    fn qwen35_from_gguf_synthetic_header() {
        let bytes = synthetic_qwen35();
        let (_dir, file) = open_bytes(&bytes);
        assert_eq!(file.header().architecture(), Some("qwen35"));

        let backend: Arc<dyn Backend> = Arc::new(CpuBackend);
        let model = architecture_for("qwen35")
            .unwrap()
            .from_gguf(&file, backend.clone())
            .expect("qwen35 from_gguf on synthetic header");
        assert_eq!(model.architecture(), "qwen35");
        assert_eq!(model.n_layer(), 2);
        assert_eq!(model.n_embd(), 8);
        assert_eq!(model.n_vocab(), 4);
        assert_eq!(model.n_ctx(), 8);
        assert_eq!(model.n_head(), 2);
        assert_eq!(model.n_head_kv(), 1);
        assert_eq!(model.head_dim(), 4);
        assert_eq!(model.mtp_loaded(), 1);

        let layout = model.state_layout();
        assert_eq!(layout.n_layer, 2);
        assert_eq!(layout.n_ctx, 8);
        assert_eq!(layout.n_head_kv, 1);
        assert_eq!(layout.head_dim, 4);
        assert_eq!(layout.n_recurrent, 1);
        assert_eq!(layout.recurrent_shape, [1, 4, 4]);

        let loaded = load_model(&file, backend).expect("load_model dispatches on general.architecture");
        assert_eq!(loaded.architecture(), "qwen35");
        assert_eq!(loaded.mtp_loaded(), 1);
    }

    #[test]
    fn qwen35_forward_prompt_then_decode() {
        let bytes = synthetic_qwen35();
        let (_dir, file) = open_bytes(&bytes);
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend);
        let model = load_model(&file, backend).unwrap();
        let mut state = SeqState::allocate(&model.state_layout());
        assert!(state.recurrent.iter().any(|r| r.is_some()));

        let logits1 = model
            .forward(&[1, 2], &[0, 1], &mut state)
            .expect("prompt forward");
        assert_eq!(state.n, 2);
        assert_eq!(logits1.n_vocab(), 4);
        assert!(logits1.as_slice().iter().all(|v| v.is_finite()));

        let logits2 = model
            .forward(&[3], &[2], &mut state)
            .expect("decode forward with KV + recurrent state");
        assert_eq!(state.n, 3);
        assert_eq!(logits2.n_vocab(), 4);
        assert!(logits2.as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn qwen3_architecture_for_still_resolves() {
        let arch = architecture_for("qwen3").expect("qwen3 must self-register via inventory");
        assert_eq!(arch.name(), "qwen3");
        assert_eq!(QWEN3.name(), "qwen3");
        assert_eq!(QWEN3_ARCH, "qwen3");
    }

    /// Incremental decode last-token logits must match a one-shot prompt (KV cache).
    #[test]
    fn qwen3_decode_logits_match_full_prompt() {
        let bytes = synthetic_qwen3();
        let (_dir, file) = open_bytes(&bytes);
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend);
        let model = load_model(&file, backend).unwrap();
        assert_eq!(model.architecture(), "qwen3");

        let mut full = SeqState::allocate(&model.state_layout());
        let logits_full = model
            .forward(&[1, 2, 3], &[0, 1, 2], &mut full)
            .expect("full prompt");
        assert_eq!(full.n, 3);
        assert_finite_logits(&logits_full, 4);

        let mut inc = SeqState::allocate(&model.state_layout());
        let _ = model
            .forward(&[1, 2], &[0, 1], &mut inc)
            .expect("prompt prefix");
        let logits_inc = model
            .forward(&[3], &[2], &mut inc)
            .expect("decode last token");
        assert_eq!(inc.n, 3);
        assert_finite_logits(&logits_inc, 4);

        let a = logits_full.as_slice();
        let b = logits_inc.as_slice();
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() <= 1e-4,
                "logit {i}: full={x} incremental={y}"
            );
        }
    }

    #[test]
    fn qwen3_two_layer_forward_prompt_then_decode() {
        let bytes = synthetic_qwen3_two_layer();
        let (_dir, file) = open_bytes(&bytes);
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend);
        let model = load_model(&file, backend).unwrap();
        assert_eq!(model.n_layer(), 2);
        assert_eq!(model.state_layout().n_layer, 2);
        assert_eq!(model.state_layout().n_recurrent, 0);
        let mut state = SeqState::allocate(&model.state_layout());
        assert_eq!(state.k.len(), 2);
        assert!(state.recurrent.iter().all(|r| r.is_none()));

        let logits1 = model
            .forward(&[1, 2], &[0, 1], &mut state)
            .expect("two-layer prompt");
        assert_eq!(state.n, 2);
        assert_finite_logits(&logits1, 4);

        let logits2 = model
            .forward(&[3], &[2], &mut state)
            .expect("two-layer decode");
        assert_eq!(state.n, 3);
        assert_finite_logits(&logits2, 4);
    }

    #[test]
    fn qwen3_tied_output_forward_finite() {
        let bytes = synthetic_qwen3_tied_output();
        let (_dir, file) = open_bytes(&bytes);
        assert!(file.tensor_info("output.weight").is_err());
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend);
        let model = load_model(&file, backend).expect("tied output falls back to token_embd");
        assert_eq!(model.n_vocab(), 4);
        let mut state = SeqState::allocate(&model.state_layout());
        let logits = model
            .forward(&[1, 2], &[0, 1], &mut state)
            .expect("tied-output forward");
        assert_finite_logits(&logits, 4);
    }

    #[test]
    fn qwen3_q8_0_q4_k_q6_k_load_and_forward() {
        let bytes = synthetic_qwen3_quant_mix();
        let (_dir, file) = open_bytes(&bytes);
        assert_eq!(
            file.tensor_info("blk.0.attn_q.weight").unwrap().ggml_type,
            TYPE_Q8_0
        );
        assert_eq!(
            file.tensor_info("blk.0.ffn_gate.weight").unwrap().ggml_type,
            TYPE_Q4_K
        );
        assert_eq!(
            file.tensor_info("blk.0.ffn_up.weight").unwrap().ggml_type,
            TYPE_Q6_K
        );
        assert_eq!(
            file.tensor_info("blk.0.ffn_down.weight").unwrap().ggml_type,
            TYPE_Q4_K
        );

        let backend: Arc<dyn Backend> = Arc::new(CpuBackend);
        let model = architecture_for("qwen3")
            .unwrap()
            .from_gguf(&file, backend)
            .expect("qwen3 loads Q8_0 / Q4_K / Q6_K weights");
        assert_eq!(model.architecture(), "qwen3");
        assert_eq!(model.n_embd(), 8);
        let mut state = SeqState::allocate(&model.state_layout());
        let logits1 = model
            .forward(&[1, 2], &[0, 1], &mut state)
            .expect("quant prompt");
        assert_eq!(state.n, 2);
        assert_finite_logits(&logits1, 4);
        let logits2 = model
            .forward(&[3], &[2], &mut state)
            .expect("quant decode");
        assert_eq!(state.n, 3);
        assert_finite_logits(&logits2, 4);
    }

    #[test]
    fn qwen3_rejects_empty_tokens_and_kv_overflow() {
        let bytes = synthetic_qwen3();
        let (_dir, file) = open_bytes(&bytes);
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend);
        let model = load_model(&file, backend).unwrap();
        let mut state = SeqState::allocate(&model.state_layout());
        let empty = model.forward(&[], &[], &mut state).unwrap_err().to_string();
        assert!(empty.contains("empty"), "{empty}");

        let fill = model
            .forward(&[0, 1, 2, 3, 0, 1, 2, 3], &[0, 1, 2, 3, 4, 5, 6, 7], &mut state)
            .expect("fill n_ctx=8");
        assert_eq!(state.n, 8);
        assert_finite_logits(&fill, 4);
        let overflow = model
            .forward(&[1], &[8], &mut state)
            .unwrap_err()
            .to_string();
        assert!(overflow.contains("overflow") || overflow.contains("n_ctx"), "{overflow}");
    }

    #[test]
    fn qwen3_rejects_moe_expert_count() {
        let mut meta = qwen3_toy_meta(1, 16);
        meta.insert("qwen3.expert_count".into(), MetaValue::U32(8));
        let mut tensors = vec![
            f32_write("token_embd.weight", vec![8, 4], 0.02),
            ones_write("output_norm.weight", vec![8]),
            f32_write("output.weight", vec![8, 4], 0.01),
        ];
        tensors.extend(qwen3_dense_layer_f32(0, 16));
        let bytes = write_gguf(&meta, &tensors).unwrap();
        let (_dir, file) = open_bytes(&bytes);
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend);
        let err = match architecture_for("qwen3")
            .unwrap()
            .from_gguf(&file, backend)
        {
            Ok(_) => panic!("expected moe load to fail"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("qwen3moe") || err.contains("expert_count"), "{err}");
    }
}
