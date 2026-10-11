//! Architecture registry for bfi.
//!
//! Architectures self-register through [`inventory`] ([`ArchReg`]). A new model
//! is one module plus an `inventory::submit!` — the engine is not edited.
//!
//! M2: `qwen3` (dense, all full attention, Q/K-norm per head, GQA, SwiGLU).
//! M3: `qwen35` (Bonsai): hybrid gated-delta + full attention, multi-section RoPE.

mod arch;
mod qwen3;
mod qwen35;
mod state;
mod tutil;

pub use arch::{architecture_for, load_model, ArchReg, Architecture, Model};
pub use qwen3::{Qwen3Arch, ARCH_NAME as QWEN3_ARCH, QWEN3};
pub use qwen35::{Qwen35Arch, ARCH_NAME as QWEN35_ARCH, QWEN35};
pub use state::{Logits, SeqState, StateLayout};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use bfi_core::{Backend, CpuBackend};
    use bfi_gguf::writer::{write_gguf, TensorWrite};
    use bfi_gguf::{lookup_type, GgufFile, MetaValue, TYPE_F32};

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

    #[test]
    fn registry_lookup_by_architecture_name() {
        let arch = architecture_for("qwen3").expect("qwen3 must self-register via inventory");
        assert_eq!(arch.name(), "qwen3");
        assert_eq!(QWEN3.name(), "qwen3");
        let q35 = architecture_for("qwen35").expect("qwen35 must self-register via inventory");
        assert_eq!(q35.name(), "qwen35");
        assert_eq!(QWEN35.name(), "qwen35");
        assert!(architecture_for("llama").is_none());
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
}
