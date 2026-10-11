//! Sequence state the runtime allocates from [`StateLayout`].
//!
//! qwen3 uses a per-layer KV cache only (`n_recurrent = 0`). qwen35 fills
//! [`SeqState::recurrent`] for gated-delta layers (`n_recurrent > 0`).

use bfi_core::Tensor;

use crate::tutil;

/// Last-token logits, length `n_vocab`.
#[derive(Clone, Debug)]
pub struct Logits {
    pub values: Vec<f32>,
}

impl Logits {
    pub fn n_vocab(&self) -> usize {
        self.values.len()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }
}

/// What [`crate::Model::state_layout`] tells the runtime to allocate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateLayout {
    pub n_layer: usize,
    /// KV capacity (context length).
    pub n_ctx: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    /// Gated-delta layers. Always 0 for qwen3.
    pub n_recurrent: usize,
    /// `[n_kv_heads, head_dim, head_dim]` per recurrent layer. Unused for qwen3.
    pub recurrent_shape: [usize; 3],
}

/// Per-sequence KV (and later recurrent) state.
pub struct SeqState {
    /// Tokens already committed to the cache.
    pub n: usize,
    /// Per full-attention layer: `[n_ctx, n_head_kv, head_dim]`.
    pub k: Vec<Tensor>,
    pub v: Vec<Tensor>,
    /// Per-layer recurrent state; `None` for full-attention layers.
    pub recurrent: Vec<Option<Tensor>>,
}

impl SeqState {
    pub fn allocate(layout: &StateLayout) -> Self {
        let kv_shape = vec![layout.n_ctx, layout.n_head_kv, layout.head_dim];
        let k = (0..layout.n_layer)
            .map(|_| tutil::zeros_f32(kv_shape.clone()))
            .collect();
        let v = (0..layout.n_layer)
            .map(|_| tutil::zeros_f32(kv_shape.clone()))
            .collect();
        let rec_shape = layout.recurrent_shape;
        let recurrent = (0..layout.n_layer)
            .map(|_| {
                if layout.n_recurrent > 0 && rec_shape.iter().all(|&d| d > 0) {
                    Some(tutil::zeros_f32(rec_shape.to_vec()))
                } else {
                    None
                }
            })
            .collect();
        Self {
            n: 0,
            k,
            v,
            recurrent,
        }
    }
}
