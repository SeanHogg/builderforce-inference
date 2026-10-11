//! Memory estimation: how much VRAM/RAM a model needs for a given context size.
//!
//! This implements the math from PRD §5.7:
//! `need = Σ tensor bytes (exact, from the header) + KV(ctx) + recurrent_state + activations(batch) + runtime overhead`

use std::path::Path;

use anyhow::{anyhow, Result};
use bfi_gguf::{GgufFile, ModelHeader};

/// Memory breakdown for a model at a given context size.
#[derive(Debug, Clone)]
pub struct FitBreakdown {
    /// Total weight bytes (from header).
    pub weights: u64,
    /// KV cache for full-attention layers at this context.
    pub kv_cache: u64,
    /// Recurrent state for delta-net layers.
    pub recurrent_state: u64,
    /// Estimated activation memory.
    pub activations: u64,
    /// Runtime overhead (code, allocator, etc.).
    pub overhead: u64,
}

impl FitBreakdown {
    pub fn total(&self) -> u64 {
        self.weights + self.kv_cache + self.recurrent_state + self.activations + self.overhead
    }
}

/// Fit report: can this model+context fit in the given budget?
#[derive(Debug)]
pub struct FitReport {
    /// True if the model fits in the budget.
    pub fits: bool,
    /// Memory needed in bytes.
    pub need_bytes: u64,
    /// Available budget in bytes.
    pub budget_bytes: u64,
    /// Breakdown of memory usage.
    pub breakdown: FitBreakdown,
    /// Maximum context size that fits in the budget.
    pub max_ctx_that_fits: usize,
    /// Device: "cuda", "cpu", etc.
    pub device: String,
}

/// Estimate memory needed for a GGUF model at a given context size.
///
/// The formula:
/// - weights: from GGUF header (exact)
/// - KV(ctx) = full_attn_layers × 2 × n_kv_heads × head_dim × ctx × kv_dtype_bytes
/// - recurrent_state = delta_net_layers × per_head_state_size
/// - activations: estimated based on n_params and batch size
/// - overhead: ~100MB for runtime
pub fn fit(path: &Path, ctx: usize, budget_bytes: u64) -> Result<FitReport> {
    let gguf = GgufFile::open(path)?;
    let header = gguf.header();

    let weights = header.weight_bytes()?;
    let layout = model_layout(header)?;

    // Calculate KV cache: full-attention layers only
    // Each layer needs: 2 (K+V) × n_kv_heads × head_dim × ctx
    let kv_per_layer = 2 * layout.n_kv_heads * layout.head_dim * ctx * 4; // 4 bytes per f32
    let kv_cache = layout.full_attn_layers as u64 * kv_per_layer as u64;

    // Recurrent state: delta-net layers have fixed-size state
    let recurrent_state = layout.delta_net_layers as u64 * layout.recurrent_state_size as u64 * 4;

    // Activations: rough estimate based on model size
    // Assume activations ≈ 2x weights during forward pass
    let activations = weights * 2;

    // Overhead: runtime code + allocator + buffers
    let overhead = 100 * 1024 * 1024; // 100MB

    let breakdown = FitBreakdown {
        weights,
        kv_cache,
        recurrent_state,
        activations,
        overhead,
    };

    let need_bytes = breakdown.total();

    // Find max context that fits
    // This is a simple binary search would be better in production
    let max_ctx = find_max_ctx(&layout, weights, budget_bytes);

    Ok(FitReport {
        fits: need_bytes <= budget_bytes,
        need_bytes,
        budget_bytes,
        breakdown,
        max_ctx_that_fits: max_ctx,
        device: "cpu".to_string(),
    })
}

/// Model layout extracted from GGUF header.
#[derive(Debug)]
struct ModelLayout {
    full_attn_layers: usize,
    delta_net_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    recurrent_state_size: usize,
}

fn model_layout(header: &ModelHeader) -> Result<ModelLayout> {
    let arch = header.architecture().ok_or_else(|| anyhow!("missing architecture"))?;

    let block_count = header
        .get_u64(&format!("{}.block_count", arch))
        .unwrap_or(0) as usize;

    let n_embd = header
        .get_u64(&format!("{}.embedding_length", arch))
        .unwrap_or(0) as usize;

    let n_kv_heads = header
        .get_u64(&format!("{}.attention.head_count_kv", arch))
        .unwrap_or(1) as usize;

    let head_dim = header
        .get_u64(&format!("{}.attention.head_dim", arch))
        .unwrap_or((n_embd as u64) / (n_kv_heads.max(1) as u64)) as usize;

    // Note: M3 will add logic to detect delta-net layers from GGUF metadata.
    // Currently treating all layers as full-attention.
    let full_attn_layers = block_count;
    let delta_net_layers = 0;
    let recurrent_state_size = n_kv_heads * head_dim; // Per-layer recurrent state

    Ok(ModelLayout {
        full_attn_layers,
        delta_net_layers,
        n_kv_heads,
        head_dim,
        recurrent_state_size,
    })
}

fn find_max_ctx(layout: &ModelLayout, weights: u64, budget: u64) -> usize {
    // Binary search for the largest context that still fits.
    let mut low = 128usize;
    let mut high = 262144usize; // 256K max

    while low < high {
        let mid = (low + high + 1) / 2;
        let kv_per_layer = 2 * layout.n_kv_heads * layout.head_dim * mid * 4;
        let kv_cache = layout.full_attn_layers as u64 * kv_per_layer as u64;
        let need = weights + kv_cache + (weights * 2) + (100 * 1024 * 1024);

        if need <= budget {
            low = mid;
        } else {
            high = mid - 1;
        }
    }

    low
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_breakdown_total() {
        let b = FitBreakdown {
            weights: 1_000_000_000,
            kv_cache: 100_000_000,
            recurrent_state: 10_000_000,
            activations: 2_000_000_000,
            overhead: 100_000_000,
        };
        assert_eq!(b.total(), 3_210_000_000);
    }
}
