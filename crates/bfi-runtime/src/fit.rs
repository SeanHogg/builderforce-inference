//! Memory estimation: how much VRAM/RAM a model needs for a given context size.
//!
//! PRD §5.7:
//! `need = Σ tensor bytes (exact, from the header) + KV(ctx) + recurrent_state + activations(batch) + runtime overhead`
//!
//! - `KV(ctx) = full_attn_layers × 2 × n_kv_heads × head_dim × ctx × kv_dtype_bytes`
//! - `recurrent_state = delta_net_layers × n_kv_heads × head_dim × head_dim × kv_dtype_bytes`
//!   (gated-delta per-head state `[d, d]`, matching qwen35 `StateLayout.recurrent_shape`)
//!
//! Layer kinds come from `{arch}.attention.layer_types` when present. `qwen35` without
//! that key uses the 3+1 hybrid (`(il+1)%4==0` is full attention). Other archs default
//! to all full-attention. Header-only: no weight mapping, no network.

use std::path::Path;

use anyhow::{anyhow, Result};
use bfi_gguf::{MetaValue, ModelHeader};

/// f32 KV / recurrent / activation working set (matches `SeqState` f32 tensors).
const KV_DTYPE_BYTES: u64 = 4;
/// Runtime code + allocator + buffers. Kept as data (PRD: measured per backend later).
pub const RUNTIME_OVERHEAD_BYTES: u64 = 100 * 1024 * 1024;
/// Decode/prefill batch used for the activations term.
const BATCH: u64 = 1;
/// Reused f32 buffers per layer (residual, attn, gate, up) — inference, not 2× weights.
const ACT_BUFFERS: u64 = 4;
/// Upper bound for `max_ctx_that_fits` when KV is free or the header has no cap.
const MAX_CTX_SEARCH: usize = 262_144;

/// Memory breakdown for a model at a given context size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FitBreakdown {
    /// Total weight bytes (from header).
    pub weights: u64,
    /// KV cache for full-attention layers at this context.
    pub kv_cache: u64,
    /// Recurrent state for delta-net layers (does not grow with ctx).
    pub recurrent_state: u64,
    /// Estimated activation memory (batch=1 inference working set).
    pub activations: u64,
    /// Runtime overhead (code, allocator, etc.).
    pub overhead: u64,
}

impl FitBreakdown {
    pub fn total(&self) -> u64 {
        self.weights
            .saturating_add(self.kv_cache)
            .saturating_add(self.recurrent_state)
            .saturating_add(self.activations)
            .saturating_add(self.overhead)
    }
}

/// Fit report: can this model+context fit in the given budget?
#[derive(Debug, Clone, PartialEq, Eq)]
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

impl FitReport {
    /// PRD §5.7 estimate from a GGUF header (no weights, no I/O).
    pub fn from_header(
        header: &ModelHeader,
        ctx: usize,
        budget_bytes: u64,
        device: &str,
    ) -> Result<Self> {
        let weights = header
            .weight_bytes()
            .map_err(|e| anyhow!("tensor sizes from header: {e}"))?;
        let layout = model_layout(header)?;
        let breakdown = breakdown_at(&layout, weights, ctx);
        let need_bytes = breakdown.total();
        Ok(Self {
            fits: need_bytes <= budget_bytes,
            need_bytes,
            budget_bytes,
            max_ctx_that_fits: find_max_ctx(&layout, weights, budget_bytes),
            device: device.to_string(),
            breakdown,
        })
    }
}

/// Estimate memory needed for a local GGUF at a given context size.
///
/// Reads only the header (stops before tensor data). Device is `"cpu"`.
pub fn fit(path: &Path, ctx: usize, budget_bytes: u64) -> Result<FitReport> {
    let header = ModelHeader::open(path).map_err(|e| anyhow!("GGUF header {}: {e}", path.display()))?;
    FitReport::from_header(&header, ctx, budget_bytes, "cpu")
}

/// Model layout extracted from GGUF header.
#[derive(Debug, Clone)]
struct ModelLayout {
    full_attn_layers: usize,
    delta_net_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_embd: usize,
    n_layer: usize,
}

fn model_layout(header: &ModelHeader) -> Result<ModelLayout> {
    let arch = header
        .architecture()
        .ok_or_else(|| anyhow!("missing architecture"))?;

    let n_layer = header
        .get_u64(&format!("{arch}.block_count"))
        .unwrap_or(0) as usize;

    let n_embd = header
        .get_u64(&format!("{arch}.embedding_length"))
        .unwrap_or(0) as usize;

    let n_head = header
        .get_u64(&format!("{arch}.attention.head_count"))
        .unwrap_or(0) as usize;

    let n_kv_heads = header
        .get_u64(&format!("{arch}.attention.head_count_kv"))
        .map(|v| v as usize)
        .unwrap_or(n_head);

    let head_dim = header
        .get_u64(&format!("{arch}.attention.head_dim"))
        .or_else(|| header.get_u64(&format!("{arch}.attention.key_length")))
        .map(|v| v as usize)
        .unwrap_or_else(|| {
            if n_head > 0 {
                n_embd / n_head
            } else {
                0
            }
        });

    let (full_attn_layers, delta_net_layers) = count_layer_kinds(header, arch, n_layer);

    Ok(ModelLayout {
        full_attn_layers,
        delta_net_layers,
        n_kv_heads,
        head_dim,
        n_embd,
        n_layer,
    })
}

/// `true` = recurrent / gated-delta.
fn count_layer_kinds(header: &ModelHeader, arch: &str, n_layer: usize) -> (usize, usize) {
    let kinds = layer_is_recurrent(header, arch, n_layer);
    let delta = kinds.iter().filter(|r| **r).count();
    let full = n_layer.saturating_sub(delta);
    (full, delta)
}

fn layer_is_recurrent(header: &ModelHeader, arch: &str, n_layer: usize) -> Vec<bool> {
    let key = format!("{arch}.attention.layer_types");
    if let Some(v) = header.get(&key) {
        return parse_layer_types(v, n_layer, arch);
    }
    (0..n_layer)
        .map(|il| default_is_recurrent(arch, il))
        .collect()
}

fn parse_layer_types(v: &MetaValue, n_layer: usize, arch: &str) -> Vec<bool> {
    if let Some(ss) = v.as_string_array() {
        return (0..n_layer)
            .map(|i| match ss.get(i) {
                Some(s) => match layer_kind_from_str(s) {
                    Some(rec) => rec,
                    None => default_is_recurrent(arch, i),
                },
                None => default_is_recurrent(arch, i),
            })
            .collect();
    }
    if let MetaValue::Array(items) = v {
        return (0..n_layer)
            .map(|i| match items.get(i).and_then(|x| x.as_u64()) {
                Some(0) => false,
                Some(_) => true,
                None => default_is_recurrent(arch, i),
            })
            .collect();
    }
    (0..n_layer)
        .map(|il| default_is_recurrent(arch, il))
        .collect()
}

fn layer_kind_from_str(s: &str) -> Option<bool> {
    let s = s.to_ascii_lowercase();
    if matches!(
        s.as_str(),
        "linear_attention" | "recurrent" | "gated_delta" | "delta_net" | "ssm" | "linear"
    ) {
        Some(true)
    } else if matches!(
        s.as_str(),
        "full_attention" | "attention" | "full" | "mha" | "softmax"
    ) {
        Some(false)
    } else {
        None
    }
}

/// qwen35 3+1 hybrid: full iff `(il + 1) % 4 == 0`. Other archs: all full-attn.
fn default_is_recurrent(arch: &str, il: usize) -> bool {
    if arch == "qwen35" {
        (il + 1) % 4 != 0
    } else {
        false
    }
}

fn kv_bytes(layout: &ModelLayout, ctx: usize) -> u64 {
    (layout.full_attn_layers as u64)
        .saturating_mul(2)
        .saturating_mul(layout.n_kv_heads as u64)
        .saturating_mul(layout.head_dim as u64)
        .saturating_mul(ctx as u64)
        .saturating_mul(KV_DTYPE_BYTES)
}

fn state_bytes(layout: &ModelLayout) -> u64 {
    (layout.delta_net_layers as u64)
        .saturating_mul(layout.n_kv_heads as u64)
        .saturating_mul(layout.head_dim as u64)
        .saturating_mul(layout.head_dim as u64)
        .saturating_mul(KV_DTYPE_BYTES)
}

fn activations_bytes(layout: &ModelLayout) -> u64 {
    BATCH
        .saturating_mul(layout.n_layer as u64)
        .saturating_mul(layout.n_embd as u64)
        .saturating_mul(KV_DTYPE_BYTES)
        .saturating_mul(ACT_BUFFERS)
}

fn breakdown_at(layout: &ModelLayout, weights: u64, ctx: usize) -> FitBreakdown {
    FitBreakdown {
        weights,
        kv_cache: kv_bytes(layout, ctx),
        recurrent_state: state_bytes(layout),
        activations: activations_bytes(layout),
        overhead: RUNTIME_OVERHEAD_BYTES,
    }
}

fn need_at(layout: &ModelLayout, weights: u64, ctx: usize) -> u64 {
    breakdown_at(layout, weights, ctx).total()
}

fn find_max_ctx(layout: &ModelLayout, weights: u64, budget: u64) -> usize {
    if need_at(layout, weights, 0) > budget {
        return 0;
    }
    let mut low = 0usize;
    let mut high = MAX_CTX_SEARCH;
    while low < high {
        let mid = (low + high + 1) / 2;
        if need_at(layout, weights, mid) <= budget {
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
    use bfi_gguf::writer::{write_gguf, TensorWrite};
    use bfi_gguf::{lookup_type, TYPE_F32};
    use std::collections::BTreeMap;

    fn f32_tensor(name: &str, n: u64) -> TensorWrite {
        let mut data = Vec::new();
        for _ in 0..n {
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }
        TensorWrite {
            name: name.into(),
            dims: vec![n],
            ty: lookup_type(TYPE_F32).unwrap(),
            data,
        }
    }

    fn header_qwen3() -> ModelHeader {
        let mut meta = BTreeMap::new();
        meta.insert(
            "general.architecture".into(),
            MetaValue::String("qwen3".into()),
        );
        meta.insert("qwen3.block_count".into(), MetaValue::U32(4));
        meta.insert("qwen3.embedding_length".into(), MetaValue::U32(16));
        meta.insert("qwen3.attention.head_count".into(), MetaValue::U32(4));
        meta.insert("qwen3.attention.head_count_kv".into(), MetaValue::U32(2));
        meta.insert("qwen3.attention.key_length".into(), MetaValue::U32(8));
        let bytes = write_gguf(&meta, &[f32_tensor("token_embd.weight", 8)]).unwrap();
        ModelHeader::from_bytes(&bytes).unwrap()
    }

    fn header_qwen35_types() -> ModelHeader {
        let mut meta = BTreeMap::new();
        meta.insert(
            "general.architecture".into(),
            MetaValue::String("qwen35".into()),
        );
        meta.insert("qwen35.block_count".into(), MetaValue::U32(2));
        meta.insert("qwen35.embedding_length".into(), MetaValue::U32(8));
        meta.insert("qwen35.attention.head_count".into(), MetaValue::U32(2));
        meta.insert("qwen35.attention.head_count_kv".into(), MetaValue::U32(1));
        meta.insert("qwen35.attention.key_length".into(), MetaValue::U32(4));
        meta.insert(
            "qwen35.attention.layer_types".into(),
            MetaValue::Array(vec![
                MetaValue::String("linear_attention".into()),
                MetaValue::String("full_attention".into()),
            ]),
        );
        let bytes = write_gguf(&meta, &[f32_tensor("token_embd.weight", 8)]).unwrap();
        ModelHeader::from_bytes(&bytes).unwrap()
    }

    fn header_qwen35_hybrid_default() -> ModelHeader {
        let mut meta = BTreeMap::new();
        meta.insert(
            "general.architecture".into(),
            MetaValue::String("qwen35".into()),
        );
        meta.insert("qwen35.block_count".into(), MetaValue::U32(4));
        meta.insert("qwen35.embedding_length".into(), MetaValue::U32(16));
        meta.insert("qwen35.attention.head_count".into(), MetaValue::U32(4));
        meta.insert("qwen35.attention.head_count_kv".into(), MetaValue::U32(2));
        meta.insert("qwen35.attention.key_length".into(), MetaValue::U32(8));
        let bytes = write_gguf(&meta, &[f32_tensor("token_embd.weight", 8)]).unwrap();
        ModelHeader::from_bytes(&bytes).unwrap()
    }

    fn header_qwen35_int_types() -> ModelHeader {
        let mut meta = BTreeMap::new();
        meta.insert(
            "general.architecture".into(),
            MetaValue::String("qwen35".into()),
        );
        meta.insert("qwen35.block_count".into(), MetaValue::U32(2));
        meta.insert("qwen35.embedding_length".into(), MetaValue::U32(8));
        meta.insert("qwen35.attention.head_count".into(), MetaValue::U32(2));
        meta.insert("qwen35.attention.head_count_kv".into(), MetaValue::U32(1));
        meta.insert("qwen35.attention.key_length".into(), MetaValue::U32(4));
        meta.insert(
            "qwen35.attention.layer_types".into(),
            MetaValue::Array(vec![MetaValue::U32(1), MetaValue::U32(0)]),
        );
        let bytes = write_gguf(&meta, &[f32_tensor("token_embd.weight", 8)]).unwrap();
        ModelHeader::from_bytes(&bytes).unwrap()
    }

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

    #[test]
    fn qwen3_all_full_attn_kv_formula() {
        let h = header_qwen3();
        let ctx = 1024usize;
        let report = FitReport::from_header(&h, ctx, 10 * 1024 * 1024 * 1024, "cpu").unwrap();
        // KV = 4 × 2 × 2 × 8 × 1024 × 4 = 524_288
        assert_eq!(report.breakdown.kv_cache, 524_288);
        assert_eq!(report.breakdown.recurrent_state, 0);
        // activations = 1 × 4 × 16 × 4 × 4 = 1_024
        assert_eq!(report.breakdown.activations, 1_024);
        assert_eq!(report.breakdown.overhead, RUNTIME_OVERHEAD_BYTES);
        assert_eq!(report.breakdown.weights, 32);
        assert_eq!(
            report.need_bytes,
            32 + 524_288 + 0 + 1_024 + RUNTIME_OVERHEAD_BYTES
        );
        assert!(report.fits);
        assert_eq!(report.device, "cpu");
    }

    #[test]
    fn qwen35_layer_types_splits_kv_and_state() {
        let h = header_qwen35_types();
        let ctx = 8usize;
        let report = FitReport::from_header(&h, ctx, 10 * 1024 * 1024 * 1024, "cuda").unwrap();
        // 1 full layer: KV = 1 × 2 × 1 × 4 × 8 × 4 = 256
        assert_eq!(report.breakdown.kv_cache, 256);
        // 1 delta layer: state = 1 × 1 × 4 × 4 × 4 = 64
        assert_eq!(report.breakdown.recurrent_state, 64);
        assert_eq!(report.device, "cuda");
        assert!(report.fits);
    }

    #[test]
    fn qwen35_int_layer_types_match_strings() {
        let a = FitReport::from_header(&header_qwen35_types(), 8, u64::MAX, "cpu").unwrap();
        let b = FitReport::from_header(&header_qwen35_int_types(), 8, u64::MAX, "cpu").unwrap();
        assert_eq!(a.breakdown.kv_cache, b.breakdown.kv_cache);
        assert_eq!(a.breakdown.recurrent_state, b.breakdown.recurrent_state);
    }

    #[test]
    fn qwen35_default_hybrid_three_plus_one() {
        let h = header_qwen35_hybrid_default();
        let ctx = 1024usize;
        let report = FitReport::from_header(&h, ctx, u64::MAX, "cpu").unwrap();
        // layers 0,1,2 recurrent; 3 full → 1 full, 3 delta
        // KV = 1 × 2 × 2 × 8 × 1024 × 4 = 131_072
        assert_eq!(report.breakdown.kv_cache, 131_072);
        // state = 3 × 2 × 8 × 8 × 4 = 1_536
        assert_eq!(report.breakdown.recurrent_state, 1_536);
    }

    #[test]
    fn max_ctx_binary_search_and_zero_when_fixed_exceeds_budget() {
        let h = header_qwen3();
        let weights = 32u64;
        let layout = model_layout(&h).unwrap();
        let fixed = weights + state_bytes(&layout) + activations_bytes(&layout) + RUNTIME_OVERHEAD_BYTES;
        let kv_at_1000 = kv_bytes(&layout, 1000);
        let report = FitReport::from_header(&h, 10_000, fixed + kv_at_1000, "cpu").unwrap();
        assert!(!report.fits);
        assert_eq!(report.max_ctx_that_fits, 1000);

        let too_small = FitReport::from_header(&h, 1024, 1, "cpu").unwrap();
        assert!(!too_small.fits);
        assert_eq!(too_small.max_ctx_that_fits, 0);
    }

    #[test]
    fn fit_path_is_header_only() {
        let h = header_qwen3();
        let bytes = {
            let mut meta = BTreeMap::new();
            meta.insert(
                "general.architecture".into(),
                MetaValue::String("qwen3".into()),
            );
            meta.insert("qwen3.block_count".into(), MetaValue::U32(4));
            meta.insert("qwen3.embedding_length".into(), MetaValue::U32(16));
            meta.insert("qwen3.attention.head_count".into(), MetaValue::U32(4));
            meta.insert("qwen3.attention.head_count_kv".into(), MetaValue::U32(2));
            meta.insert("qwen3.attention.key_length".into(), MetaValue::U32(8));
            write_gguf(&meta, &[f32_tensor("token_embd.weight", 8)]).unwrap()
        };
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "bfi-fit-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&path, &bytes).unwrap();
        let from_path = fit(&path, 1024, 10 * 1024 * 1024 * 1024).unwrap();
        let _ = std::fs::remove_file(&path);
        let from_header = FitReport::from_header(&h, 1024, 10 * 1024 * 1024 * 1024, "cpu").unwrap();
        assert_eq!(from_path.need_bytes, from_header.need_bytes);
        assert_eq!(from_path.breakdown.kv_cache, from_header.breakdown.kv_cache);
        assert_eq!(from_path.device, "cpu");
    }
}
