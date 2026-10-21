//! How many tokens of KV cache a device can serve. Pure arithmetic, no GPU:
//! the browser engine sizes its cache with it, and the TypeScript side calls the
//! same function (`BfiEngine.contextFor`) to predict the served context before
//! downloading a model.
//!
//! ```text
//! bytes_per_elem = f16 ? 2 : 4
//! per_token      = n_layer · 2 · n_head_kv · head_dim · bytes_per_elem   (K and V, every layer)
//! per_buffer_row = n_head_kv · head_dim · bytes_per_elem                 (one layer's K or V)
//! ctx = min(trained_ctx,
//!           floor(kv_budget_bytes / per_token),
//!           floor(max_buffer_bytes / per_buffer_row))
//! ctx = ctx ≥ 256 ? floor(ctx / 256) · 256 : ctx
//! ```
//!
//! The budget caps *total* KV memory (each layer's K and V are separate
//! buffers); `max_buffer_bytes` is the largest single storage binding the
//! device allows (`min(maxStorageBufferBindingSize, maxBufferSize)`, see
//! [`kv_buffer_limit`]).

/// Default total KV-cache budget (2 GiB).
pub const DEFAULT_KV_BUDGET_BYTES: u64 = 2 << 30;
/// WebGPU's guaranteed `maxStorageBufferBindingSize` (128 MiB).
pub const WEBGPU_MIN_BINDING_BYTES: u64 = 128 << 20;
/// Served contexts are rounded down to a multiple of this.
pub const CONTEXT_STEP: usize = 256;
/// Allocation retries halve the context down to this floor (or the trained
/// context, if smaller), never below.
pub const CONTEXT_FLOOR: usize = 2048;

/// The KV-relevant shape of a decoder (from GGUF metadata).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvShape {
    pub n_layer: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub trained_ctx: usize,
}

pub fn kv_elem_bytes(f16: bool) -> u64 {
    if f16 {
        2
    } else {
        4
    }
}

/// K + V bytes per token across all layers.
pub fn kv_bytes_per_token(s: &KvShape, f16: bool) -> u64 {
    (s.n_layer * 2 * s.n_head_kv * s.head_dim) as u64 * kv_elem_bytes(f16)
}

/// Bytes of one layer's K (or V) buffer at `ctx` tokens.
pub fn kv_buffer_bytes(s: &KvShape, f16: bool, ctx: usize) -> u64 {
    (ctx * s.n_head_kv * s.head_dim) as u64 * kv_elem_bytes(f16)
}

/// The context the engine will try first (see the module formula).
pub fn served_context(s: &KvShape, f16: bool, kv_budget_bytes: u64, max_buffer_bytes: u64) -> usize {
    let per_token = kv_bytes_per_token(s, f16).max(1);
    let per_row = kv_buffer_bytes(s, f16, 1).max(1);
    let cap = (kv_budget_bytes / per_token).min(max_buffer_bytes / per_row);
    let cap = (cap.min(usize::MAX as u64) as usize).min(s.trained_ctx);
    round_ctx(cap)
}

fn round_ctx(n: usize) -> usize {
    if n >= CONTEXT_STEP {
        n / CONTEXT_STEP * CONTEXT_STEP
    } else {
        n.max(1)
    }
}

/// Default KV budget on the CPU backend (512 MiB). The cache lives in wasm memory next
/// to the packed weights, so it is budgeted far tighter than VRAM.
pub const DEFAULT_CPU_KV_BUDGET_BYTES: u64 = 512 << 20;
/// wasm32's linear-memory ceiling (65 536 pages of 64 KiB). Weights, KV cache and the
/// working set all live inside it on the CPU backend.
pub const WASM32_MEMORY_BYTES: u64 = 4 << 30;
/// Headroom kept for activations, the tokenizer, the one tensor being streamed in and
/// allocator slack, on top of weights + KV.
pub const CPU_WORKING_SET_BYTES: u64 = 256 << 20;

/// The context the CPU backend serves: f32 KV (no per-buffer cap — host memory has no
/// storage-binding limit) under `kv_budget_bytes`, never above the trained context.
pub fn cpu_served_context(s: &KvShape, kv_budget_bytes: u64) -> usize {
    served_context(s, false, kv_budget_bytes, u64::MAX)
}

/// Memory the CPU backend needs to hold a model: its packed weights, a `ctx`-token f32
/// KV cache and [`CPU_WORKING_SET_BYTES`]. Compared against [`WASM32_MEMORY_BYTES`].
pub fn cpu_memory_bytes(weight_bytes: u64, s: &KvShape, ctx: usize) -> u64 {
    weight_bytes + kv_bytes_per_token(s, false) * ctx as u64 + CPU_WORKING_SET_BYTES
}

/// The largest single KV buffer the device serves: `min(binding, buffer)` of
/// whichever limits are known, [`WEBGPU_MIN_BINDING_BYTES`] when neither is.
/// Both the engine (adapter limits) and `contextFor` (caller-supplied limits)
/// go through this, so the prediction and the load agree.
pub fn kv_buffer_limit(binding_bytes: Option<u64>, buffer_bytes: Option<u64>) -> u64 {
    match (binding_bytes.filter(|b| *b > 0), buffer_bytes.filter(|b| *b > 0)) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => WEBGPU_MIN_BINDING_BYTES,
    }
}

/// Contexts to try, largest first: `start`, then halvings, ending at
/// `min(CONTEXT_FLOOR, trained_ctx)`. A `start` already below that floor (a
/// small budget or binding) is tried as-is: `[start]`. The first entry is
/// always what [`served_context`] predicted, so `contextFor` matches what
/// `load` serves when the first allocation succeeds.
pub fn retry_contexts(start: usize, s: &KvShape) -> Vec<usize> {
    let floor = CONTEXT_FLOOR.min(s.trained_ctx).max(1);
    let mut out = Vec::new();
    let mut c = start.min(s.trained_ctx).max(1);
    if c <= floor {
        out.push(c);
        return out;
    }
    while c > floor {
        out.push(c);
        c = round_ctx(c / 2).max(floor);
    }
    out.push(floor);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shapes from the HF configs of prism-ml/Bonsai-{8B,4B,1.7B}-unpacked
    // (num_hidden_layers, num_key_value_heads, head_dim, max_position_embeddings).
    /// Bonsai 8B: 36 layers, 8 KV heads, head_dim 128, 65 536 trained.
    const B8: KvShape = KvShape { n_layer: 36, n_head_kv: 8, head_dim: 128, trained_ctx: 65_536 };
    /// Bonsai 4B: 36 layers, 8 KV heads, head_dim 128, 32 768 trained.
    const B4: KvShape = KvShape { n_layer: 36, n_head_kv: 8, head_dim: 128, trained_ctx: 32_768 };
    /// Bonsai 1.7B: 28 layers, 8 KV heads, head_dim 128, 32 768 trained.
    const B17: KvShape = KvShape { n_layer: 28, n_head_kv: 8, head_dim: 128, trained_ctx: 32_768 };

    #[test]
    fn bonsai_served_contexts_at_default_budget() {
        let b = DEFAULT_KV_BUDGET_BYTES;
        let m = WEBGPU_MIN_BINDING_BYTES;
        assert_eq!(kv_bytes_per_token(&B8, true), 147_456);
        assert_eq!(served_context(&B8, true, b, m), 14_336);
        assert_eq!(served_context(&B8, false, b, m), 7_168);
        assert_eq!(served_context(&B4, true, b, m), 14_336);
        assert_eq!(served_context(&B4, false, b, m), 7_168);
        assert_eq!(served_context(&B17, true, b, m), 18_688);
        assert_eq!(served_context(&B17, false, b, m), 9_216);
    }

    #[test]
    fn trained_context_and_binding_cap() {
        let small = KvShape { trained_ctx: 4096, ..B17 };
        assert_eq!(served_context(&small, true, u64::MAX, u64::MAX), 4096);
        // 1 MiB binding / (8·128·2 B per row) = 512 tokens.
        assert_eq!(served_context(&B8, true, u64::MAX, 1 << 20), 512);
    }

    #[test]
    fn retries_halve_to_the_floor() {
        assert_eq!(retry_contexts(14_336, &B8), vec![14_336, 7_168, 3_584, 2_048]);
        assert_eq!(retry_contexts(2_048, &B8), vec![2_048]);
        let tiny = KvShape { trained_ctx: 32, ..B8 };
        assert_eq!(retry_contexts(32, &tiny), vec![32]);
    }

    #[test]
    fn start_below_the_floor_is_tried_as_is() {
        // A small budget plans under 2 048: that context is still attempted.
        assert_eq!(retry_contexts(1_000, &B8), vec![1_000]);
        assert_eq!(retry_contexts(1, &B8), vec![1]);
        let planned = served_context(&B8, true, 64 << 20, WEBGPU_MIN_BINDING_BYTES);
        assert!(planned < CONTEXT_FLOOR);
        assert_eq!(retry_contexts(planned, &B8).first(), Some(&planned));
    }

    #[test]
    fn first_retry_is_the_prediction() {
        for (budget, f16) in [(DEFAULT_KV_BUDGET_BYTES, true), (DEFAULT_KV_BUDGET_BYTES, false), (16 << 20, true)] {
            let planned = served_context(&B17, f16, budget, WEBGPU_MIN_BINDING_BYTES);
            assert_eq!(retry_contexts(planned, &B17)[0], planned);
        }
    }

    #[test]
    fn bonsai_cpu_contexts_and_memory_at_the_default_budget() {
        let b = DEFAULT_CPU_KV_BUDGET_BYTES;
        // f32 KV: 1.7B is 229 376 B/token, 4B / 8B 294 912 B/token.
        assert_eq!(cpu_served_context(&B17, b), 2_304);
        assert_eq!(cpu_served_context(&B4, b), 1_792);
        assert_eq!(cpu_served_context(&B8, b), 1_792);
        // Packed weights + KV + working set: every Bonsai fits wasm32's 4 GiB.
        for (weights, shape) in [(248_302_272u64, B17), (572_270_624, B4), (1_158_654_496, B8)] {
            let need = cpu_memory_bytes(weights, &shape, cpu_served_context(&shape, b));
            assert!(need < WASM32_MEMORY_BYTES, "{need}");
        }
        // A budget past the ceiling is what `load` refuses.
        let huge = cpu_served_context(&B8, 8 << 30);
        assert!(cpu_memory_bytes(1_158_654_496, &B8, huge) > WASM32_MEMORY_BYTES);
    }

    #[test]
    fn buffer_limit_is_the_min_of_known_limits() {
        assert_eq!(kv_buffer_limit(Some(256 << 20), Some(1 << 30)), 256 << 20);
        assert_eq!(kv_buffer_limit(Some(1 << 30), Some(256 << 20)), 256 << 20);
        assert_eq!(kv_buffer_limit(Some(512), None), 512);
        assert_eq!(kv_buffer_limit(None, Some(512)), 512);
        assert_eq!(kv_buffer_limit(Some(0), None), WEBGPU_MIN_BINDING_BYTES);
        assert_eq!(kv_buffer_limit(None, None), WEBGPU_MIN_BINDING_BYTES);
    }
}
