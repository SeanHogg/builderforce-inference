//! `BfiEngine.contextFor`: the JS face of `bfi_models::kv_plan::served_context` (and
//! `cpu_served_context` for `backend: 'cpu'`), so the host can predict the served
//! context before downloading a model.

use bfi_models::kv_plan::{
    cpu_served_context, kv_buffer_limit, served_context, KvShape, DEFAULT_CPU_KV_BUDGET_BYTES,
    DEFAULT_KV_BUDGET_BYTES,
};
use js_sys::Object;

use crate::js;

fn num(o: &Object, key: &str) -> Option<f64> {
    js::get(o, key).as_f64().filter(|v| v.is_finite() && *v >= 0.0)
}

/// Missing / invalid shape fields yield 0.
///
/// The per-buffer cap is [`kv_buffer_limit`] — the same `min(binding, buffer)`
/// `load` applies to the adapter — over:
/// - binding: `maxStorageBufferBindingSizeBytes`;
/// - buffer: `maxBufferSizeBytes`, else the deprecated `maxBufferBytes`.
///
/// `maxBufferBytes` is the alias `BfiSupport` has always carried, where it means
/// `maxBufferSize`; older callers passed it here alone as the binding limit,
/// and alone it is still used as the cap (unchanged behaviour). When a binding
/// limit is also given it is min-ed in, so `contextFor` never predicts more than
/// `load` serves (`{ ...support, ...shape }` predicts exactly what `load` plans).
pub fn context_for_js(o: &Object) -> usize {
    let (Some(layers), Some(kv_heads), Some(head_dim), Some(trained)) = (
        num(o, "layers"),
        num(o, "kvHeads"),
        num(o, "headDim"),
        num(o, "trainedContext"),
    ) else {
        return 0;
    };
    let shape = KvShape {
        n_layer: layers as usize,
        n_head_kv: kv_heads as usize,
        head_dim: head_dim as usize,
        trained_ctx: trained as usize,
    };
    let budget = num(o, "kvBudgetBytes").map(|v| v as u64);
    if js::get(o, "backend").as_string().as_deref() == Some("cpu") {
        // What a CPU `load` serves: f32 KV, no storage-binding cap.
        return cpu_served_context(&shape, budget.unwrap_or(DEFAULT_CPU_KV_BUDGET_BYTES));
    }
    let budget = budget.unwrap_or(DEFAULT_KV_BUDGET_BYTES);
    let binding = num(o, "maxStorageBufferBindingSizeBytes").map(|v| v as u64);
    let buffer = num(o, "maxBufferSizeBytes")
        .or_else(|| num(o, "maxBufferBytes"))
        .map(|v| v as u64);
    let max_buffer = kv_buffer_limit(binding, buffer);
    let f16 = js::get(o, "f16").as_bool().unwrap_or(false);
    served_context(&shape, f16, budget, max_buffer)
}
