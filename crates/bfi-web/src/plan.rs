//! `BfiEngine.contextFor`: the JS face of `bfi_models::kv_plan::served_context`,
//! so the host can predict the served context before downloading a model.

use bfi_models::kv_plan::{served_context, KvShape, DEFAULT_KV_BUDGET_BYTES, WEBGPU_MIN_BINDING_BYTES};
use js_sys::Object;

use crate::js;

fn num(o: &Object, key: &str) -> Option<f64> {
    js::get(o, key).as_f64().filter(|v| v.is_finite() && *v >= 0.0)
}

/// `{ layers, kvHeads, headDim, trainedContext, kvBudgetBytes?, f16?, maxBufferBytes? }` → tokens.
/// Missing / invalid shape fields yield 0.
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
    let budget = num(o, "kvBudgetBytes").map(|v| v as u64).unwrap_or(DEFAULT_KV_BUDGET_BYTES);
    let max_buffer = num(o, "maxBufferBytes").map(|v| v as u64).unwrap_or(WEBGPU_MIN_BINDING_BYTES);
    let f16 = js::get(o, "f16").as_bool().unwrap_or(false);
    served_context(&shape, f16, budget, max_buffer)
}
