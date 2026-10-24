//! `BfiEngine.contextFor`: `bfi_models::kv_plan::served_context` (and
//! `cpu_served_context` for `backend: 'cpu'`) over a caller-supplied shape, so a host can
//! predict the served context before downloading a model — the same function `load`
//! plans with.

use bfi_models::kv_plan::{cpu_served_context, kv_buffer_limit, served_context, KvShape, DEFAULT_KV_BUDGET_BYTES};
use serde_json::Value;

use crate::host::HostProfile;

fn num(o: &Value, key: &str) -> Option<f64> {
    o.get(key).and_then(Value::as_f64).filter(|v| v.is_finite() && *v >= 0.0)
}

/// Missing / invalid shape fields yield 0.
///
/// The per-buffer cap is [`kv_buffer_limit`] — the same `min(binding, buffer)` `load`
/// applies to the adapter — over:
/// - binding: `maxStorageBufferBindingSizeBytes`;
/// - buffer: `maxBufferSizeBytes`, else the deprecated `maxBufferBytes`.
///
/// `maxBufferBytes` is the alias `BfiSupport` has always carried, where it means
/// `maxBufferSize`; older callers passed it here alone as the binding limit, and alone it
/// is still used as the cap. When a binding limit is also given it is min-ed in, so
/// `contextFor` never predicts more than `load` serves (`{ ...support, ...shape }`
/// predicts exactly what `load` plans).
pub fn context_for(o: &Value, profile: &HostProfile) -> usize {
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
    if o.get("backend").and_then(Value::as_str) == Some("cpu") {
        // What a CPU `load` serves: f32 KV, no storage-binding cap.
        return cpu_served_context(&shape, budget.unwrap_or(profile.cpu_kv_budget_bytes));
    }
    let budget = budget.unwrap_or(DEFAULT_KV_BUDGET_BYTES);
    let binding = num(o, "maxStorageBufferBindingSizeBytes").map(|v| v as u64);
    let buffer = num(o, "maxBufferSizeBytes")
        .or_else(|| num(o, "maxBufferBytes"))
        .map(|v| v as u64);
    let max_buffer = kv_buffer_limit(binding, buffer);
    let f16 = o.get("f16").and_then(Value::as_bool).unwrap_or(false);
    served_context(&shape, f16, budget, max_buffer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const BONSAI_1_7B: (u64, u64, u64, u64) = (28, 8, 128, 32_768);

    fn shape(extra: Value) -> Value {
        let (l, k, h, t) = BONSAI_1_7B;
        let mut v = json!({ "layers": l, "kvHeads": k, "headDim": h, "trainedContext": t });
        for (key, val) in extra.as_object().unwrap() {
            v[key] = val.clone();
        }
        v
    }

    #[test]
    fn matches_the_documented_predictions() {
        // README: 18688 with f16, 9216 without; CPU 2304 under wasm's 512 MiB budget.
        assert_eq!(context_for(&shape(json!({ "f16": true })), &HostProfile::WASM), 18_688);
        assert_eq!(context_for(&shape(json!({})), &HostProfile::WASM), 9_216);
        assert_eq!(context_for(&shape(json!({ "backend": "cpu" })), &HostProfile::WASM), 2_304);
    }

    #[test]
    fn a_native_cpu_plans_on_the_larger_budget() {
        let native = HostProfile::native(16 << 30);
        assert!(context_for(&shape(json!({ "backend": "cpu" })), &native) > 2_304);
    }

    #[test]
    fn an_incomplete_shape_predicts_nothing() {
        assert_eq!(context_for(&json!({ "layers": 28 }), &HostProfile::WASM), 0);
    }
}
