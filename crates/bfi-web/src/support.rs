//! `BfiEngine.support()`: WebGPU adapter probe. Requests an adapter only —
//! never a device — so hosts can call it on every capability check.

use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

use crate::js;

/// WebGPU's guaranteed `maxStorageBufferBindingSize` when an adapter reports none.
const WEBGPU_MIN_BINDING_BYTES: f64 = 128.0 * 1024.0 * 1024.0;

#[derive(Debug, Clone)]
pub struct AdapterInfo {
    pub supported: bool,
    pub reason: Option<String>,
    /// Adapter `limits.maxBufferSize`.
    pub max_buffer_bytes: f64,
    /// Adapter `limits.maxStorageBufferBindingSize` (sizes the KV cache).
    pub max_binding_bytes: f64,
    /// Adapter offers `shader-f16`.
    pub f16: bool,
}

impl AdapterInfo {
    fn unsupported(reason: impl Into<String>) -> Self {
        Self {
            supported: false,
            reason: Some(reason.into()),
            max_buffer_bytes: 0.0,
            max_binding_bytes: 0.0,
            f16: false,
        }
    }

    /// Version 2 keeps allocation and storage-binding limits explicit.
    pub fn to_js(&self) -> JsValue {
        let mut pairs = vec![
            ("contractVersion", JsValue::from_f64(2.0)),
            ("supported", JsValue::from_bool(self.supported)),
            ("maxBufferSizeBytes", JsValue::from_f64(self.max_buffer_bytes)),
            ("maxStorageBufferBindingSizeBytes", JsValue::from_f64(self.max_binding_bytes)),
            ("maxBufferBytes", JsValue::from_f64(self.max_buffer_bytes)),
            ("requestedPowerPreference", JsValue::from_str("high-performance")),
            ("f16", JsValue::from_bool(self.f16)),
        ];
        if let Some(r) = &self.reason {
            pairs.push(("reason", JsValue::from_str(r)));
        }
        js::object(&pairs)
    }
}

pub async fn probe() -> AdapterInfo {
    let global = js_sys::global();
    let navigator = js::get(&global, "navigator");
    if !js::present(&navigator) {
        return AdapterInfo::unsupported("no navigator in this context");
    }
    let gpu = js::get(&navigator, "gpu");
    if !js::present(&gpu) {
        return AdapterInfo::unsupported(
            "navigator.gpu is missing (browser without WebGPU, or not a secure context)",
        );
    }
    let opts = js::object(&[("powerPreference", JsValue::from_str("high-performance"))]);
    let promise = match js::call(&gpu, "requestAdapter", &[opts]) {
        Ok(p) => p,
        Err(e) => return AdapterInfo::unsupported(format!("requestAdapter threw: {e:?}")),
    };
    let adapter = match JsFuture::from(js_sys::Promise::from(promise)).await {
        Ok(a) => a,
        Err(e) => return AdapterInfo::unsupported(format!("requestAdapter rejected: {e:?}")),
    };
    if !js::present(&adapter) {
        return AdapterInfo::unsupported("no WebGPU adapter (blocklisted GPU or disabled WebGPU)");
    }
    let features = js::get(&adapter, "features");
    let f16 = js::call(&features, "has", &[JsValue::from_str("shader-f16")])
        .ok()
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let limits = js::get(&adapter, "limits");
    let max_buffer_bytes = js::get(&limits, "maxBufferSize").as_f64().unwrap_or(0.0);
    let max_binding_bytes = js::get(&limits, "maxStorageBufferBindingSize")
        .as_f64()
        .unwrap_or(WEBGPU_MIN_BINDING_BYTES);
    AdapterInfo {
        supported: true,
        reason: None,
        max_buffer_bytes,
        max_binding_bytes,
        f16,
    }
}
