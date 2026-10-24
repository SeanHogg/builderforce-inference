//! `BfiEngine.support()` in a browser: the WebGPU adapter (requested only — never a
//! device — so hosts can call it on every capability check) and `navigator.deviceMemory`,
//! in `bfi-engine`'s record shape.

use bfi_engine::{AdapterInfo, CpuInfo, HostProfile, Support};
use bfi_models::kv_plan::{cpu_memory_ceiling, WEBGPU_MIN_BINDING_BYTES};
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

use crate::js;

pub async fn probe() -> Support {
    Support {
        webgpu: probe_webgpu().await,
        cpu: probe_cpu(),
    }
}

fn probe_cpu() -> CpuInfo {
    let navigator = js::get(&js_sys::global(), "navigator");
    let device_memory_bytes = js::present(&navigator)
        .then(|| js::get(&navigator, "deviceMemory").as_f64())
        .flatten()
        .filter(|gb| gb.is_finite() && *gb > 0.0)
        .map(|gb| gb * (1u64 << 30) as f64);
    CpuInfo {
        simd: cfg!(target_feature = "simd128"),
        threads: crate::threads::count(),
        kv_budget_bytes: HostProfile::WASM.cpu_kv_budget_bytes,
        // What the browser will actually grant: wasm32's cap, or half the device's RAM
        // (an unreported RAM reads as 2 GiB) — `kv_plan::cpu_memory_ceiling`.
        max_memory_bytes: cpu_memory_ceiling(device_memory_bytes.map(|b| b as u64)),
        device_memory_bytes,
    }
}

pub async fn probe_webgpu() -> AdapterInfo {
    if !cfg!(feature = "webgpu") {
        return AdapterInfo::unsupported("this bfi-web build is CPU-only (the threaded artifact)");
    }
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
    let opts: JsValue = js::to_js(&serde_json::json!({ "powerPreference": "high-performance" }));
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
        // WebGPU's guaranteed binding when the adapter reports none.
        .unwrap_or(WEBGPU_MIN_BINDING_BYTES as f64);
    AdapterInfo {
        supported: true,
        reason: None,
        max_buffer_bytes,
        max_binding_bytes,
        f16,
    }
}
