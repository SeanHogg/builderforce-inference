//! `BfiEngine.support()`: what each backend can do here. Requests a WebGPU adapter only
//! — never a device — so hosts can call it on every capability check.
//!
//! The record keeps its version-2 top level (the WebGPU adapter's fields, as before) and
//! adds `backends`: `webgpu` (the same adapter facts) and `cpu` (always supported; SIMD
//! and thread count of this build, the KV budget and memory ceiling the CPU backend
//! loads under, and `navigator.deviceMemory` when the browser reports it).

use bfi_models::kv_plan::{cpu_memory_ceiling, DEFAULT_CPU_KV_BUDGET_BYTES, WEBGPU_MIN_BINDING_BYTES};
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

use crate::js;

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

/// The CPU backend: the same kernels the native `bfi` runs, in wasm memory.
#[derive(Debug, Clone)]
pub struct CpuInfo {
    /// Built with wasm SIMD128 (`.cargo/config.toml`).
    pub simd: bool,
    /// Threads the kernels run on: the rayon pool `initThreads` started in the threaded
    /// artifact, else 1.
    pub threads: u32,
    /// `navigator.deviceMemory` in bytes, when the browser reports it (Chromium; capped
    /// at 8 GiB by the spec). Hosts weigh a CPU model against it.
    pub device_memory_bytes: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct Support {
    pub webgpu: AdapterInfo,
    pub cpu: CpuInfo,
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

    fn backend_js(&self) -> JsValue {
        let mut pairs = vec![
            ("supported", JsValue::from_bool(self.supported)),
            ("maxBufferBytes", JsValue::from_f64(self.max_buffer_bytes)),
            ("maxStorageBufferBindingSizeBytes", JsValue::from_f64(self.max_binding_bytes)),
            ("f16", JsValue::from_bool(self.f16)),
        ];
        if let Some(r) = &self.reason {
            pairs.push(("reason", JsValue::from_str(r)));
        }
        js::object(&pairs)
    }
}

impl CpuInfo {
    fn backend_js(&self) -> JsValue {
        let mut pairs = vec![
            ("supported", JsValue::TRUE),
            ("simd", JsValue::from_bool(self.simd)),
            ("threads", JsValue::from_f64(self.threads as f64)),
            ("kvBudgetBytes", JsValue::from_f64(DEFAULT_CPU_KV_BUDGET_BYTES as f64)),
            // What the browser will actually grant: wasm32's cap, or half the device's
            // RAM (an unreported RAM reads as 2 GiB) — `kv_plan::cpu_memory_ceiling`.
            (
                "maxMemoryBytes",
                JsValue::from_f64(cpu_memory_ceiling(self.device_memory_bytes.map(|b| b as u64)) as f64),
            ),
        ];
        if let Some(m) = self.device_memory_bytes {
            pairs.push(("deviceMemoryBytes", JsValue::from_f64(m)));
        }
        js::object(&pairs)
    }
}

impl Support {
    /// Version 2 at the top level (the WebGPU adapter — unchanged for older hosts), plus
    /// `backends`.
    pub fn to_js(&self) -> JsValue {
        let w = &self.webgpu;
        let mut pairs = vec![
            ("contractVersion", JsValue::from_f64(2.0)),
            ("supported", JsValue::from_bool(w.supported)),
            ("maxBufferSizeBytes", JsValue::from_f64(w.max_buffer_bytes)),
            ("maxStorageBufferBindingSizeBytes", JsValue::from_f64(w.max_binding_bytes)),
            ("maxBufferBytes", JsValue::from_f64(w.max_buffer_bytes)),
            ("requestedPowerPreference", JsValue::from_str("high-performance")),
            ("f16", JsValue::from_bool(w.f16)),
        ];
        if let Some(r) = &w.reason {
            pairs.push(("reason", JsValue::from_str(r)));
        }
        pairs.push((
            "backends",
            js::object(&[("webgpu", w.backend_js()), ("cpu", self.cpu.backend_js())]),
        ));
        js::object(&pairs)
    }
}

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
