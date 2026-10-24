//! `BfiEngine.support()`'s record: what each backend can do here. The SHAPE is shared;
//! the facts are the host's — a browser asks `navigator.gpu` (adapter only, never a
//! device) and `navigator.deviceMemory`, a native host asks wgpu and the OS.
//!
//! The record keeps its version-2 top level (the WebGPU adapter's fields) and adds
//! `backends`: `webgpu` (the same adapter facts) and `cpu` (always supported; SIMD and
//! thread count, the KV budget and memory ceiling the CPU backend loads under, and the
//! device's RAM when the host can read it).

use serde_json::{json, Value};

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
    pub fn unsupported(reason: impl Into<String>) -> Self {
        Self {
            supported: false,
            reason: Some(reason.into()),
            max_buffer_bytes: 0.0,
            max_binding_bytes: 0.0,
            f16: false,
        }
    }

    fn backend_json(&self) -> Value {
        let mut v = json!({
            "supported": self.supported,
            "maxBufferBytes": self.max_buffer_bytes,
            "maxStorageBufferBindingSizeBytes": self.max_binding_bytes,
            "f16": self.f16,
        });
        if let Some(r) = &self.reason {
            v["reason"] = json!(r);
        }
        v
    }
}

/// The CPU backend: the native `bfi`'s kernels (in wasm memory, or natively).
#[derive(Debug, Clone)]
pub struct CpuInfo {
    /// Built with SIMD (wasm SIMD128, or the native target's vector units).
    pub simd: bool,
    /// Threads the kernels run on.
    pub threads: u32,
    /// The KV budget a CPU load uses by default ([`crate::HostProfile`]).
    pub kv_budget_bytes: u64,
    /// The memory the CPU backend may plan to use: weights + KV + working set must fit.
    pub max_memory_bytes: u64,
    /// The device's RAM in bytes, when the host can read it.
    pub device_memory_bytes: Option<f64>,
}

impl CpuInfo {
    fn backend_json(&self) -> Value {
        let mut v = json!({
            "supported": true,
            "simd": self.simd,
            "threads": self.threads,
            "kvBudgetBytes": self.kv_budget_bytes,
            "maxMemoryBytes": self.max_memory_bytes,
        });
        if let Some(m) = self.device_memory_bytes {
            v["deviceMemoryBytes"] = json!(m);
        }
        v
    }
}

#[derive(Debug, Clone)]
pub struct Support {
    pub webgpu: AdapterInfo,
    pub cpu: CpuInfo,
}

impl Support {
    /// Version 2 at the top level (the WebGPU adapter — unchanged for older hosts), plus
    /// `backends`.
    pub fn to_json(&self) -> Value {
        let w = &self.webgpu;
        let mut v = json!({
            "contractVersion": 2,
            "supported": w.supported,
            "maxBufferSizeBytes": w.max_buffer_bytes,
            "maxStorageBufferBindingSizeBytes": w.max_binding_bytes,
            "maxBufferBytes": w.max_buffer_bytes,
            "requestedPowerPreference": "high-performance",
            "f16": w.f16,
            "backends": { "webgpu": w.backend_json(), "cpu": self.cpu.backend_json() },
        });
        if let Some(r) = &w.reason {
            v["reason"] = json!(r);
        }
        v
    }
}
