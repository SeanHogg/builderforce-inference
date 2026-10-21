//! Which backend an engine runs on, chosen once at `BfiEngine.create`.
//!
//! - `webgpu`: the GPU-resident engine (`Qwen3Resident`); rejects `webgpu_unavailable`
//!   without an adapter.
//! - `cpu`: the host-tensor engine (`Qwen3StreamLoader` → `Session`) on bfi-core's
//!   `CpuBackend` — the native `bfi`'s kernels in wasm memory, SIMD128.
//! - `auto` (default): WebGPU when an adapter exists, else CPU.

use bfi_core::resident::{open_webgpu, Gpu};
use js_sys::Object;
use wasm_bindgen::JsValue;

use crate::errors::{error, Kind};
use crate::js;
use crate::support::{self, AdapterInfo};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    Webgpu,
    Cpu,
    Auto,
}

impl BackendChoice {
    /// `create({ backend })`; absent = `auto`, anything else unknown = `invalid_request`.
    pub fn from_js(options: Option<&Object>) -> Result<Self, JsValue> {
        let raw = options.map(|o| js::get(o, "backend")).filter(js::present);
        let Some(raw) = raw else {
            return Ok(Self::Auto);
        };
        match raw.as_string().as_deref() {
            Some("webgpu") => Ok(Self::Webgpu),
            Some("cpu") => Ok(Self::Cpu),
            Some("auto") => Ok(Self::Auto),
            _ => Err(error(
                Kind::InvalidRequest,
                format!("backend must be \"webgpu\", \"cpu\" or \"auto\", got {raw:?}"),
            )),
        }
    }
}

/// The backend an engine owns (a clone shares the same GPU device).
#[derive(Clone)]
pub enum Device {
    Gpu { gpu: Gpu, adapter: AdapterInfo },
    Cpu,
}

impl Device {
    pub async fn open(choice: BackendChoice) -> Result<Self, JsValue> {
        if choice == BackendChoice::Cpu {
            return Ok(Self::Cpu);
        }
        let adapter = support::probe_webgpu().await;
        if !adapter.supported {
            if choice == BackendChoice::Auto {
                return Ok(Self::Cpu);
            }
            return Err(error(
                Kind::WebgpuUnavailable,
                adapter.reason.clone().unwrap_or_else(|| "no adapter".into()),
            ));
        }
        // CubeCL awaits requestAdapter/requestDevice itself (wasm cannot block).
        let gpu = open_webgpu().await;
        Ok(Self::Gpu { gpu, adapter })
    }

    /// `"webgpu"` or `"cpu"` — `BfiModelInfo.backend` and `BfiEngine.backend`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Gpu { .. } => "webgpu",
            Self::Cpu => "cpu",
        }
    }

    /// Why the device is gone (a lost GPU); a CPU is never lost.
    pub fn lost(&self) -> Option<String> {
        match self {
            Self::Gpu { gpu, .. } => gpu.lost(),
            Self::Cpu => None,
        }
    }

    /// Return freed GPU buffers to the device; nothing to do on the CPU (dropping the
    /// model frees its wasm memory for reuse).
    pub fn release_unused(&self) {
        if let Self::Gpu { gpu, .. } = self {
            gpu.release_unused();
        }
    }
}
