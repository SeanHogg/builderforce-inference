//! Which backend an engine runs on, chosen once at `BfiEngine.create`.
//!
//! - `webgpu`: the GPU-resident engine (`Qwen3Resident`) on bfi-core's wgpu runtime —
//!   the browser's WebGPU on wasm32, wgpu on Vulkan / Metal / DX12 natively; rejects
//!   `webgpu_unavailable` without an adapter.
//! - `cpu`: the host-tensor engine (`Qwen3StreamLoader` → `Session`) on bfi-core's
//!   `CpuBackend` — the native `bfi`'s kernels, in wasm memory or natively.
//! - `auto` (default): the GPU when an adapter exists, else the CPU.
//!
//! Opening the GPU is the host's (an awaited browser setup, or a native wgpu instance);
//! the host hands the opened [`Device`] to [`crate::EngineState::new`].

#[cfg(feature = "webgpu")]
use bfi_core::resident::Gpu;

use crate::errors::{error, EngineError, Kind};
#[cfg(feature = "webgpu")]
use crate::support::AdapterInfo;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    Webgpu,
    Cpu,
    Auto,
}

impl BackendChoice {
    /// `create({ backend })`; absent = `auto`, anything else unknown = `invalid_request`.
    pub fn parse(raw: Option<&str>) -> Result<Self, EngineError> {
        match raw {
            None | Some("auto") => Ok(Self::Auto),
            Some("webgpu") => Ok(Self::Webgpu),
            Some("cpu") => Ok(Self::Cpu),
            Some(other) => Err(error(
                Kind::InvalidRequest,
                format!("backend must be \"webgpu\", \"cpu\" or \"auto\", got {other:?}"),
            )),
        }
    }

    /// Resolve against whether a GPU adapter exists: `cpu` → CPU; `auto` without an
    /// adapter → CPU; `webgpu` without one → `webgpu_unavailable: <reason>`. `Ok(true)` =
    /// open the GPU.
    pub fn wants_gpu(self, adapter_supported: bool, reason: Option<&str>) -> Result<bool, EngineError> {
        match self {
            Self::Cpu => Ok(false),
            _ if adapter_supported => Ok(true),
            Self::Auto => Ok(false),
            Self::Webgpu => Err(error(Kind::WebgpuUnavailable, reason.unwrap_or("no adapter"))),
        }
    }
}

/// The backend an engine owns (a clone shares the same GPU device).
#[derive(Clone)]
pub enum Device {
    #[cfg(feature = "webgpu")]
    Gpu { gpu: Gpu, adapter: AdapterInfo },
    Cpu,
}

impl Device {
    /// `"webgpu"` or `"cpu"` — `BfiModelInfo.backend` and `BfiEngine.backend`.
    pub fn name(&self) -> &'static str {
        match self {
            #[cfg(feature = "webgpu")]
            Self::Gpu { .. } => "webgpu",
            Self::Cpu => "cpu",
        }
    }

    /// Why the device is gone (a lost GPU); a CPU is never lost.
    pub fn lost(&self) -> Option<String> {
        match self {
            #[cfg(feature = "webgpu")]
            Self::Gpu { gpu, .. } => gpu.lost(),
            Self::Cpu => None,
        }
    }

    /// Return freed GPU buffers to the device; nothing to do on the CPU (dropping the
    /// model frees its memory for reuse).
    pub fn release_unused(&self) {
        #[cfg(feature = "webgpu")]
        if let Self::Gpu { gpu, .. } = self {
            gpu.release_unused();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_falls_back_to_the_cpu_and_webgpu_refuses() {
        assert_eq!(BackendChoice::parse(None).unwrap(), BackendChoice::Auto);
        assert!(!BackendChoice::Auto.wants_gpu(false, None).unwrap());
        assert!(BackendChoice::Auto.wants_gpu(true, None).unwrap());
        assert!(!BackendChoice::Cpu.wants_gpu(true, None).unwrap());
        let refused = BackendChoice::Webgpu.wants_gpu(false, Some("no adapter here")).unwrap_err();
        assert_eq!(refused.to_string(), "webgpu_unavailable: no adapter here");
        assert_eq!(BackendChoice::parse(Some("tpu")).unwrap_err().kind, Kind::InvalidRequest);
    }
}
