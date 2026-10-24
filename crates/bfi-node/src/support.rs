//! `BfiEngine.support()` in a native process, and how the engine's backend is opened.
//!
//! The GPU facts are bfi-core's native device's — the graphics API CubeCL opens (Vulkan,
//! Metal on macOS), its limits and `shader-f16` — so the support record and the KV cache
//! a load allocates describe the same GPU. The CPU facts are the machine's: rayon's
//! thread count, and the RAM share the CPU backend plans to use.
//!
//! The adapter is probed once per process (an adapter request can take hundreds of
//! milliseconds, and a probe runs on every picker frame and every local-first turn),
//! but the GPU's state is read live: once its device failed to open or was lost it is
//! unsupported, so the next fit lands on the CPU.

use std::sync::OnceLock;

#[cfg(feature = "webgpu")]
use bfi_engine::{error, Kind};
use bfi_engine::{AdapterInfo, BackendChoice, CpuInfo, Device, EngineError, HostProfile, Support};
use bfi_models::kv_plan::{ASSUMED_DESKTOP_MEMORY_BYTES, DEVICE_MEMORY_SHARE_DIVISOR};

/// The machine's RAM, in bytes (`None` when the OS will not say).
fn total_memory_bytes() -> Option<u64> {
    static MEMORY: OnceLock<Option<u64>> = OnceLock::new();
    *MEMORY.get_or_init(|| {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        Some(sys.total_memory()).filter(|&m| m > 0)
    })
}

/// The CPU backend's rules here: the RAM share it plans against (an unreported RAM
/// reads as a desktop's, as the catalog's fit check assumes).
pub fn profile() -> HostProfile {
    HostProfile::native(total_memory_bytes().unwrap_or(ASSUMED_DESKTOP_MEMORY_BYTES) / DEVICE_MEMORY_SHARE_DIVISOR)
}

pub fn probe() -> Support {
    let profile = profile();
    Support {
        webgpu: probe_adapter(),
        cpu: CpuInfo {
            // bfi-core's explicit SIMD paths are wasm32's; natively the kernels are
            // auto-vectorized scalar code spread over rayon's pool.
            simd: false,
            threads: rayon::current_num_threads() as u32,
            kv_budget_bytes: profile.cpu_kv_budget_bytes,
            max_memory_bytes: profile.cpu_memory_cap,
            device_memory_bytes: total_memory_bytes().map(|m| m as f64),
        },
    }
}

#[cfg(feature = "webgpu")]
fn adapter_info(limits: bfi_core::resident::NativeLimits) -> AdapterInfo {
    AdapterInfo {
        supported: true,
        reason: None,
        max_buffer_bytes: limits.max_buffer_bytes as f64,
        max_binding_bytes: limits.max_binding_bytes as f64,
        f16: limits.f16,
    }
}

#[cfg(feature = "webgpu")]
fn probe_adapter() -> AdapterInfo {
    bfi_core::resident::probe_native().map_or_else(AdapterInfo::unsupported, adapter_info)
}

#[cfg(not(feature = "webgpu"))]
fn probe_adapter() -> AdapterInfo {
    AdapterInfo::unsupported("this bfi-node build is CPU-only")
}

/// Open the backend `choice` names: the GPU when an adapter exists (and CubeCL opens it),
/// else the CPU for `auto`.
pub fn open(choice: BackendChoice) -> Result<Device, EngineError> {
    if choice == BackendChoice::Cpu {
        return Ok(Device::Cpu);
    }
    let adapter = probe_adapter();
    if !choice.wants_gpu(adapter.supported, adapter.reason.as_deref())? {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "webgpu")]
    {
        // The opened device's own limits, which the probe above may have read off its
        // adapter before it opened.
        match bfi_core::resident::open_native().zip(bfi_core::resident::native_limits()) {
            Some((gpu, limits)) => Ok(Device::Gpu { gpu, adapter: adapter_info(limits) }),
            None if choice == BackendChoice::Auto => Ok(Device::Cpu),
            None => Err(error(
                Kind::WebgpuUnavailable,
                bfi_core::resident::probe_native().err().unwrap_or_else(|| "the GPU device would not open".into()),
            )),
        }
    }
    #[cfg(not(feature = "webgpu"))]
    unreachable!("probe_adapter reports unsupported in a CPU-only build")
}
