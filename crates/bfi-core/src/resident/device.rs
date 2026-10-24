//! Opening the wgpu device.
//!
//! wasm32: the browser cannot block, so CubeCL's synchronous device init
//! (`DeviceService::init` → `future::block_on`) would panic. [`open_webgpu`]
//! instead awaits `cubecl::wgpu::init_setup_async::<WebGpu>`, which requests the
//! adapter (high performance) and the device with the adapter's own limits and
//! features (so `shader-f16` is enabled when offered), and registers the server
//! for `WgpuDevice::DefaultDevice`. `WgpuRuntime::client` then *loads* that
//! registered server instead of initialising one.
//!
//! CubeCL panics (it `expect`s) when no adapter exists, and a wasm panic aborts
//! the module — so callers must confirm `navigator.gpu.requestAdapter()` first
//! (`bfi-web` does, and rejects with `webgpu_unavailable:` otherwise).
//!
//! Native: the same setup, synchronously (`init_setup`), on the graphics API CubeCL
//! picks ([`native_backend`]: Vulkan, Metal on macOS) — opened ONCE per process and
//! shared, because CubeCL registers one server per device. [`probe_native`] reports
//! that device's limits (or its adapter's, before it opens) so a support record never
//! describes a different GPU than the one the engine computes on, and reports it
//! unsupported once it failed to open or was lost.

use cubecl::prelude::Runtime;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

use super::Gpu;

#[cfg(target_family = "wasm")]
pub async fn open_webgpu() -> Gpu {
    let device = WgpuDevice::DefaultDevice;
    let setup = cubecl::wgpu::init_setup_async::<cubecl::wgpu::WebGpu>(
        &device,
        cubecl::wgpu::RuntimeOptions::default(),
    )
    .await;
    // Keep the raw device for error-scoped allocations and `device.lost`
    // (health.rs); CubeCL exposes neither.
    Gpu::with_probe_device(WgpuRuntime::client(&device), setup.device)
}

/// What a native GPU offers the engine: the limits and `shader-f16` that size its KV
/// cache. Read from the opened DEVICE (or, before one is open, an adapter on the same
/// graphics API), never from a second adapter on another API.
#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy, Debug)]
pub struct NativeLimits {
    pub max_buffer_bytes: u64,
    pub max_binding_bytes: u64,
    pub f16: bool,
}

/// The process's ONE native device. CubeCL registers a single server for
/// `WgpuDevice::DefaultDevice` per process, so every engine (every editor worker) shares
/// it: opened once, its raw `wgpu::Device` kept for error scopes and `device.lost`.
#[cfg(not(target_family = "wasm"))]
struct NativeDevice {
    device: wgpu::Device,
    limits: NativeLimits,
    health: super::health::DeviceHealth,
}

#[cfg(not(target_family = "wasm"))]
static NATIVE: std::sync::OnceLock<Option<NativeDevice>> = std::sync::OnceLock::new();

/// The graphics API CubeCL opens natively: Vulkan, or Metal on macOS (never DX12).
#[cfg(not(target_family = "wasm"))]
pub fn native_backend() -> wgpu::Backend {
    use cubecl::wgpu::GraphicsApi;
    cubecl::wgpu::AutoGraphicsApi::backend()
}

#[cfg(not(target_family = "wasm"))]
fn native_device() -> Option<&'static NativeDevice> {
    NATIVE
        .get_or_init(|| {
            std::panic::catch_unwind(|| {
                let setup = cubecl::wgpu::init_setup::<cubecl::wgpu::AutoGraphicsApi>(
                    &WgpuDevice::DefaultDevice,
                    cubecl::wgpu::RuntimeOptions::default(),
                );
                let limits = setup.device.limits();
                let health = super::health::DeviceHealth::default();
                health.watch(&setup.device);
                // wgpu's default panics on an uncaptured error, on CubeCL's device
                // thread; record it for `Gpu::checked` / `drain` instead.
                health.capture(&setup.device);
                NativeDevice {
                    limits: NativeLimits {
                        max_buffer_bytes: limits.max_buffer_size,
                        max_binding_bytes: limits.max_storage_buffer_binding_size,
                        f16: setup.device.features().contains(wgpu::Features::SHADER_F16),
                    },
                    device: setup.device,
                    health,
                }
            })
            .ok()
        })
        .as_ref()
}

/// Native wgpu on [`native_backend`]: the shared device, with the device-lost watch
/// and error capture the browser path has. `None` when it cannot be opened, or once
/// it was lost (a driver reset, a sleep): CubeCL holds one server per device for the
/// life of the process, so a lost device stays lost and the engine runs on the CPU.
#[cfg(not(target_family = "wasm"))]
pub fn open_native() -> Option<Gpu> {
    let native = native_device()?;
    if native.health.lost().is_some() {
        return None;
    }
    let client = WgpuRuntime::client(&WgpuDevice::DefaultDevice);
    Some(Gpu {
        client,
        probe_device: native.device.clone(),
        health: native.health.clone(),
    })
}

/// The opened native device's limits (opening it if needed).
#[cfg(not(target_family = "wasm"))]
pub fn native_limits() -> Option<NativeLimits> {
    native_device().map(|n| n.limits)
}

/// Whether a native GPU can serve, and its limits, without opening a device: the
/// opened device's state when there is one (unsupported once it failed to open or
/// was lost), else the high-performance adapter on [`native_backend`], probed once
/// per process (CubeCL requests every feature and limit that adapter offers).
#[cfg(not(target_family = "wasm"))]
pub fn probe_native() -> Result<NativeLimits, String> {
    match NATIVE.get() {
        Some(Some(native)) => match native.health.lost() {
            Some(why) => Err(format!("the GPU device was lost ({why}); restart the editor to use it again")),
            None => Ok(native.limits),
        },
        Some(None) => Err(format!("the {:?} GPU device would not open", native_backend())),
        None => {
            static ADAPTER: std::sync::OnceLock<Result<NativeLimits, String>> = std::sync::OnceLock::new();
            ADAPTER.get_or_init(probe_adapter).clone()
        }
    }
}

#[cfg(not(target_family = "wasm"))]
fn probe_adapter() -> Result<NativeLimits, String> {
    let backend = native_backend();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: backend.into(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    };
    let adapter = super::block_on(instance.request_adapter(&options)).map_err(|e| format!("no {backend:?} GPU adapter: {e}"))?;
    let limits = adapter.limits();
    Ok(NativeLimits {
        max_buffer_bytes: limits.max_buffer_size,
        max_binding_bytes: limits.max_storage_buffer_binding_size,
        f16: adapter.features().contains(wgpu::Features::SHADER_F16),
    })
}
