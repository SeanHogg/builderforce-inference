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
    // Keep the raw device for KV allocation probes (error scopes).
    Gpu::with_probe_device(WgpuRuntime::client(&device), setup.device)
}

/// Native wgpu (Vulkan / Metal / DX12). `None` when no adapter enumerates.
#[cfg(not(target_family = "wasm"))]
pub fn open_native() -> Option<Gpu> {
    std::panic::catch_unwind(|| {
        let device = WgpuDevice::DefaultDevice;
        let client = WgpuRuntime::client(&device);
        let _ = client.empty(4);
        Gpu::from_client(client)
    })
    .ok()
}
