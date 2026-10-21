//! Tensor + Backend trait; CPU reference kernels for bfi.
//!
//! - [`Backend`] / [`CpuBackend`] / [`GpuBackend`]: synchronous, host tensors (native `bfi`).
//! - `kernels` (feature `cube`): `#[cube]` kernels shared by every CubeCL runtime.
//! - `resident` (feature `webgpu`): device-resident tensors with async readback, for
//!   the browser engine (`bfi-web`), runnable natively on wgpu for tests.

mod backend;
mod cpu;
mod dtype;
mod error;
mod fused_gemv;
mod gpu;
#[cfg(feature = "cube")]
pub mod kernels;
mod par;
mod simd;
#[cfg(feature = "webgpu")]
pub mod resident;
mod tensor;

pub use backend::{
    device_available, open_backend, select_backend, select_backend_from, Backend, DeviceKind,
    GatedDeltaParams, RopeParams, DEVICE_PROBE_ORDER,
};
pub use cpu::CpuBackend;
pub use dtype::DType;
pub use error::CoreError;
pub use fused_gemv::{fused_gemv_pq2_0, fused_gemv_ptq1_0, fused_gemv_q1_0, max_rel_err};
pub use gpu::GpuBackend;
pub use par::with_sequential;
pub use tensor::{Storage, Tensor};

/// Whether a run demands a real GPU (`BFI_REQUIRE_GPU` set). GPU tests skip silently when
/// no device enumerates, so a CI or bench run on a GPU machine sets this to turn "no device"
/// into a failure instead of a pass that never touched the GPU.
pub fn gpu_required() -> bool {
    std::env::var_os("BFI_REQUIRE_GPU").is_some()
}

/// The skip gate every GPU test shares: `Some` passes through, `None` skips — or panics
/// when [`gpu_required`].
pub fn gpu_or_skip<T>(found: Option<T>, what: &str) -> Option<T> {
    if found.is_none() && gpu_required() {
        panic!("BFI_REQUIRE_GPU is set but no GPU device opened for {what}");
    }
    found
}
