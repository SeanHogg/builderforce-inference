//! Tensor + Backend trait; CPU reference kernels for bfi.

mod backend;
mod cpu;
mod dtype;
mod error;
mod tensor;

pub use backend::{
    device_available, open_backend, select_backend, select_backend_from, Backend, DeviceKind,
    GatedDeltaParams, RopeParams, DEVICE_PROBE_ORDER,
};
pub use cpu::CpuBackend;
pub use dtype::DType;
pub use error::CoreError;
pub use tensor::{Storage, Tensor};
