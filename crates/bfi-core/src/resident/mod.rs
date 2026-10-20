//! GPU-resident tensors and ops on CubeCL's wgpu runtime (feature `webgpu`).
//!
//! Why a second path next to [`crate::Backend`]: `Backend` is synchronous and
//! host-tensor based (each GPU op reads its result back). Browsers cannot block
//! on a GPU readback, so the WebGPU engine keeps weights, activations and the KV
//! cache on the device, *enqueues* a whole forward pass, and awaits exactly one
//! readback (the logits) per step via [`Gpu::read_f32`].
//!
//! The same code runs natively (wgpu → Vulkan/Metal/DX12), which is how the
//! kernels are tested against [`crate::CpuBackend`] without a browser.
//!
//! Weights stay packed: one device buffer per tensor holding the GGUF block bytes.
//! Supported weight types are [`RESIDENT_WEIGHT_TYPES`]; anything else is a
//! [`ResidentError::UnsupportedType`] naming the tensor.

mod device;
mod health;
mod ops;

#[cfg(test)]
mod tests;

use bfi_gguf::{lookup_type, TYPE_F16, TYPE_F32, TYPE_Q1_0, TYPE_Q8_0};
use cubecl::client::ComputeClient;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;

#[cfg(target_family = "wasm")]
pub use device::open_webgpu;
#[cfg(not(target_family = "wasm"))]
pub use device::open_native;

/// Drive a resident future (e.g. [`Gpu::read_f32`]) to completion. Native only:
/// on wasm32 the browser cannot block, so callers `.await` instead.
#[cfg(not(target_family = "wasm"))]
pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
    cubecl::future::block_on(f)
}

/// The runtime every resident tensor lives on.
pub type ResidentClient = ComputeClient<WgpuRuntime>;

/// Weight types the resident path multiplies and gathers in packed form.
/// Adding one = one row here + one `gemm_*` / `rows_*` kernel pair in
/// [`crate::kernels`] + one arm in `ops.rs`.
pub const RESIDENT_WEIGHT_TYPES: &[u32] = &[TYPE_Q1_0, TYPE_Q8_0, TYPE_F16, TYPE_F32];

pub fn resident_supports(ggml_type: u32) -> bool {
    RESIDENT_WEIGHT_TYPES.contains(&ggml_type)
}

#[derive(Debug, thiserror::Error)]
pub enum ResidentError {
    #[error("tensor {tensor} has GGML type {type_name} (id {id}), which the WebGPU engine cannot run")]
    UnsupportedType {
        tensor: String,
        id: u32,
        type_name: String,
    },
    #[error("tensor {tensor}: {msg}")]
    BadShape { tensor: String, msg: String },
    #[error("GPU readback failed: {0}")]
    Readback(String),
    /// An allocation (upload, KV cache) hit a WebGPU out-of-memory or
    /// validation error; the buffers it made are invalid.
    #[error("GPU allocation failed: {0}")]
    Allocation(String),
    /// `device.lost` fired; nothing on this device works any more.
    #[error("the GPU device was lost ({0})")]
    DeviceLost(String),
}

/// An f32 (or u32) device buffer of `len` elements.
#[derive(Clone)]
pub struct DeviceTensor {
    pub(crate) handle: Handle,
    pub len: usize,
}

/// One layer's K or V cache, `[ctx, n_head_kv, head_dim]`, stored as f32 or as
/// packed f16 (two halves per `u32` word — half the memory, so twice the context).
#[derive(Clone)]
pub struct KvBuffer {
    pub(crate) words: DeviceTensor,
    /// Logical element count (`ctx · n_head_kv · head_dim`).
    pub elems: usize,
    pub f16: bool,
}

/// A packed weight matrix `[n_out, n_in]` (GGUF dims `[n_in, n_out]`).
#[derive(Clone)]
pub struct Weight {
    pub(crate) handle: Handle,
    pub(crate) words: usize,
    pub ggml_type: u32,
    pub n_out: u32,
    pub n_in: u32,
}

impl Weight {
    pub fn device_bytes(&self) -> usize {
        self.words * 4
    }
}

/// Device handle plus the op set a decoder needs.
#[derive(Clone)]
pub struct Gpu {
    client: ResidentClient,
    /// The raw wgpu device, when we created it (browser path): lets
    /// [`Gpu::kv_fits`] / [`Gpu::checked`] run allocations inside error scopes,
    /// which CubeCL does not expose (see `health.rs`).
    probe_device: Option<wgpu::Device>,
    /// Set by the device-lost callback.
    health: health::DeviceHealth,
}

impl Gpu {
    pub fn from_client(client: ResidentClient) -> Self {
        Self {
            client,
            probe_device: None,
            health: health::DeviceHealth::default(),
        }
    }

    #[cfg_attr(not(target_family = "wasm"), allow(dead_code))]
    pub(crate) fn with_probe_device(client: ResidentClient, device: wgpu::Device) -> Self {
        let health = health::DeviceHealth::default();
        health.watch(&device);
        Self {
            client,
            probe_device: Some(device),
            health,
        }
    }

    /// Uninitialised KV buffer of `elems` logical elements (`elems` even for f16).
    pub fn empty_kv(&self, elems: usize, f16: bool) -> KvBuffer {
        let words = if f16 { elems.div_ceil(2) } else { elems };
        KvBuffer {
            words: DeviceTensor {
                handle: self.client.empty(words.max(1) * 4),
                len: words,
            },
            elems,
            f16,
        }
    }

    pub fn client(&self) -> &ResidentClient {
        &self.client
    }

    /// Upload one GGUF tensor's packed bytes as-is (padded to whole words).
    pub fn upload_weight(
        &self,
        tensor: &str,
        ggml_type: u32,
        dims: &[u64],
        bytes: &[u8],
    ) -> Result<Weight, ResidentError> {
        if !resident_supports(ggml_type) {
            return Err(ResidentError::UnsupportedType {
                tensor: tensor.to_string(),
                id: ggml_type,
                type_name: lookup_type(ggml_type)
                    .map(|t| t.name.to_string())
                    .unwrap_or_else(|| "unknown".into()),
            });
        }
        let n_in = *dims.first().ok_or_else(|| ResidentError::BadShape {
            tensor: tensor.into(),
            msg: "no dims".into(),
        })?;
        let n_out: u64 = dims[1..].iter().product::<u64>().max(1);
        if n_in > u32::MAX as u64 || n_out > u32::MAX as u64 {
            return Err(ResidentError::BadShape {
                tensor: tensor.into(),
                msg: format!("dims {dims:?} exceed u32"),
            });
        }
        let ty = lookup_type(ggml_type).expect("resident types are registered");
        let want = ty.bytes_for_elems(n_in * n_out).ok_or_else(|| ResidentError::BadShape {
            tensor: tensor.into(),
            msg: format!("{n_in}x{n_out} is not a whole number of {} blocks", ty.name),
        })?;
        if want as usize != bytes.len() || n_in % ty.block_elems as u64 != 0 {
            return Err(ResidentError::BadShape {
                tensor: tensor.into(),
                msg: format!("{} bytes for {n_in}x{n_out} {}", bytes.len(), ty.name),
            });
        }
        let words = bytes.len().div_ceil(4);
        let handle = if bytes.len() % 4 == 0 {
            self.client.create_from_slice(bytes)
        } else {
            let mut padded = Vec::with_capacity(words * 4);
            padded.extend_from_slice(bytes);
            padded.resize(words * 4, 0);
            self.client.create_from_slice(&padded)
        };
        Ok(Weight {
            handle,
            words,
            ggml_type,
            n_out: n_out as u32,
            n_in: n_in as u32,
        })
    }

    pub fn upload_f32(&self, v: &[f32]) -> DeviceTensor {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        DeviceTensor {
            handle: self.client.create_from_slice(&bytes),
            len: v.len(),
        }
    }

    pub fn upload_u32(&self, v: &[u32]) -> DeviceTensor {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        DeviceTensor {
            handle: self.client.create_from_slice(&bytes),
            len: v.len(),
        }
    }

    /// Uninitialised f32 buffer (callers write before they read).
    pub fn empty_f32(&self, len: usize) -> DeviceTensor {
        DeviceTensor {
            handle: self.client.empty(len.max(1) * 4),
            len,
        }
    }

    /// Await the buffer's contents. The only host sync point in a decode step.
    pub async fn read_f32(&self, x: &DeviceTensor) -> Result<Vec<f32>, ResidentError> {
        let mut out = self
            .client
            .read_async(vec![x.handle.clone()])
            .await
            .map_err(|e| ResidentError::Readback(format!("{e:?}")))?;
        let bytes = out.remove(0);
        Ok(bytes
            .chunks_exact(4)
            .take(x.len)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Release pooled device memory that no live handle references (after unload).
    pub fn release_unused(&self) {
        self.client.memory_cleanup();
    }
}
