//! CubeCL GPU backends wrapping the `Backend` trait.
//!
//! Default `cargo test` does not enable any GPU feature, so `try_open` is
//! always `None` and `select_backend` still falls through to CPU. When a
//! feature is compiled *and* the device is present, `open_backend` returns
//! a [`GpuBackend`] whose fused PQ2_0 / PTQ1_0 GEMV runs on CubeCL.

#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
))]
use std::sync::OnceLock;

#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
))]
use cubecl::prelude::Runtime;

use anyhow::Context;

use crate::backend::{Backend, DeviceKind, GatedDeltaParams, RopeParams};
use crate::cpu::CpuBackend;
use crate::fused_gemv::{fused_gemv_pq2_0, fused_gemv_ptq1_0, fused_gemv_q1_0};
use crate::tensor::{Storage, Tensor};
use crate::DType;

#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
))]
#[path = "gpu_kernels.rs"]
mod gpu_kernels;

#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
))]
enum CubeClient {
    #[cfg(feature = "cuda")]
    Cuda(cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>),
    #[cfg(feature = "rocm")]
    Hip(cubecl::prelude::ComputeClient<cubecl::hip::HipRuntime>),
    #[cfg(any(feature = "vulkan", feature = "metal"))]
    Wgpu(cubecl::prelude::ComputeClient<cubecl::wgpu::WgpuRuntime>),
}

/// GPU device wrapping CubeCL. Unimplemented ops fall back to [`CpuBackend`].
pub struct GpuBackend {
    kind: DeviceKind,
    cpu: CpuBackend,
    #[cfg(any(
        feature = "cuda",
        feature = "rocm",
        feature = "vulkan",
        feature = "metal"
    ))]
    client: CubeClient,
}

impl GpuBackend {
    /// `Some` when this build compiled the matching CubeCL runtime *and* the
    /// device enumerates. Otherwise `None` so [`crate::select_backend`] lands on CPU.
    pub fn try_open(kind: DeviceKind) -> Option<Self> {
        if !probe(kind) {
            return None;
        }
        #[cfg(any(
            feature = "cuda",
            feature = "rocm",
            feature = "vulkan",
            feature = "metal"
        ))]
        {
            let client = open_client(kind)?;
            return Some(Self {
                kind,
                cpu: CpuBackend,
                client,
            });
        }
        #[cfg(not(any(
            feature = "cuda",
            feature = "rocm",
            feature = "vulkan",
            feature = "metal"
        )))]
        {
            let _ = kind;
            None
        }
    }
}

/// Compiled-feature + live-device probe. Cached per kind.
pub(crate) fn probe(kind: DeviceKind) -> bool {
    match kind {
        DeviceKind::Cpu => true,
        DeviceKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                static V: OnceLock<bool> = OnceLock::new();
                return *V.get_or_init(probe_cuda);
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        }
        DeviceKind::Rocm => {
            #[cfg(feature = "rocm")]
            {
                static V: OnceLock<bool> = OnceLock::new();
                return *V.get_or_init(probe_hip);
            }
            #[cfg(not(feature = "rocm"))]
            {
                false
            }
        }
        DeviceKind::Metal => {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                static V: OnceLock<bool> = OnceLock::new();
                return *V.get_or_init(probe_wgpu);
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            {
                false
            }
        }
        DeviceKind::Vulkan => {
            #[cfg(all(feature = "vulkan", not(target_os = "macos")))]
            {
                static V: OnceLock<bool> = OnceLock::new();
                return *V.get_or_init(probe_wgpu);
            }
            #[cfg(not(all(feature = "vulkan", not(target_os = "macos"))))]
            {
                false
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn probe_cuda() -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let device = cubecl::cuda::CudaDevice::new(0);
        let client = cubecl::cuda::CudaRuntime::client(&device);
        let _ = client.empty(4);
        true
    }))
    .unwrap_or(false)
}

#[cfg(feature = "rocm")]
fn probe_hip() -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let device = cubecl::hip::HipDevice::new(0);
        let client = cubecl::hip::HipRuntime::client(&device);
        let _ = client.empty(4);
        true
    }))
    .unwrap_or(false)
}

#[cfg(any(feature = "vulkan", feature = "metal"))]
fn probe_wgpu() -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let device = cubecl::wgpu::WgpuDevice::DefaultDevice;
        let client = cubecl::wgpu::WgpuRuntime::client(&device);
        let _ = client.empty(4);
        true
    }))
    .unwrap_or(false)
}

#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
))]
fn open_client(kind: DeviceKind) -> Option<CubeClient> {
    match kind {
        DeviceKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                return std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let device = cubecl::cuda::CudaDevice::new(0);
                    CubeClient::Cuda(cubecl::cuda::CudaRuntime::client(&device))
                }))
                .ok();
            }
            #[cfg(not(feature = "cuda"))]
            None
        }
        DeviceKind::Rocm => {
            #[cfg(feature = "rocm")]
            {
                return std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let device = cubecl::hip::HipDevice::new(0);
                    CubeClient::Hip(cubecl::hip::HipRuntime::client(&device))
                }))
                .ok();
            }
            #[cfg(not(feature = "rocm"))]
            None
        }
        DeviceKind::Metal | DeviceKind::Vulkan => {
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            {
                return std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let device = cubecl::wgpu::WgpuDevice::DefaultDevice;
                    CubeClient::Wgpu(cubecl::wgpu::WgpuRuntime::client(&device))
                }))
                .ok();
            }
            #[cfg(not(any(feature = "vulkan", feature = "metal")))]
            None
        }
        DeviceKind::Cpu => None,
    }
}

fn gemv_dims(weights: &Tensor, x: &Tensor) -> anyhow::Result<(usize, usize, usize)> {
    if x.shape.is_empty() || weights.shape.is_empty() {
        anyhow::bail!("matmul_q: empty shape");
    }
    let n_in = *x.shape.last().unwrap_or(&0);
    if n_in == 0 || x.n_elems() % n_in != 0 {
        anyhow::bail!("matmul_q: bad x shape {:?}", x.shape);
    }
    let n_tokens = x.n_elems() / n_in;
    let w_in = *weights.shape.last().unwrap_or(&0);
    if w_in != n_in {
        anyhow::bail!("matmul_q: weight in-dim {w_in} != x in-dim {n_in}");
    }
    if weights.n_elems() % w_in != 0 {
        anyhow::bail!("matmul_q: weights n_elems not divisible by n_in");
    }
    let n_out = weights.n_elems() / w_in;
    Ok((n_tokens, n_out, n_in))
}

fn quant_bytes(t: &Tensor) -> anyhow::Result<&[u8]> {
    match &t.storage {
        Storage::Quant { bytes, .. } => Ok(bytes),
        Storage::F32(_) => anyhow::bail!("expected quantized weights"),
    }
}

impl GpuBackend {
    fn fused_gemv_host(
        &self,
        dtype: DType,
        bytes: &[u8],
        x: &[f32],
        n_out: usize,
        n_in: usize,
    ) -> anyhow::Result<Vec<f32>> {
        #[cfg(any(
            feature = "cuda",
            feature = "rocm",
            feature = "vulkan",
            feature = "metal"
        ))]
        {
            let y = match (&self.client, dtype) {
                #[cfg(feature = "cuda")]
                (CubeClient::Cuda(c), DType::PQ2_0) => {
                    gpu_kernels::gemv_pq2_or_cpu(c, bytes, x, n_out, n_in)?
                }
                #[cfg(feature = "cuda")]
                (CubeClient::Cuda(c), DType::PTQ1_0) => {
                    gpu_kernels::gemv_ptq1_or_cpu(c, bytes, x, n_out, n_in)?
                }
                #[cfg(feature = "cuda")]
                (CubeClient::Cuda(c), DType::Q1_0) => {
                    gpu_kernels::gemv_q1_or_cpu(c, bytes, x, n_out, n_in)?
                }
                #[cfg(feature = "rocm")]
                (CubeClient::Hip(c), DType::PQ2_0) => {
                    gpu_kernels::gemv_pq2_or_cpu(c, bytes, x, n_out, n_in)?
                }
                #[cfg(feature = "rocm")]
                (CubeClient::Hip(c), DType::PTQ1_0) => {
                    gpu_kernels::gemv_ptq1_or_cpu(c, bytes, x, n_out, n_in)?
                }
                #[cfg(feature = "rocm")]
                (CubeClient::Hip(c), DType::Q1_0) => {
                    gpu_kernels::gemv_q1_or_cpu(c, bytes, x, n_out, n_in)?
                }
                #[cfg(any(feature = "vulkan", feature = "metal"))]
                (CubeClient::Wgpu(c), DType::PQ2_0) => {
                    gpu_kernels::gemv_pq2_or_cpu(c, bytes, x, n_out, n_in)?
                }
                #[cfg(any(feature = "vulkan", feature = "metal"))]
                (CubeClient::Wgpu(c), DType::PTQ1_0) => {
                    gpu_kernels::gemv_ptq1_or_cpu(c, bytes, x, n_out, n_in)?
                }
                #[cfg(any(feature = "vulkan", feature = "metal"))]
                (CubeClient::Wgpu(c), DType::Q1_0) => {
                    gpu_kernels::gemv_q1_or_cpu(c, bytes, x, n_out, n_in)?
                }
                _ => match dtype {
                    DType::PQ2_0 => fused_gemv_pq2_0(bytes, x, n_out, n_in)?,
                    DType::PTQ1_0 => fused_gemv_ptq1_0(bytes, x, n_out, n_in)?,
                    DType::Q1_0 => fused_gemv_q1_0(bytes, x, n_out, n_in)?,
                    _ => unreachable!(),
                },
            };
            return Ok(y);
        }
        #[cfg(not(any(
            feature = "cuda",
            feature = "rocm",
            feature = "vulkan",
            feature = "metal"
        )))]
        {
            match dtype {
                DType::PQ2_0 => fused_gemv_pq2_0(bytes, x, n_out, n_in),
                DType::PTQ1_0 => fused_gemv_ptq1_0(bytes, x, n_out, n_in),
                DType::Q1_0 => fused_gemv_q1_0(bytes, x, n_out, n_in),
                _ => anyhow::bail!("fused gemv: unsupported {dtype:?}"),
            }
        }
    }
}

impl Backend for GpuBackend {
    fn name(&self) -> &str {
        self.kind.as_str()
    }

    fn alloc(&self, shape: &[usize], dtype: DType) -> anyhow::Result<Tensor> {
        self.cpu.alloc(shape, dtype)
    }

    fn upload_f32(&self, shape: &[usize], data: &[f32]) -> anyhow::Result<Tensor> {
        self.cpu.upload_f32(shape, data)
    }

    fn upload_quant(&self, shape: &[usize], dtype: DType, bytes: &[u8]) -> anyhow::Result<Tensor> {
        self.cpu.upload_quant(shape, dtype, bytes)
    }

    fn dequant(&self, t: &Tensor) -> anyhow::Result<Tensor> {
        // CubeCL dequant kernel is not in the M1 GEMV drop; CPU reference.
        self.cpu.dequant(t)
    }

    fn matmul_q(&self, weights: &Tensor, x: &Tensor) -> anyhow::Result<Tensor> {
        let (n_tokens, n_out, n_in) = gemv_dims(weights, x)?;
        let fused = matches!(weights.dtype, DType::PQ2_0 | DType::PTQ1_0 | DType::Q1_0) && n_tokens == 1;
        if fused {
            let x_f = x.as_f32().context("matmul_q x")?;
            let bytes = quant_bytes(weights)?;
            let y = self.fused_gemv_host(weights.dtype, bytes, x_f, n_out, n_in)?;
            return Ok(Tensor::f32(vec![n_tokens, n_out], y));
        }
        // Batched GEMM / Q4_K / Q6_K / Q8_0 CubeCL kernels are later M1 items.
        self.cpu.matmul_q(weights, x)
    }

    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> anyhow::Result<Tensor> {
        // CubeCL rms_norm kernel not ready; CPU reference.
        self.cpu.rms_norm(x, weight, eps)
    }

    fn rope(&self, x: &mut Tensor, pos: &[i32], params: &RopeParams) -> anyhow::Result<()> {
        self.cpu.rope(x, pos, params)
    }

    fn rope_multi(&self, x: &mut Tensor, pos: &[i32], params: &RopeParams) -> anyhow::Result<()> {
        self.cpu.rope_multi(x, pos, params)
    }

    fn softmax_inplace(&self, x: &mut Tensor) -> anyhow::Result<()> {
        // Flash-attn / softmax CubeCL kernel is later; CPU reference.
        self.cpu.softmax_inplace(x)
    }

    fn silu(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        self.cpu.silu(x)
    }

    fn swiglu(&self, gate: &Tensor, up: &Tensor) -> anyhow::Result<Tensor> {
        self.cpu.swiglu(gate, up)
    }

    fn sigmoid(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        self.cpu.sigmoid(x)
    }

    fn softplus(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        self.cpu.softplus(x)
    }

    fn l2_norm(&self, x: &Tensor, eps: f32) -> anyhow::Result<Tensor> {
        self.cpu.l2_norm(x, eps)
    }

    fn conv1d_causal(&self, x: &Tensor, kernel: &Tensor) -> anyhow::Result<Tensor> {
        self.cpu.conv1d_causal(x, kernel)
    }

    fn gated_delta_step(&self, p: GatedDeltaParams<'_>) -> anyhow::Result<Tensor> {
        // Gated-delta-net CubeCL step is later; CPU reference.
        self.cpu.gated_delta_step(p)
    }

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        causal: bool,
    ) -> anyhow::Result<Tensor> {
        // Flash-attn decode kernel is later; CPU reference.
        self.cpu.attention(q, k, v, causal)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> anyhow::Result<Tensor> {
        self.cpu.add(a, b)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> anyhow::Result<Tensor> {
        self.cpu.mul(a, b)
    }

    fn get_rows(&self, weight: &Tensor, ids: &[i32]) -> anyhow::Result<Tensor> {
        self.cpu.get_rows(weight, ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused_gemv::max_rel_err;
    use crate::{open_backend, DEVICE_PROBE_ORDER};
    use bfi_gguf::quantize_row_pq2_0;

    fn first_gpu() -> Option<DeviceKind> {
        crate::gpu_or_skip(
            DEVICE_PROBE_ORDER
                .into_iter()
                .find(|k| *k != DeviceKind::Cpu && probe(*k)),
            "the GpuBackend tests",
        )
    }

    fn synthetic_pq2(n_out: usize, n_in: usize) -> (Vec<u8>, Vec<f32>) {
        let n_blocks = n_in / 128;
        let mut bytes = Vec::with_capacity(n_out * n_blocks * 34);
        for o in 0..n_out {
            let mut row = vec![0.25f32 * (o as f32 + 1.0); 128];
            row[0] = 1.0;
            let mut block = vec![0u8; 34];
            quantize_row_pq2_0(&row, &mut block);
            for _ in 0..n_blocks {
                bytes.extend_from_slice(&block);
            }
        }
        let x: Vec<f32> = (0..n_in).map(|i| 0.01 * (i as f32 + 1.0)).collect();
        (bytes, x)
    }

    #[test]
    fn fused_gemv_matches_cpu_backend() {
        let cpu = CpuBackend;
        let (bytes, x) = synthetic_pq2(4, 128);
        let w = cpu.upload_quant(&[4, 128], DType::PQ2_0, &bytes).unwrap();
        let xt = cpu.upload_f32(&[1, 128], &x).unwrap();
        let y_cpu = cpu.matmul_q(&w, &xt).unwrap();
        let y_fused = fused_gemv_pq2_0(&bytes, &x, 4, 128).unwrap();
        let err = max_rel_err(y_cpu.as_f32().unwrap(), &y_fused);
        assert!(err <= 1e-3, "rel err {err}");
    }

    #[test]
    fn gpu_pq2_matches_cpu_when_device_present() {
        let Some(kind) = first_gpu() else {
            // No compiled GPU feature / no device: CPU-only reference already
            // covered by fused_gemv_matches_cpu_backend.
            return;
        };
        let gpu = open_backend(kind).expect("probe said the device is present");
        let cpu = CpuBackend;
        let (bytes, x) = synthetic_pq2(4, 128);
        let w = cpu.upload_quant(&[4, 128], DType::PQ2_0, &bytes).unwrap();
        let xt = cpu.upload_f32(&[1, 128], &x).unwrap();
        let y_cpu = cpu.matmul_q(&w, &xt).unwrap();
        let y_gpu = gpu.matmul_q(&w, &xt).unwrap();
        let err = max_rel_err(y_gpu.as_f32().unwrap(), y_cpu.as_f32().unwrap());
        assert!(
            err <= 1e-3,
            "GPU vs CPU rel err {err} gpu={:?} cpu={:?}",
            y_gpu.as_f32().unwrap(),
            y_cpu.as_f32().unwrap()
        );
    }

    #[test]
    fn gpu_ptq1_matches_cpu_when_device_present() {
        let Some(kind) = first_gpu() else {
            return;
        };
        let gpu = open_backend(kind).expect("probe said the device is present");
        let cpu = CpuBackend;
        let mut bytes = vec![0u8; 28];
        bytes[26..28].copy_from_slice(&0x4000u16.to_le_bytes());
        let w = cpu.upload_quant(&[1, 128], DType::PTQ1_0, &bytes).unwrap();
        let xt = cpu.upload_f32(&[1, 128], &[1.0f32; 128]).unwrap();
        let y_cpu = cpu.matmul_q(&w, &xt).unwrap();
        let y_gpu = gpu.matmul_q(&w, &xt).unwrap();
        let err = max_rel_err(y_gpu.as_f32().unwrap(), y_cpu.as_f32().unwrap());
        assert!(err <= 1e-3, "GPU vs CPU rel err {err}");
    }

    fn synthetic_q1(n_out: usize, n_in: usize) -> (Vec<u8>, Vec<f32>) {
        let mut bytes = Vec::with_capacity(n_out * (n_in / 128) * 18);
        for o in 0..n_out {
            for b in 0..n_in / 128 {
                let row: Vec<f32> = (0..128)
                    .map(|j| (((o * 31 + b * 7 + j * 13) % 17) as f32 - 8.0) * 0.05)
                    .collect();
                let mut block = vec![0u8; 18];
                bfi_gguf::quantize_row_q1_0(&row, &mut block);
                bytes.extend_from_slice(&block);
            }
        }
        let x: Vec<f32> = (0..n_in).map(|i| ((i % 11) as f32 - 5.0) * 0.02).collect();
        (bytes, x)
    }

    #[test]
    fn fused_q1_matches_cpu_backend() {
        let cpu = CpuBackend;
        let (bytes, x) = synthetic_q1(5, 256);
        let w = cpu.upload_quant(&[5, 256], DType::Q1_0, &bytes).unwrap();
        let xt = cpu.upload_f32(&[1, 256], &x).unwrap();
        let y_cpu = cpu.matmul_q(&w, &xt).unwrap();
        let y_fused = fused_gemv_q1_0(&bytes, &x, 5, 256).unwrap();
        let err = max_rel_err(y_cpu.as_f32().unwrap(), &y_fused);
        assert!(err <= 1e-3, "rel err {err}");
    }

    #[test]
    fn gpu_q1_matches_cpu_when_device_present() {
        let Some(kind) = first_gpu() else {
            return;
        };
        let gpu = open_backend(kind).expect("probe said the device is present");
        let cpu = CpuBackend;
        let (bytes, x) = synthetic_q1(70, 384);
        let w = cpu.upload_quant(&[70, 384], DType::Q1_0, &bytes).unwrap();
        let xt = cpu.upload_f32(&[1, 384], &x).unwrap();
        let y_cpu = cpu.matmul_q(&w, &xt).unwrap();
        let y_gpu = gpu.matmul_q(&w, &xt).unwrap();
        let err = max_rel_err(y_gpu.as_f32().unwrap(), y_cpu.as_f32().unwrap());
        assert!(err <= 1e-3, "GPU vs CPU Q1_0 rel err {err}");
    }
}
