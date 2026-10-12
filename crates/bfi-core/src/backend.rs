use crate::cpu::CpuBackend;
use crate::gpu::GpuBackend;
use crate::{DType, Tensor};

#[derive(Clone, Debug)]
pub struct RopeParams {
    pub n_dims: usize,
    pub n_ctx: usize,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    /// rope_dimension_count, often head_dim/2 * 2
    pub n_rot: usize,
    /// rope_multi sections; None = classic RoPE
    pub sections: Option<[usize; 4]>,
}

#[derive(Debug)]
pub struct GatedDeltaParams<'a> {
    pub q: &'a Tensor,
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    /// Gate; sigmoid is applied inside the kernel.
    pub g: &'a Tensor,
    pub beta: &'a Tensor,
    /// `[n_kv_heads, head_dim, head_dim]` recurrent state
    pub state: &'a mut Tensor,
}

pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn alloc(&self, shape: &[usize], dtype: DType) -> anyhow::Result<Tensor>;
    fn upload_f32(&self, shape: &[usize], data: &[f32]) -> anyhow::Result<Tensor>;
    /// Host→device upload of F32 data (PRD 5.2 `upload`).
    fn upload(&self, shape: &[usize], data: &[f32]) -> anyhow::Result<Tensor> {
        self.upload_f32(shape, data)
    }
    fn upload_quant(&self, shape: &[usize], dtype: DType, bytes: &[u8]) -> anyhow::Result<Tensor>;
    /// Dequantize to F32.
    fn dequant(&self, t: &Tensor) -> anyhow::Result<Tensor>;
    /// `weights`: `[n_out, n_in]` quantized or F32; `x`: `[n_tokens, n_in]` F32; out: `[n_tokens, n_out]` F32.
    fn matmul_q(&self, weights: &Tensor, x: &Tensor) -> anyhow::Result<Tensor>;
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> anyhow::Result<Tensor>;
    /// `x` is `[n_tokens, n_heads, head_dim]`.
    fn rope(&self, x: &mut Tensor, pos: &[i32], params: &RopeParams) -> anyhow::Result<()>;
    fn rope_multi(&self, x: &mut Tensor, pos: &[i32], params: &RopeParams) -> anyhow::Result<()>;
    /// Softmax over the last dim.
    fn softmax_inplace(&self, x: &mut Tensor) -> anyhow::Result<()>;
    /// Softmax over the last dim, returning a new tensor (PRD 5.2 `softmax`).
    fn softmax(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let mut y = x.clone();
        self.softmax_inplace(&mut y)?;
        Ok(y)
    }
    fn silu(&self, x: &Tensor) -> anyhow::Result<Tensor>;
    /// `silu(gate) * up`
    fn swiglu(&self, gate: &Tensor, up: &Tensor) -> anyhow::Result<Tensor>;
    fn sigmoid(&self, x: &Tensor) -> anyhow::Result<Tensor>;
    fn softplus(&self, x: &Tensor) -> anyhow::Result<Tensor>;
    /// L2 normalize along the last dim.
    fn l2_norm(&self, x: &Tensor, eps: f32) -> anyhow::Result<Tensor>;
    /// `x`: `[tokens, channels]`; `kernel`: `[channels, k_len]` or `[k_len]` broadcast.
    /// Causal depthwise conv over time.
    fn conv1d_causal(&self, x: &Tensor, kernel: &Tensor) -> anyhow::Result<Tensor>;
    fn gated_delta_step(&self, p: GatedDeltaParams<'_>) -> anyhow::Result<Tensor>;
    /// `q`: `[t, hq, d]`, `k`: `[tk, hkv, d]`, `v`: `[tk, hkv, d]`; GQA: `hq % hkv == 0`.
    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> anyhow::Result<Tensor>;
    fn add(&self, a: &Tensor, b: &Tensor) -> anyhow::Result<Tensor>;
    fn mul(&self, a: &Tensor, b: &Tensor) -> anyhow::Result<Tensor>;
    /// `weight` `[n_vocab, n_embd]` maybe quant; out `[ids.len(), n_embd]` F32.
    fn get_rows(&self, weight: &Tensor, ids: &[i32]) -> anyhow::Result<Tensor>;
}

/// Device names accepted by `BFI_BACKEND` (case-insensitive).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    Cuda,
    Rocm,
    Metal,
    Vulkan,
    Cpu,
}

impl DeviceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DeviceKind::Cuda => "cuda",
            DeviceKind::Rocm => "rocm",
            DeviceKind::Metal => "metal",
            DeviceKind::Vulkan => "vulkan",
            DeviceKind::Cpu => "cpu",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "cuda" | "nvidia" => Some(Self::Cuda),
            "rocm" | "hip" | "amd" => Some(Self::Rocm),
            "metal" | "mps" => Some(Self::Metal),
            "vulkan" | "wgpu" | "gpu" => Some(Self::Vulkan),
            "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }
}

/// Default probe order from PRD 5.2: CUDA > ROCm > Metal > Vulkan > CPU.
pub const DEVICE_PROBE_ORDER: [DeviceKind; 5] = [
    DeviceKind::Cuda,
    DeviceKind::Rocm,
    DeviceKind::Metal,
    DeviceKind::Vulkan,
    DeviceKind::Cpu,
];

/// Whether this build can open `kind`. GPU kinds are true only when the
/// matching CubeCL feature is compiled *and* the device enumerates.
pub fn device_available(kind: DeviceKind) -> bool {
    match kind {
        DeviceKind::Cpu => true,
        other => crate::gpu::probe(other),
    }
}

pub fn open_backend(kind: DeviceKind) -> Option<Box<dyn Backend>> {
    match kind {
        DeviceKind::Cpu => Some(Box::new(CpuBackend)),
        DeviceKind::Cuda | DeviceKind::Rocm | DeviceKind::Metal | DeviceKind::Vulkan => {
            GpuBackend::try_open(kind).map(|b| Box::new(b) as Box<dyn Backend>)
        }
    }
}

/// Resolve an optional `BFI_BACKEND` override, then walk the remaining probe
/// order so a requested GPU with no compiled kernel still lands on CPU.
pub fn select_backend_from(override_name: Option<&str>) -> Box<dyn Backend> {
    let requested = override_name.and_then(DeviceKind::parse);
    let mut order: Vec<DeviceKind> = Vec::with_capacity(DEVICE_PROBE_ORDER.len());
    if let Some(first) = requested {
        order.push(first);
    }
    for k in DEVICE_PROBE_ORDER {
        if Some(k) != requested {
            order.push(k);
        }
    }
    for k in order {
        if let Some(b) = open_backend(k) {
            return b;
        }
    }
    Box::new(CpuBackend)
}

/// Automatic device selection. Honors `BFI_BACKEND`; missing GPU features or
/// devices fall through to CPU.
pub fn select_backend() -> Box<dyn Backend> {
    select_backend_from(std::env::var("BFI_BACKEND").ok().as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_kind_parse_aliases() {
        assert_eq!(DeviceKind::parse("CUDA"), Some(DeviceKind::Cuda));
        assert_eq!(DeviceKind::parse(" nvidia "), Some(DeviceKind::Cuda));
        assert_eq!(DeviceKind::parse("HIP"), Some(DeviceKind::Rocm));
        assert_eq!(DeviceKind::parse("mps"), Some(DeviceKind::Metal));
        assert_eq!(DeviceKind::parse("wgpu"), Some(DeviceKind::Vulkan));
        assert_eq!(DeviceKind::parse("cpu"), Some(DeviceKind::Cpu));
        assert_eq!(DeviceKind::parse("nope"), None);
    }

    #[test]
    fn probe_order_is_cuda_rocm_metal_vulkan_cpu() {
        assert_eq!(
            DEVICE_PROBE_ORDER,
            [
                DeviceKind::Cuda,
                DeviceKind::Rocm,
                DeviceKind::Metal,
                DeviceKind::Vulkan,
                DeviceKind::Cpu,
            ]
        );
        assert!(device_available(DeviceKind::Cpu));
        for kind in [
            DeviceKind::Cuda,
            DeviceKind::Rocm,
            DeviceKind::Metal,
            DeviceKind::Vulkan,
        ] {
            assert_eq!(
                open_backend(kind).is_some(),
                device_available(kind),
                "{kind:?}"
            );
        }
        assert_eq!(open_backend(DeviceKind::Cpu).unwrap().name(), "cpu");
        // Default cargo test compiles no GPU feature, so CUDA cannot open.
        if !cfg!(feature = "cuda") {
            assert!(!device_available(DeviceKind::Cuda));
            assert!(open_backend(DeviceKind::Cuda).is_none());
        }
    }

    #[test]
    fn select_backend_from_honors_override_and_falls_back() {
        assert_eq!(select_backend_from(Some("cpu")).name(), "cpu");
        for (name, kind) in [
            ("cuda", DeviceKind::Cuda),
            ("ROCm", DeviceKind::Rocm),
            ("metal", DeviceKind::Metal),
            ("vulkan", DeviceKind::Vulkan),
        ] {
            let b = select_backend_from(Some(name));
            if device_available(kind) {
                assert_eq!(b.name(), kind.as_str(), "{name}");
            } else {
                assert_eq!(b.name(), "cpu", "{name} falls through to cpu");
            }
        }
        assert_eq!(select_backend_from(Some("garbage")).name(), "cpu");
        let expected = DEVICE_PROBE_ORDER
            .iter()
            .find(|k| device_available(**k))
            .unwrap();
        assert_eq!(select_backend_from(None).name(), expected.as_str());
    }
}
