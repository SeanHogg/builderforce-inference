//! Streaming load: [`Qwen3ResidentLoader`] uploads each tensor as
//! [`bfi_gguf::TensorStream`] completes it, then assembles a [`Qwen3Resident`].

use std::collections::HashMap;

use anyhow::{anyhow, ensure, Context, Result};
use bfi_core::resident::{DeviceTensor, Gpu, Weight};
use bfi_gguf::{dequant_tensor, require_type, ModelHeader, StreamedTensor};

use super::{Layer, Qwen3Resident};
use crate::kv_plan::KvShape;
use crate::qwen3::{parse_config, Qwen3Config, ARCH_NAME};

enum Slot {
    Matrix(Weight),
    Vector(DeviceTensor),
}

/// Collects streamed tensors into device buffers, then assembles the model.
pub struct Qwen3ResidentLoader {
    gpu: Gpu,
    cfg: Qwen3Config,
    tensors: HashMap<String, Slot>,
    device_bytes: u64,
    params: u64,
}

impl Qwen3ResidentLoader {
    pub fn new(gpu: Gpu, header: &ModelHeader) -> Result<Self> {
        let arch = header
            .architecture()
            .ok_or_else(|| anyhow!("GGUF missing metadata key general.architecture"))?;
        ensure!(
            arch == ARCH_NAME,
            "architecture '{arch}' is not supported by the WebGPU engine (qwen3 only)"
        );
        if let Some(n) = header.arch_u64("expert_count") {
            ensure!(n == 0, "qwen3moe (expert_count={n}) is not supported; dense qwen3 only");
        }
        let cfg = parse_config(header)?;
        Ok(Self {
            gpu,
            cfg,
            tensors: HashMap::new(),
            device_bytes: 0,
            params: 0,
        })
    }

    /// Upload one tensor. Matrices stay packed; an unsupported weight type fails
    /// with [`bfi_core::resident::ResidentError::UnsupportedType`] (downcastable).
    pub fn accept(&mut self, t: StreamedTensor) -> Result<()> {
        let info = t.info;
        self.params += info.n_elems();
        let slot = if info.dims.len() <= 1 {
            let ty = require_type(info.ggml_type, &info.name)?;
            let mut v = vec![0.0f32; info.n_elems() as usize];
            dequant_tensor(ty, &t.bytes, &mut v)
                .with_context(|| format!("dequant {}", info.name))?;
            self.device_bytes += (v.len() * 4) as u64;
            Slot::Vector(self.gpu.upload_f32(&v))
        } else {
            let w = self
                .gpu
                .upload_weight(&info.name, info.ggml_type, &info.dims, &t.bytes)?;
            self.device_bytes += w.device_bytes() as u64;
            Slot::Matrix(w)
        };
        self.tensors.insert(info.name, slot);
        Ok(())
    }

    /// Bytes uploaded so far.
    pub fn device_bytes(&self) -> u64 {
        self.device_bytes
    }

    /// Context length the GGUF was trained for (`qwen3.context_length`).
    pub fn trained_ctx(&self) -> usize {
        self.cfg.n_ctx
    }

    /// The KV-relevant shape, for [`crate::kv_plan::served_context`].
    pub fn kv_shape(&self) -> KvShape {
        KvShape {
            n_layer: self.cfg.n_layer,
            n_head_kv: self.cfg.n_head_kv,
            head_dim: self.cfg.head_dim,
            trained_ctx: self.cfg.n_ctx,
        }
    }

    /// Assemble the model with a KV cache of `n_ctx` tokens, stored as f16
    /// when `kv_f16` (half the memory) else f32. `max_binding_bytes` is the
    /// device's largest storage binding; it sizes the prefill chunk.
    /// Shorthand for [`Self::assemble`] + [`Qwen3Resident::allocate_kv`].
    pub fn finish(self, n_ctx: usize, kv_f16: bool, max_binding_bytes: u64) -> Result<Qwen3Resident> {
        let mut model = self.assemble(kv_f16, max_binding_bytes)?;
        model.allocate_kv(n_ctx);
        Ok(model)
    }

    /// Assemble the model's weights with no KV cache yet. The caller then
    /// [`Qwen3Resident::allocate_kv`]s, and may retry smaller contexts when the
    /// allocation fails without re-uploading any weight.
    pub fn assemble(mut self, kv_f16: bool, max_binding_bytes: u64) -> Result<Qwen3Resident> {
        let tok_embd = self.matrix("token_embd.weight")?;
        let mut cfg = self.cfg;
        cfg.n_vocab = tok_embd.n_out as usize;
        ensure!(
            tok_embd.n_in as usize == cfg.n_embd,
            "token_embd n_embd {} != embedding_length {}",
            tok_embd.n_in,
            cfg.n_embd
        );
        // Bonsai 8B ships an untied lm_head; 1.7B / 4B tie it to the embeddings.
        let output = match self.tensors.remove("output.weight") {
            Some(Slot::Matrix(w)) => w,
            Some(Slot::Vector(_)) => return Err(anyhow!("output.weight must be a matrix")),
            None => tok_embd.clone(),
        };
        ensure!(
            output.n_in as usize == cfg.n_embd && output.n_out as usize == cfg.n_vocab,
            "output.weight is {}x{}, expected {}x{}",
            output.n_out,
            output.n_in,
            cfg.n_vocab,
            cfg.n_embd
        );
        self.cfg = cfg;
        let output_norm = self.vector("output_norm.weight", self.cfg.n_embd)?;
        let mut layers = Vec::with_capacity(self.cfg.n_layer);
        for i in 0..self.cfg.n_layer {
            layers.push(self.layer(i)?);
        }

        let kv_row = self.cfg.n_head_kv * self.cfg.head_dim;
        ensure!(!kv_f16 || kv_row % 2 == 0, "f16 KV needs an even n_head_kv·head_dim, got {kv_row}");
        let half = self.cfg.n_rot.min(self.cfg.head_dim) / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| 1.0 / self.cfg.rope_freq_base.powf((2 * i) as f32 / (2 * half) as f32))
            .collect();
        let inv_freq = self.gpu.upload_f32(&inv);

        Ok(Qwen3Resident {
            gpu: self.gpu,
            cfg: self.cfg,
            n_ctx: 0,
            kv_f16,
            max_binding_bytes,
            prefill_chunk: 1,
            params: self.params,
            inv_freq,
            tok_embd,
            output,
            output_norm,
            layers,
            k: Vec::new(),
            v: Vec::new(),
            n_past: 0,
        })
    }

    fn matrix(&mut self, name: &str) -> Result<Weight> {
        match self.tensors.remove(name) {
            Some(Slot::Matrix(w)) => Ok(w),
            Some(Slot::Vector(_)) => Err(anyhow!("{name} must be a matrix")),
            None => Err(anyhow!("missing tensor {name}")),
        }
    }

    fn vector(&mut self, name: &str, len: usize) -> Result<DeviceTensor> {
        match self.tensors.remove(name) {
            Some(Slot::Vector(v)) if v.len == len => Ok(v),
            Some(Slot::Vector(v)) => Err(anyhow!("{name} has {} elements, expected {len}", v.len)),
            Some(Slot::Matrix(_)) => Err(anyhow!("{name} must be a vector")),
            None => Err(anyhow!("missing tensor {name}")),
        }
    }

    fn layer(&mut self, i: usize) -> Result<Layer> {
        let c = &self.cfg;
        let (e, d, q_out, kv_out, ff) = (
            c.n_embd,
            c.head_dim,
            c.n_head * c.head_dim,
            c.n_head_kv * c.head_dim,
            c.n_ff,
        );
        let p = format!("blk.{i}");
        let layer = Layer {
            attn_norm: self.vector(&format!("{p}.attn_norm.weight"), e)?,
            wq: self.matrix(&format!("{p}.attn_q.weight"))?,
            wk: self.matrix(&format!("{p}.attn_k.weight"))?,
            wv: self.matrix(&format!("{p}.attn_v.weight"))?,
            wo: self.matrix(&format!("{p}.attn_output.weight"))?,
            q_norm: self.vector(&format!("{p}.attn_q_norm.weight"), d)?,
            k_norm: self.vector(&format!("{p}.attn_k_norm.weight"), d)?,
            ffn_norm: self.vector(&format!("{p}.ffn_norm.weight"), e)?,
            gate: self.matrix(&format!("{p}.ffn_gate.weight"))?,
            up: self.matrix(&format!("{p}.ffn_up.weight"))?,
            down: self.matrix(&format!("{p}.ffn_down.weight"))?,
        };
        for (w, n_in, n_out, name) in [
            (&layer.wq, e, q_out, "attn_q"),
            (&layer.wk, e, kv_out, "attn_k"),
            (&layer.wv, e, kv_out, "attn_v"),
            (&layer.wo, q_out, e, "attn_output"),
            (&layer.gate, e, ff, "ffn_gate"),
            (&layer.up, e, ff, "ffn_up"),
            (&layer.down, ff, e, "ffn_down"),
        ] {
            ensure!(
                w.n_in as usize == n_in && w.n_out as usize == n_out,
                "blk.{i}.{name} is {}x{} (out x in), expected {n_out}x{n_in}",
                w.n_out,
                w.n_in
            );
        }
        Ok(layer)
    }
}
