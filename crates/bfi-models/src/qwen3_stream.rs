//! Streaming load of qwen3 onto a host-tensor [`Backend`] — the browser engine's CPU
//! backend. Each tensor arrives from [`bfi_gguf::TensorStream`], goes onto the backend
//! through the SAME `weight_from_bytes` the file loader uses (matrices stay packed:
//! `Q1_0` is 18 bytes per 128 weights in wasm memory), and the model is assembled by the
//! SAME `qwen3::assemble` — this file only collects tensors by name.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, ensure, Result};
use bfi_core::{Backend, Tensor};
use bfi_gguf::{ModelHeader, StreamedTensor};

use crate::kv_plan::KvShape;
use crate::qwen3::{assemble, dense_config, weight_from_bytes, Qwen3Config, ARCH_NAME};
use crate::session::Session;

pub struct Qwen3StreamLoader {
    backend: Arc<dyn Backend>,
    cfg: Qwen3Config,
    tensors: HashMap<String, Tensor>,
    params: u64,
}

impl Qwen3StreamLoader {
    pub fn new(backend: Arc<dyn Backend>, header: &ModelHeader) -> Result<Self> {
        let arch = header
            .architecture()
            .ok_or_else(|| anyhow!("GGUF missing metadata key general.architecture"))?;
        ensure!(
            arch == ARCH_NAME,
            "architecture '{arch}' is not supported by the browser engine (qwen3 only)"
        );
        Ok(Self {
            backend,
            cfg: dense_config(header)?,
            tensors: HashMap::new(),
            params: 0,
        })
    }

    /// The KV-relevant shape, for [`crate::kv_plan::cpu_served_context`].
    pub fn kv_shape(&self) -> KvShape {
        KvShape {
            n_layer: self.cfg.n_layer,
            n_head_kv: self.cfg.n_head_kv,
            head_dim: self.cfg.head_dim,
            trained_ctx: self.cfg.n_ctx,
        }
    }

    /// Put one streamed tensor on the backend; its host bytes are dropped after.
    pub fn accept(&mut self, t: StreamedTensor) -> Result<()> {
        let info = t.info;
        self.params += info.n_elems();
        let tensor = weight_from_bytes(self.backend.as_ref(), &info.name, info.ggml_type, &info.dims, &t.bytes)?;
        drop(t.bytes);
        self.tensors.insert(info.name, tensor);
        Ok(())
    }

    /// Assemble the model and a sequence with a KV cache of `n_ctx` tokens (clamped to
    /// the trained context).
    pub fn finish(self, n_ctx: usize) -> Result<Session> {
        let Self { backend, cfg, mut tensors, params } = self;
        let mut model = assemble(cfg, backend, |_, name| Ok(tensors.remove(name)))?;
        model.cap_context(n_ctx);
        Ok(Session::new(Box::new(model), params))
    }
}

#[cfg(test)]
mod tests;
