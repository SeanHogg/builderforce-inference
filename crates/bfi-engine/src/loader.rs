//! `BfiEngine.load`: stream GGUF bytes from the host onto the engine's backend.
//!
//! The host owns download and cache and hands over a [`ByteSource`];
//! [`crate::stream::pump`] turns it into whole tensors and the backend's sink places
//! them — packed GPU buffers (`load_gpu`) or packed host memory (`load_host`). Either
//! way `Q1_0` stays 18 B / 128 weights and the host never holds the whole file.

use std::sync::Arc;

use bfi_gguf::ModelHeader;
use bfi_runtime::Tokenizer;
use serde_json::{json, Value};

use crate::device::Device;
use crate::errors::{classify, EngineError, Kind};
use crate::host::HostProfile;
use crate::runner::Runner;
use crate::stream::ByteSource;

/// `load` options (`{ kvBudgetBytes?: number }`).
#[derive(Debug, Clone, Copy, Default)]
pub struct LoadOptions {
    /// Total KV-cache budget; `None` = the backend's default (2 GiB on the GPU, the
    /// [`HostProfile`]'s on the CPU). With 1-bit weights the cache, not the model, is what
    /// fills memory.
    pub kv_budget_bytes: Option<u64>,
}

impl LoadOptions {
    /// From the options object as JSON (absent / invalid fields = defaults).
    pub fn from_json(opts: Option<&Value>) -> Self {
        let kv_budget_bytes = opts
            .and_then(|o| o.get("kvBudgetBytes"))
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|v| v as u64);
        Self { kv_budget_bytes }
    }
}

/// What `load` resolves with: `{ arch, backend, contextTokens, vocab, params, kvF16 }`;
/// `contextTokens` is what was actually allocated.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelInfo {
    pub arch: &'static str,
    pub backend: &'static str,
    pub context_tokens: usize,
    pub kv_f16: bool,
    pub vocab: usize,
    pub params: u64,
}

impl ModelInfo {
    pub fn to_json(&self) -> Value {
        json!({
            "arch": self.arch,
            "backend": self.backend,
            "contextTokens": self.context_tokens,
            "kvF16": self.kv_f16,
            "vocab": self.vocab,
            "params": self.params,
        })
    }
}

/// A loaded model plus everything `chat` needs between turns.
pub struct Loaded {
    pub model: Runner,
    pub tokenizer: Tokenizer,
    pub template: Option<String>,
    /// Token ids currently in the KV cache (prompt-prefix reuse across turns).
    pub cached: Vec<u32>,
    pieces: Option<Arc<Vec<String>>>,
}

impl Loaded {
    pub fn new(model: Runner, (tokenizer, template): (Tokenizer, Option<String>)) -> Self {
        Self {
            model,
            tokenizer,
            template,
            cached: Vec::new(),
            pieces: None,
        }
    }

    /// `tokenizer.decode(&[id])` for every id; built on first constrained request.
    pub fn pieces(&mut self) -> Arc<Vec<String>> {
        if let Some(p) = &self.pieces {
            return p.clone();
        }
        let n = self.model.n_vocab();
        let p: Arc<Vec<String>> = Arc::new(
            (0..n as u32)
                .map(|id| self.tokenizer.decode(&[id]).unwrap_or_default())
                .collect(),
        );
        self.pieces = Some(p.clone());
        p
    }

    pub fn info(&self) -> ModelInfo {
        let m = &self.model;
        ModelInfo {
            arch: m.architecture(),
            backend: m.backend(),
            context_tokens: m.n_ctx(),
            kv_f16: m.kv_f16(),
            vocab: m.n_vocab(),
            params: m.params(),
        }
    }
}

/// The GGUF's tokenizer and chat template — both backends build them from the header.
pub fn tokenizer_for(header: &ModelHeader) -> Result<(Tokenizer, Option<String>), EngineError> {
    let tokenizer = Tokenizer::from_header(header).map_err(|e| classify(Kind::LoadFailed, &e))?;
    let template = tokenizer.chat_template().map(str::to_string);
    Ok((tokenizer, template))
}

pub async fn load(
    device: &Device,
    profile: &HostProfile,
    source: &mut impl ByteSource,
    total_bytes: f64,
    on_progress: &mut dyn FnMut(f64, f64),
    opts: LoadOptions,
) -> Result<Loaded, EngineError> {
    match device {
        #[cfg(feature = "webgpu")]
        Device::Gpu { gpu, adapter } => {
            crate::load_gpu::load(gpu, adapter, source, total_bytes, on_progress, opts).await
        }
        Device::Cpu => crate::load_host::load(profile, source, total_bytes, on_progress, opts).await,
    }
}
