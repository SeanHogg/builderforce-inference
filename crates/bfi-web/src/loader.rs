//! `BfiEngine.load`: stream GGUF bytes from the host onto the engine's backend.
//!
//! The host owns download and cache and hands over a `ReadableStream<Uint8Array>`;
//! [`crate::stream::pump`] turns it into whole tensors and the backend's sink places
//! them — packed GPU buffers ([`crate::load_gpu`]) or packed wasm memory
//! ([`crate::load_cpu`]). Either way `Q1_0` stays 18 B / 128 weights and the host never
//! holds the whole file.

use std::rc::Rc;

use bfi_gguf::ModelHeader;
use bfi_runtime::Tokenizer;
use js_sys::{Function, Object};
use wasm_bindgen::JsValue;

use crate::device::Device;
use crate::errors::{classify, Kind};
use crate::js;
use crate::runner::Runner;

/// `load` options (`{ kvBudgetBytes?: number }`).
#[derive(Debug, Clone, Copy)]
pub struct LoadOptions {
    /// Total KV-cache budget; `None` = the backend's default (2 GiB on WebGPU, 512 MiB
    /// on the CPU). With 1-bit weights the cache, not the model, is what fills memory.
    pub kv_budget_bytes: Option<u64>,
}

impl LoadOptions {
    pub fn from_js(opts: Option<&Object>) -> Self {
        let kv_budget_bytes = opts
            .map(|o| js::get(o, "kvBudgetBytes"))
            .and_then(|v| v.as_f64())
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|v| v as u64);
        Self { kv_budget_bytes }
    }
}

/// A loaded model plus everything `chat` needs between turns.
pub struct Loaded {
    pub model: Runner,
    pub tokenizer: Tokenizer,
    pub template: Option<String>,
    /// Token ids currently in the KV cache (prompt-prefix reuse across turns).
    pub cached: Vec<u32>,
    pieces: Option<Rc<Vec<String>>>,
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
    pub fn pieces(&mut self) -> Rc<Vec<String>> {
        if let Some(p) = &self.pieces {
            return p.clone();
        }
        let n = self.model.n_vocab();
        let p: Rc<Vec<String>> = Rc::new(
            (0..n as u32)
                .map(|id| self.tokenizer.decode(&[id]).unwrap_or_default())
                .collect(),
        );
        self.pieces = Some(p.clone());
        p
    }

    /// `{ arch, backend, contextTokens, vocab, params, kvF16 }`; `contextTokens` is what
    /// was actually allocated.
    pub fn info(&self) -> JsValue {
        let m = &self.model;
        js::object(&[
            ("arch", JsValue::from_str(m.architecture())),
            ("backend", JsValue::from_str(m.backend())),
            ("contextTokens", JsValue::from_f64(m.n_ctx() as f64)),
            ("kvF16", JsValue::from_bool(m.kv_f16())),
            ("vocab", JsValue::from_f64(m.n_vocab() as f64)),
            ("params", JsValue::from_f64(m.params() as f64)),
        ])
    }
}

/// The GGUF's tokenizer and chat template — both backends build them from the header.
pub fn tokenizer_for(header: &ModelHeader) -> Result<(Tokenizer, Option<String>), JsValue> {
    let tokenizer = Tokenizer::from_header(header).map_err(|e| classify(Kind::LoadFailed, &e))?;
    let template = tokenizer.chat_template().map(str::to_string);
    Ok((tokenizer, template))
}

pub async fn load(
    device: &Device,
    chunks: &Object,
    total_bytes: f64,
    on_progress: Option<&Function>,
    opts: LoadOptions,
) -> Result<Loaded, JsValue> {
    match device {
        Device::Gpu { gpu, adapter } => {
            crate::load_gpu::load(gpu, adapter, chunks, total_bytes, on_progress, opts).await
        }
        Device::Cpu => crate::load_cpu::load(chunks, total_bytes, on_progress, opts).await,
    }
}
