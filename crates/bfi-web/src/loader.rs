//! `BfiEngine.load`: stream GGUF bytes from the host into per-tensor GPU buffers.
//!
//! The host owns download and cache and hands over a `ReadableStream<Uint8Array>`.
//! Each chunk is copied into wasm memory, fed to [`TensorStream`], and every
//! tensor it completes is uploaded packed (`Q1_0` stays 18 B / 128 weights) and
//! dropped. Peak host memory is the GGUF header plus the largest tensor (the
//! 8B embedding table, ~88 MB) — never the whole 1.15 GB file.

use std::rc::Rc;

use bfi_core::resident::Gpu;
use bfi_gguf::TensorStream;
use bfi_models::kv_plan::{kv_buffer_bytes, retry_contexts, served_context, DEFAULT_KV_BUDGET_BYTES};
use bfi_models::{Qwen3Resident, Qwen3ResidentLoader};
use bfi_runtime::Tokenizer;
use js_sys::{Function, Object, Uint8Array};
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

use crate::errors::{classify, error, gguf, Kind};
use crate::js;
use crate::support::AdapterInfo;

/// `load` options (`{ kvBudgetBytes?: number }`).
#[derive(Debug, Clone, Copy)]
pub struct LoadOptions {
    /// Total KV-cache budget. With 1-bit weights the cache, not the model, is
    /// what fills VRAM. Default [`DEFAULT_KV_BUDGET_BYTES`] (2 GiB).
    pub kv_budget_bytes: u64,
}

impl LoadOptions {
    pub fn from_js(opts: Option<&Object>) -> Self {
        let kv_budget_bytes = opts
            .map(|o| js::get(o, "kvBudgetBytes"))
            .and_then(|v| v.as_f64())
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|v| v as u64)
            .unwrap_or(DEFAULT_KV_BUDGET_BYTES);
        Self { kv_budget_bytes }
    }
}

/// The largest single storage buffer the adapter allows.
pub fn max_buffer_bytes(adapter: &AdapterInfo) -> u64 {
    let binding = adapter.max_binding_bytes;
    let buffer = if adapter.max_buffer_bytes > 0.0 { adapter.max_buffer_bytes } else { binding };
    binding.min(buffer) as u64
}

/// A loaded model plus everything `chat` needs between turns.
pub struct Loaded {
    pub model: Qwen3Resident,
    pub tokenizer: Tokenizer,
    pub template: Option<String>,
    /// Token ids currently in the KV cache (prompt-prefix reuse across turns).
    pub cached: Vec<u32>,
    pieces: Option<Rc<Vec<String>>>,
}

impl Loaded {
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

    /// `{ arch, contextTokens, vocab, params, kvF16 }`; `contextTokens` is what
    /// was actually allocated.
    pub fn info(&self) -> JsValue {
        js::object(&[
            ("arch", JsValue::from_str(self.model.architecture())),
            ("contextTokens", JsValue::from_f64(self.model.n_ctx() as f64)),
            ("kvF16", JsValue::from_bool(self.model.kv_f16())),
            ("vocab", JsValue::from_f64(self.model.n_vocab() as f64)),
            ("params", JsValue::from_f64(self.model.params() as f64)),
        ])
    }
}

struct Pending {
    loader: Qwen3ResidentLoader,
    tokenizer: Tokenizer,
    template: Option<String>,
}

pub async fn load(
    gpu: &Gpu,
    adapter: &AdapterInfo,
    chunks: &Object,
    total_bytes: f64,
    on_progress: Option<&Function>,
    opts: LoadOptions,
) -> Result<Loaded, JsValue> {
    let reader = js::call(chunks, "getReader", &[])
        .map_err(|e| error(Kind::LoadFailed, format!("chunks is not a ReadableStream: {e:?}")))?;
    let mut stream = TensorStream::new();
    let mut pending: Option<Pending> = None;
    let mut loaded = 0f64;
    let result = async {
        loop {
            let read = js::call(&reader, "read", &[])
                .map_err(|e| error(Kind::LoadFailed, format!("reader.read threw: {e:?}")))?;
            let res = JsFuture::from(js_sys::Promise::from(read))
                .await
                .map_err(|e| error(Kind::LoadFailed, format!("stream errored: {e:?}")))?;
            if js::get(&res, "done").as_bool() == Some(true) {
                break;
            }
            let bytes = Uint8Array::new(&js::get(&res, "value")).to_vec();
            loaded += bytes.len() as f64;
            let done = stream.push(&bytes).map_err(gguf)?;
            drop(bytes);
            accept(gpu, &stream, &mut pending, done)?;
            if let Some(cb) = on_progress {
                let _ = cb.call2(
                    &JsValue::NULL,
                    &JsValue::from_f64(loaded),
                    &JsValue::from_f64(total_bytes),
                );
            }
        }
        let tail = stream.flush_header().map_err(gguf)?;
        accept(gpu, &stream, &mut pending, tail)?;
        stream.finish().map_err(|_| {
            error(
                Kind::LoadFailed,
                format!("stream ended before every tensor arrived ({loaded} of {total_bytes} bytes)"),
            )
        })?;
        Ok::<(), JsValue>(())
    }
    .await;
    let _ = js::call(&reader, "releaseLock", &[]);
    result?;

    let Pending {
        loader,
        tokenizer,
        template,
    } = pending.ok_or_else(|| error(Kind::LoadFailed, "empty GGUF stream"))?;
    // f16 KV when the adapter offers shader-f16 (half the bytes → twice the
    // context); f32 otherwise. Start at the planned context and halve on
    // allocation failure down to the floor, never above the trained context.
    let shape = loader.kv_shape();
    let kv_f16 = adapter.f16;
    let max_buffer = max_buffer_bytes(adapter);
    let planned = served_context(&shape, kv_f16, opts.kv_budget_bytes, max_buffer);
    let mut n_ctx = None;
    for ctx in retry_contexts(planned, &shape) {
        if gpu.kv_fits(kv_buffer_bytes(&shape, kv_f16, ctx), 2 * shape.n_layer).await {
            n_ctx = Some(ctx);
            break;
        }
    }
    let n_ctx = n_ctx.ok_or_else(|| {
        error(
            Kind::LoadFailed,
            format!(
                "the KV cache does not fit: planned {planned} tokens, the floor failed to allocate (kvBudgetBytes {})",
                opts.kv_budget_bytes
            ),
        )
    })?;
    let model = loader
        .finish(n_ctx, kv_f16, adapter.max_binding_bytes as u64)
        .map_err(|e| classify(Kind::LoadFailed, &e))?;
    Ok(Loaded {
        model,
        tokenizer,
        template,
        cached: Vec::new(),
        pieces: None,
    })
}

/// Create the model loader + tokenizer once the header is known, then upload
/// the tensors this chunk completed.
fn accept(
    gpu: &Gpu,
    stream: &TensorStream,
    pending: &mut Option<Pending>,
    done: Vec<bfi_gguf::StreamedTensor>,
) -> Result<(), JsValue> {
    if pending.is_none() {
        let Some(header) = stream.header() else {
            return Ok(());
        };
        let loader = Qwen3ResidentLoader::new(gpu.clone(), header)
            .map_err(|e| classify(Kind::LoadFailed, &e))?;
        let tokenizer =
            Tokenizer::from_header(header).map_err(|e| classify(Kind::LoadFailed, &e))?;
        let template = tokenizer.chat_template().map(str::to_string);
        *pending = Some(Pending {
            loader,
            tokenizer,
            template,
        });
    }
    let p = pending.as_mut().expect("set above");
    for t in done {
        p.loader
            .accept(t)
            .map_err(|e| classify(Kind::LoadFailed, &e))?;
    }
    Ok(())
}

