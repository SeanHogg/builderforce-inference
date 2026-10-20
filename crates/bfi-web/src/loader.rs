//! `BfiEngine.load`: stream GGUF bytes from the host into per-tensor GPU buffers.
//!
//! The host owns download and cache and hands over a `ReadableStream<Uint8Array>`.
//! Each chunk is copied into wasm memory, fed to [`TensorStream`], and every
//! tensor it completes is uploaded packed (`Q1_0` stays 18 B / 128 weights) and
//! dropped. Peak host memory is the GGUF header plus the largest tensor (the
//! 8B embedding table, ~88 MB) — never the whole 1.15 GB file.

use std::rc::Rc;

use bfi_core::resident::{Gpu, ResidentError};
use bfi_gguf::TensorStream;
use bfi_models::kv_plan::{
    kv_buffer_bytes, kv_buffer_limit, retry_contexts, served_context, DEFAULT_KV_BUDGET_BYTES,
};
use bfi_models::{Qwen3Resident, Qwen3ResidentLoader};
use bfi_runtime::Tokenizer;
use js_sys::{Function, Object, Uint8Array};
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

use crate::errors::{classify, error, gguf, gpu_error, Kind};
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

/// The largest single storage buffer the adapter allows (same rule as `contextFor`).
pub fn max_buffer_bytes(adapter: &AdapterInfo) -> u64 {
    kv_buffer_limit(
        Some(adapter.max_binding_bytes as u64),
        Some(adapter.max_buffer_bytes as u64),
    )
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
            upload(gpu, &stream, &mut pending, done).await?;
            if let Some(cb) = on_progress {
                let _ = cb.call2(
                    &JsValue::NULL,
                    &JsValue::from_f64(loaded),
                    &JsValue::from_f64(total_bytes),
                );
            }
        }
        let tail = stream.flush_header().map_err(gguf)?;
        upload(gpu, &stream, &mut pending, tail).await?;
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
    // context); f32 otherwise. Start at the planned context (what `contextFor`
    // predicts) and halve on allocation failure down to the floor, never above
    // the trained context. A failure is either the probe or the real
    // allocation (inside error scopes) reporting out-of-memory.
    let shape = loader.kv_shape();
    let kv_f16 = adapter.f16;
    let max_buffer = max_buffer_bytes(adapter);
    let planned = served_context(&shape, kv_f16, opts.kv_budget_bytes, max_buffer);
    let mut model = gpu
        .checked("assemble", || loader.assemble(kv_f16, adapter.max_binding_bytes as u64))
        .await
        .map_err(|e| gpu_error(Kind::LoadFailed, &e))?
        .map_err(|e| classify(Kind::LoadFailed, &e))?;
    let mut failures: Vec<String> = Vec::new();
    for ctx in retry_contexts(planned, &shape) {
        if !gpu.kv_fits(kv_buffer_bytes(&shape, kv_f16, ctx), 2 * shape.n_layer).await {
            failures.push(format!("{ctx} (probe)"));
            continue;
        }
        match gpu.checked("KV cache", || model.allocate_kv(ctx)).await {
            Ok(()) => break,
            Err(e @ ResidentError::DeviceLost(_)) => return Err(gpu_error(Kind::LoadFailed, &e)),
            Err(e) => {
                model.release_kv();
                gpu.release_unused();
                failures.push(format!("{ctx} ({e})"));
            }
        }
    }
    if model.n_ctx() == 0 {
        return Err(error(
            Kind::LoadFailed,
            format!(
                "the KV cache does not fit: planned {planned} tokens; tried {} and none allocated (kvBudgetBytes {})",
                failures.join(", "),
                opts.kv_budget_bytes
            ),
        ));
    }
    Ok(Loaded {
        model,
        tokenizer,
        template,
        cached: Vec::new(),
        pieces: None,
    })
}

/// [`accept`] inside the GPU's error scopes, so an out-of-memory weight upload
/// fails the load instead of leaving an invalid buffer behind.
async fn upload(
    gpu: &Gpu,
    stream: &TensorStream,
    pending: &mut Option<Pending>,
    done: Vec<bfi_gguf::StreamedTensor>,
) -> Result<(), JsValue> {
    if done.is_empty() && pending.is_some() {
        return Ok(());
    }
    gpu.checked("weight upload", || accept(gpu, stream, pending, done))
        .await
        .map_err(|e| gpu_error(Kind::LoadFailed, &e))?
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

