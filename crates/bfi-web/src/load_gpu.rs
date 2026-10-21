//! The WebGPU backend's load: every tensor uploaded packed into its own GPU buffer as
//! the stream completes it (inside the device's error scopes, so an out-of-memory
//! upload fails the load), then a KV cache sized by `kv_plan` and allocated with
//! halving retries.

use bfi_core::resident::{Gpu, ResidentError};
use bfi_gguf::{ModelHeader, StreamedTensor};
use bfi_models::kv_plan::{
    kv_buffer_bytes, kv_buffer_limit, retry_contexts, served_context, DEFAULT_KV_BUDGET_BYTES,
};
use bfi_models::Qwen3ResidentLoader;
use bfi_runtime::Tokenizer;
use js_sys::{Function, Object};
use wasm_bindgen::JsValue;

use crate::errors::{classify, error, gpu_error, Kind};
use crate::loader::{tokenizer_for, LoadOptions, Loaded};
use crate::runner::Runner;
use crate::stream::{pump, TensorSink};
use crate::support::AdapterInfo;

/// The largest single storage buffer the adapter allows (same rule as `contextFor`).
pub fn max_buffer_bytes(adapter: &AdapterInfo) -> u64 {
    kv_buffer_limit(
        Some(adapter.max_binding_bytes as u64),
        Some(adapter.max_buffer_bytes as u64),
    )
}

struct GpuSink<'a> {
    gpu: &'a Gpu,
    pending: Option<(Qwen3ResidentLoader, (Tokenizer, Option<String>))>,
}

impl TensorSink for GpuSink<'_> {
    async fn accept(&mut self, header: &ModelHeader, done: Vec<StreamedTensor>) -> Result<(), JsValue> {
        if done.is_empty() && self.pending.is_some() {
            return Ok(());
        }
        let gpu = self.gpu;
        let pending = &mut self.pending;
        gpu.checked("weight upload", || place(gpu, header, pending, done))
            .await
            .map_err(|e| gpu_error(Kind::LoadFailed, &e))?
    }
}

/// Create the loader + tokenizer once, then upload the tensors this chunk completed.
fn place(
    gpu: &Gpu,
    header: &ModelHeader,
    pending: &mut Option<(Qwen3ResidentLoader, (Tokenizer, Option<String>))>,
    done: Vec<StreamedTensor>,
) -> Result<(), JsValue> {
    if pending.is_none() {
        let loader = Qwen3ResidentLoader::new(gpu.clone(), header).map_err(|e| classify(Kind::LoadFailed, &e))?;
        *pending = Some((loader, tokenizer_for(header)?));
    }
    let (loader, _) = pending.as_mut().expect("set above");
    for t in done {
        loader.accept(t).map_err(|e| classify(Kind::LoadFailed, &e))?;
    }
    Ok(())
}

pub async fn load(
    gpu: &Gpu,
    adapter: &AdapterInfo,
    chunks: &Object,
    total_bytes: f64,
    on_progress: Option<&Function>,
    opts: LoadOptions,
) -> Result<Loaded, JsValue> {
    let mut sink = GpuSink { gpu, pending: None };
    pump(chunks, total_bytes, on_progress, &mut sink).await?;
    let (loader, tokenizer) = sink.pending.ok_or_else(|| error(Kind::LoadFailed, "empty GGUF stream"))?;
    let kv_budget = opts.kv_budget_bytes.unwrap_or(DEFAULT_KV_BUDGET_BYTES);

    // f16 KV when the adapter offers shader-f16 (half the bytes → twice the
    // context); f32 otherwise. Start at the planned context (what `contextFor`
    // predicts) and halve on allocation failure down to the floor, never above
    // the trained context. A failure is either the probe or the real
    // allocation (inside error scopes) reporting out-of-memory.
    let shape = loader.kv_shape();
    let kv_f16 = adapter.f16;
    let planned = served_context(&shape, kv_f16, kv_budget, max_buffer_bytes(adapter));
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
                "the KV cache does not fit: planned {planned} tokens; tried {} and none allocated (kvBudgetBytes {kv_budget})",
                failures.join(", "),
            ),
        ));
    }
    Ok(Loaded::new(Runner::Gpu(model), tokenizer))
}
