//! The CPU backend's load: the same qwen3 the native `bfi` runs (`Qwen3StreamLoader` on
//! bfi-core's `CpuBackend`), every matrix kept PACKED in host memory as the stream
//! completes it, then a `Session` with an f32 KV cache sized by
//! `kv_plan::cpu_served_context` — exactly what `contextFor({ backend: "cpu" })`
//! predicts under the same [`HostProfile`].
//!
//! The header alone says how big the weights are, so a model whose weights + planned KV
//! + working set cannot fit the profile's ceiling (wasm32's 4 GiB, or natively the RAM
//! share the host plans to use) is refused with `cpu_memory_exceeded` before a single
//! weight byte is placed.

use std::sync::Arc;

use bfi_core::{Backend, CpuBackend};
use bfi_gguf::{ModelHeader, StreamedTensor};
use bfi_models::kv_plan::{cpu_memory_bytes, cpu_served_context};
use bfi_models::Qwen3StreamLoader;
use bfi_runtime::Tokenizer;

use crate::errors::{classify, error, gguf, EngineError, Kind};
use crate::host::HostProfile;
use crate::loader::{tokenizer_for, LoadOptions, Loaded};
use crate::runner::Runner;
use crate::stream::{pump, ByteSource, TensorSink};

struct CpuSink {
    kv_budget: u64,
    memory_cap: u64,
    planned: usize,
    pending: Option<(Qwen3StreamLoader, (Tokenizer, Option<String>))>,
}

impl CpuSink {
    fn start(&mut self, header: &ModelHeader) -> Result<(), EngineError> {
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend);
        let loader = Qwen3StreamLoader::new(backend, header).map_err(|e| classify(Kind::LoadFailed, &e))?;
        let shape = loader.kv_shape();
        let planned = cpu_served_context(&shape, self.kv_budget);
        let weights = header.weight_bytes().map_err(gguf)?;
        let need = cpu_memory_bytes(weights, &shape, planned);
        if need > self.memory_cap {
            return Err(error(
                Kind::CpuMemoryExceeded,
                format!(
                    "{weights} bytes of weights + a {planned}-token KV cache need {need} bytes, over this host's {} for the CPU backend; lower kvBudgetBytes ({}) or use the GPU",
                    self.memory_cap, self.kv_budget
                ),
            ));
        }
        self.planned = planned;
        self.pending = Some((loader, tokenizer_for(header)?));
        Ok(())
    }
}

impl TensorSink for CpuSink {
    async fn accept(&mut self, header: &ModelHeader, done: Vec<StreamedTensor>) -> Result<(), EngineError> {
        if self.pending.is_none() {
            self.start(header)?;
        }
        let (loader, _) = self.pending.as_mut().expect("started above");
        for t in done {
            loader.accept(t).map_err(|e| classify(Kind::LoadFailed, &e))?;
        }
        Ok(())
    }
}

pub async fn load(
    profile: &HostProfile,
    source: &mut impl ByteSource,
    total_bytes: f64,
    on_progress: &mut dyn FnMut(f64, f64),
    opts: LoadOptions,
) -> Result<Loaded, EngineError> {
    let mut sink = CpuSink {
        kv_budget: opts.kv_budget_bytes.unwrap_or(profile.cpu_kv_budget_bytes),
        memory_cap: profile.cpu_memory_cap,
        planned: 0,
        pending: None,
    };
    pump(source, total_bytes, on_progress, &mut sink).await?;
    let (loader, tokenizer) = sink.pending.ok_or_else(|| error(Kind::LoadFailed, "empty GGUF stream"))?;
    let session = loader
        .finish(sink.planned)
        .map_err(|e| classify(Kind::LoadFailed, &e))?;
    Ok(Loaded::new(Runner::Cpu(session), tokenizer))
}
