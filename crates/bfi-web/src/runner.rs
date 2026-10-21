//! The loaded model on its backend, behind the handful of operations `chat` drives —
//! so the chat loop (prompt-prefix reuse, abortable prefill, decode) is written once.
//!
//! - WebGPU: `Qwen3Resident` enqueues kernels; the logits readback is the one host sync
//!   per step (and yields to the event loop).
//! - CPU: `Session` computes synchronously on the calling thread; each step is followed
//!   by a macrotask yield ([`js::yield_now`]) so an abort or a repaint gets in.

#[cfg(feature = "webgpu")]
use bfi_models::Qwen3Resident;
use bfi_models::Session;
use wasm_bindgen::JsValue;

#[cfg(feature = "webgpu")]
use crate::errors::gpu_error;
use crate::errors::{classify, Kind};
use crate::js;

/// Prompt tokens per CPU forward between yields. CPU throughput does not grow with the
/// group (the kernels are compute-bound per token), so a small group only bounds how
/// long the thread is held — and how late an abort lands.
pub const CPU_PREFILL_YIELD_TOKENS: usize = 4;

pub enum Runner {
    #[cfg(feature = "webgpu")]
    Gpu(Qwen3Resident),
    Cpu(Session),
}

impl Runner {
    pub fn backend(&self) -> &'static str {
        match self {
            #[cfg(feature = "webgpu")]
            Self::Gpu(_) => "webgpu",
            Self::Cpu(_) => "cpu",
        }
    }
    pub fn architecture(&self) -> &'static str {
        match self {
            #[cfg(feature = "webgpu")]
            Self::Gpu(m) => m.architecture(),
            Self::Cpu(s) => s.architecture(),
        }
    }
    pub fn n_ctx(&self) -> usize {
        match self {
            #[cfg(feature = "webgpu")]
            Self::Gpu(m) => m.n_ctx(),
            Self::Cpu(s) => s.n_ctx(),
        }
    }
    pub fn n_vocab(&self) -> usize {
        match self {
            #[cfg(feature = "webgpu")]
            Self::Gpu(m) => m.n_vocab(),
            Self::Cpu(s) => s.n_vocab(),
        }
    }
    pub fn params(&self) -> u64 {
        match self {
            #[cfg(feature = "webgpu")]
            Self::Gpu(m) => m.params(),
            Self::Cpu(s) => s.params(),
        }
    }
    /// KV cache in f16 (WebGPU with `shader-f16`); the CPU cache is f32.
    pub fn kv_f16(&self) -> bool {
        match self {
            #[cfg(feature = "webgpu")]
            Self::Gpu(m) => m.kv_f16(),
            Self::Cpu(_) => false,
        }
    }
    pub fn n_past(&self) -> usize {
        match self {
            #[cfg(feature = "webgpu")]
            Self::Gpu(m) => m.n_past(),
            Self::Cpu(s) => s.n_past(),
        }
    }
    pub fn rewind(&mut self, n: usize) {
        match self {
            #[cfg(feature = "webgpu")]
            Self::Gpu(m) => m.rewind(n),
            Self::Cpu(s) => s.rewind(n),
        }
    }
    /// Prompt tokens fed between host yields during prefill.
    pub fn prefill_group(&self) -> usize {
        match self {
            #[cfg(feature = "webgpu")]
            Self::Gpu(m) => m.prefill_yield_tokens(),
            Self::Cpu(_) => CPU_PREFILL_YIELD_TOKENS,
        }
        .max(1)
    }

    /// Append `tokens` without logits (a leading prompt group), then yield.
    pub async fn prefill(&mut self, tokens: &[u32]) -> Result<(), JsValue> {
        match self {
            #[cfg(feature = "webgpu")]
            Self::Gpu(m) => {
                m.prefill(tokens).map_err(|e| classify(Kind::Engine, &e))?;
                // A one-word readback: the queue drains and the event loop runs.
                m.gpu().drain().await.map_err(|e| gpu_error(Kind::Engine, &e))
            }
            Self::Cpu(s) => {
                s.forward(tokens).map_err(|e| classify(Kind::Engine, &e))?;
                js::yield_now().await;
                Ok(())
            }
        }
    }

    /// Append `tokens` and return the last token's logits on the host.
    pub async fn logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>, JsValue> {
        match self {
            #[cfg(feature = "webgpu")]
            Self::Gpu(m) => {
                let logits = m.forward(tokens).map_err(|e| classify(Kind::Engine, &e))?;
                m.gpu().read_f32(&logits).await.map_err(|e| gpu_error(Kind::Engine, &e))
            }
            Self::Cpu(s) => {
                let logits = s.forward(tokens).map_err(|e| classify(Kind::Engine, &e))?;
                js::yield_now().await;
                Ok(logits)
            }
        }
    }
}
