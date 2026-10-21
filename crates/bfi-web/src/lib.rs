//! `bfi-web`: bfi's own engine in the browser, as the `BfiEngine` JS class
//! (Builderforce spec 34 §3.1). Runs 1-bit Bonsai (PrismML, GGUF `Q1_0`, `qwen3`
//! architecture) on WebGPU through CubeCL's wgpu runtime — or, with no GPU, on the
//! CPU through the same `bfi-core` kernels the native `bfi` runs (wasm SIMD128).
//!
//! - [`support`]: every backend's capabilities (WebGPU adapter probe, no device).
//! - [`device`]: the backend `create` opens (`webgpu` / `cpu` / `auto`).
//! - [`stream`] → [`load_gpu`] / [`load_cpu`]: `ReadableStream` → `TensorStream` →
//!   packed per-tensor GPU buffers, or packed wasm memory.
//! - [`runner`] + [`chat`]: OpenAI `chat.completions` → `bfi_runtime::openai` turn →
//!   chunk JSON, on either backend.
//! - [`plan`]: `BfiEngine.contextFor`, the served-context prediction.
//! - [`errors`]: stable `Error.message` prefixes.
//!
//! Natively this crate is empty (all dependencies are wasm32-only); build it with
//! `wasm-pack build crates/bfi-web --target web --release --out-dir pkg`.

#[cfg(target_arch = "wasm32")]
mod chat;
#[cfg(target_arch = "wasm32")]
mod device;
#[cfg(target_arch = "wasm32")]
mod engine;
#[cfg(target_arch = "wasm32")]
mod errors;
#[cfg(target_arch = "wasm32")]
mod js;
#[cfg(target_arch = "wasm32")]
mod load_cpu;
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
mod load_gpu;
#[cfg(target_arch = "wasm32")]
mod loader;
#[cfg(target_arch = "wasm32")]
mod plan;
#[cfg(target_arch = "wasm32")]
mod runner;
#[cfg(target_arch = "wasm32")]
mod stream;
#[cfg(target_arch = "wasm32")]
mod support;
#[cfg(target_arch = "wasm32")]
mod threads;

#[cfg(target_arch = "wasm32")]
pub use engine::BfiEngine;
#[cfg(all(target_arch = "wasm32", feature = "threads"))]
pub use threads::{init_threads, thread_check};
