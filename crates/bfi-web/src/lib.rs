//! `bfi-web`: bfi's own engine in the browser, as the `BfiEngine` JS class
//! (Builderforce spec 34 §3.1). Runs 1-bit Bonsai (PrismML, GGUF `Q1_0`,
//! `qwen3` architecture) on WebGPU through CubeCL's wgpu runtime.
//!
//! - [`support`]: adapter probe, no device.
//! - [`loader`]: `ReadableStream` → `TensorStream` → packed per-tensor GPU buffers.
//! - [`chat`]: OpenAI `chat.completions` → `bfi_runtime::openai` turn → chunk JSON.
//! - [`plan`]: `BfiEngine.contextFor`, the served-context prediction.
//! - [`errors`]: stable `Error.message` prefixes.
//!
//! Natively this crate is empty (all dependencies are wasm32-only); build it with
//! `wasm-pack build crates/bfi-web --target web --release --out-dir pkg`.

#[cfg(target_arch = "wasm32")]
mod chat;
#[cfg(target_arch = "wasm32")]
mod engine;
#[cfg(target_arch = "wasm32")]
mod errors;
#[cfg(target_arch = "wasm32")]
mod js;
#[cfg(target_arch = "wasm32")]
mod loader;
#[cfg(target_arch = "wasm32")]
mod plan;
#[cfg(target_arch = "wasm32")]
mod support;

#[cfg(target_arch = "wasm32")]
pub use engine::BfiEngine;
