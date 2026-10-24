//! `bfi-web`: bfi's own engine in the browser, as the `BfiEngine` JS class
//! (Builderforce spec 34 §3.1). Runs 1-bit Bonsai (PrismML, GGUF `Q1_0`, `qwen3`
//! architecture) on WebGPU through CubeCL's wgpu runtime — or, with no GPU, on the
//! CPU through the same `bfi-core` kernels the native `bfi` runs (wasm SIMD128).
//!
//! The engine is `bfi-engine` (state machine, load, chat, `contextFor`), shared with
//! `bfi-node`; this crate is its wasm-bindgen face — what only a browser realm has:
//!
//! - [`engine`]: the exported class, over `bfi_engine::EngineState`.
//! - [`support`]: the capability probe (`navigator.gpu` adapter, `navigator.deviceMemory`).
//! - [`js`]: `Reflect` helpers, the realm's [`bfi_engine::Host`] (clock + macrotask
//!   yield), the `ReadableStream` byte source, JSON ↔ JS.
//! - [`threads`]: the threaded artifact's rayon pool.
//!
//! Natively this crate is empty (all dependencies are wasm32-only); build it with
//! `wasm-pack build crates/bfi-web --target web --release --out-dir pkg`.

#[cfg(target_arch = "wasm32")]
mod engine;
#[cfg(target_arch = "wasm32")]
mod js;
#[cfg(target_arch = "wasm32")]
mod support;
#[cfg(target_arch = "wasm32")]
mod threads;

#[cfg(target_arch = "wasm32")]
pub use engine::BfiEngine;
#[cfg(all(target_arch = "wasm32", feature = "threads"))]
pub use threads::{init_threads, thread_check};
