//! `bfi-engine`: the `BfiEngine` contract (Builderforce spec 34 §3.1) with the host
//! taken out — everything `bfi-web` (wasm-bindgen, the browser and Node workers) and
//! `bfi-node` (napi, the editor's extension host) share:
//!
//! - [`state`]: the engine's exclusive load / chat / unload state machine.
//! - [`device`]: the backend an engine owns (the GPU-resident engine, or the CPU).
//! - [`stream`] → [`loader`]: a host's GGUF byte chunks → packed tensors on the backend.
//! - [`runner`] + [`chat`]: OpenAI `chat.completions` → `bfi_runtime::openai` turn →
//!   chunk JSON.
//! - [`plan`]: `contextFor`, the served-context prediction.
//! - [`support`]: the capability record's shape (each host probes its own facts).
//! - [`errors`]: stable message prefixes.
//!
//! What differs per host is a [`Host`] (clock + yield), a [`ByteSource`] (where the GGUF
//! bytes come from), a [`HostProfile`] (the CPU backend's memory rules) and how its
//! device is opened. The glue crates own exactly those.

mod chat;
mod device;
mod errors;
mod host;
#[cfg(feature = "webgpu")]
mod load_gpu;
mod load_host;
mod loader;
mod plan;
mod runner;
mod state;
mod stream;
mod support;

pub use device::{BackendChoice, Device};
pub use errors::{classify, error, EngineError, Kind};
pub use host::{Host, HostProfile};
pub use loader::{LoadOptions, Loaded, ModelInfo};
pub use plan::context_for;
pub use state::EngineState;
pub use stream::ByteSource;
pub use support::{AdapterInfo, CpuInfo, Support};
