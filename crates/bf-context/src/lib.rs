//! bf-context — the Builderforce local context service.
//!
//! The desktop app and headless binary share this indexed workspace registry
//! and its loopback HTTP service.

pub mod activity;
pub mod api;
pub mod client;
pub mod compressor;
pub mod discovery;
pub mod embedder;
pub mod paths;
pub mod registry;
pub mod server;

pub use registry::Registry;
