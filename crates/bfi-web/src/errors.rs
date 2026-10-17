//! JS `Error`s with stable message prefixes (spec 34 §3.1). Hosts branch on
//! `err.message.startsWith(prefix)`, so these strings are contract.

use std::fmt::Display;

use bfi_core::resident::ResidentError;
use bfi_gguf::GgufError;
use wasm_bindgen::JsValue;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// No WebGPU adapter / device.
    WebgpuUnavailable,
    /// A tensor type the WebGPU engine cannot run (names the tensor).
    UnsupportedType,
    /// Prompt (+ generation) does not fit the loaded context.
    ContextExceeded,
    /// The request's `AbortSignal` fired (or `unload()` was called mid-chat).
    Aborted,
    /// Malformed `chat` request JSON or fields.
    InvalidRequest,
    /// `chat` before `load`.
    NotLoaded,
    /// `load` / `chat` while another `load` / `chat` is running.
    Busy,
    /// The GGUF stream was truncated, malformed, or not a supported architecture.
    LoadFailed,
    /// Anything else (GPU readback, tokenizer).
    Engine,
}

impl Kind {
    pub fn prefix(self) -> &'static str {
        match self {
            Kind::WebgpuUnavailable => "webgpu_unavailable",
            Kind::UnsupportedType => "unsupported_type",
            Kind::ContextExceeded => "context_exceeded",
            Kind::Aborted => "aborted",
            Kind::InvalidRequest => "invalid_request",
            Kind::NotLoaded => "not_loaded",
            Kind::Busy => "busy",
            Kind::LoadFailed => "load_failed",
            Kind::Engine => "engine_error",
        }
    }
}

/// `new Error("<prefix>: <msg>")`; `aborted` is the bare word.
pub fn error(kind: Kind, msg: impl Display) -> JsValue {
    let msg = msg.to_string();
    let text = if kind == Kind::Aborted && msg.is_empty() {
        kind.prefix().to_string()
    } else {
        format!("{}: {}", kind.prefix(), strip_prefix(kind, &msg))
    };
    js_sys::Error::new(&text).into()
}

/// Inner messages already carrying the prefix (e.g. `context_exceeded: …` from
/// bfi-models / bfi-runtime) are not doubled.
fn strip_prefix<'a>(kind: Kind, msg: &'a str) -> &'a str {
    msg.strip_prefix(kind.prefix())
        .and_then(|r| r.strip_prefix(':'))
        .map(str::trim_start)
        .unwrap_or(msg)
}

/// Classify an engine error, falling back to `default`.
pub fn classify(default: Kind, e: &anyhow::Error) -> JsValue {
    for cause in e.chain() {
        if let Some(ResidentError::UnsupportedType { .. }) = cause.downcast_ref::<ResidentError>() {
            return error(Kind::UnsupportedType, cause);
        }
        if let Some(GgufError::UnknownGgmlType { .. }) = cause.downcast_ref::<GgufError>() {
            return error(Kind::UnsupportedType, cause);
        }
    }
    let msg = format!("{e:#}");
    if msg.starts_with(bfi_models::CONTEXT_EXCEEDED) {
        return error(Kind::ContextExceeded, msg);
    }
    error(default, msg)
}

pub fn gguf(e: GgufError) -> JsValue {
    match e {
        GgufError::UnknownGgmlType { .. } | GgufError::UnknownGgmlTypeId(_) => {
            error(Kind::UnsupportedType, e)
        }
        other => error(Kind::LoadFailed, other),
    }
}
