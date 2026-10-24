//! Errors with stable message prefixes (spec 34 §3.1). Hosts branch on
//! `err.message.startsWith(prefix)`, so these strings are contract — every glue crate
//! turns an [`EngineError`] into its realm's `Error` with exactly [`EngineError`]'s
//! `Display` as the message.

use std::fmt::{self, Display};

#[cfg(feature = "webgpu")]
use bfi_core::resident::ResidentError;
use bfi_gguf::GgufError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// No WebGPU adapter / device, or the device was lost (create a new engine).
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
    /// CPU backend: packed weights + KV cache would not fit the host's memory ceiling.
    CpuMemoryExceeded,
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
            Kind::CpuMemoryExceeded => "cpu_memory_exceeded",
            Kind::Engine => "engine_error",
        }
    }
}

/// An engine failure. `Display` is the contract message: `<prefix>: <msg>`, or the bare
/// word `aborted`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineError {
    pub kind: Kind,
    text: String,
}

impl Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

impl std::error::Error for EngineError {}

/// `<prefix>: <msg>`; `aborted` with no message is the bare word.
pub fn error(kind: Kind, msg: impl Display) -> EngineError {
    let msg = msg.to_string();
    let text = if kind == Kind::Aborted && msg.is_empty() {
        kind.prefix().to_string()
    } else {
        format!("{}: {}", kind.prefix(), strip_prefix(kind, &msg))
    };
    EngineError { kind, text }
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
pub fn classify(default: Kind, e: &anyhow::Error) -> EngineError {
    for cause in e.chain() {
        #[cfg(feature = "webgpu")]
        if let Some(r) = cause.downcast_ref::<ResidentError>() {
            if matches!(r, ResidentError::UnsupportedType { .. } | ResidentError::DeviceLost(_)) {
                return gpu_error(default, r);
            }
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

#[cfg(feature = "webgpu")]
/// A resident-GPU error: a lost device is `webgpu_unavailable:`, an
/// unsupported tensor type `unsupported_type:`, anything else `default`.
pub fn gpu_error(default: Kind, e: &ResidentError) -> EngineError {
    let kind = match e {
        ResidentError::DeviceLost(_) => Kind::WebgpuUnavailable,
        ResidentError::UnsupportedType { .. } => Kind::UnsupportedType,
        _ => default,
    };
    error(kind, e)
}

pub fn gguf(e: GgufError) -> EngineError {
    match e {
        GgufError::UnknownGgmlType { .. } | GgufError::UnknownGgmlTypeId(_) => {
            error(Kind::UnsupportedType, e)
        }
        other => error(Kind::LoadFailed, other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_carry_the_contract_prefix_once() {
        assert_eq!(error(Kind::Busy, "x").to_string(), "busy: x");
        assert_eq!(error(Kind::Aborted, "").to_string(), "aborted");
        assert_eq!(
            error(Kind::ContextExceeded, "context_exceeded: 10 > 8").to_string(),
            "context_exceeded: 10 > 8"
        );
    }
}
