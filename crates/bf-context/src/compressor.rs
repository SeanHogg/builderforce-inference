//! The service's LLMLingua-2 compressor: loaded once, on first use, shared by every client.
//!
//! Unlike the embedder it is not warmed at start — the model is ~179 MB and only a client
//! that asks to compress should pay for it. A failed load (offline first run) is remembered
//! and reported through `state()` / `error()`, the same contract as `LazyEmbedder`.

use anyhow::{anyhow, Result};
use serde::Serialize;
use std::path::PathBuf;
use std::sync::OnceLock;

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum CompressorState {
    /// Built in; the model loads on the first request.
    Idle,
    Ready,
    Unavailable,
    /// Built without the `compress` feature.
    Disabled,
}

/// The result of one compression, in the shape every transport returns.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CompressOutput {
    pub text: String,
    pub original_tokens: usize,
    pub kept_tokens: usize,
}

#[cfg(feature = "compress")]
type Loaded = bf_compress::Compressor;
#[cfg(not(feature = "compress"))]
type Loaded = ();

pub struct LazyCompressor {
    #[cfg_attr(not(feature = "compress"), allow(dead_code))]
    cache_dir: PathBuf,
    cell: OnceLock<Result<Loaded, String>>,
}

impl LazyCompressor {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self { cache_dir, cell: OnceLock::new() }
    }

    pub fn state(&self) -> CompressorState {
        if !cfg!(feature = "compress") {
            return CompressorState::Disabled;
        }
        match self.cell.get() {
            None => CompressorState::Idle,
            Some(Ok(_)) => CompressorState::Ready,
            Some(Err(_)) => CompressorState::Unavailable,
        }
    }

    pub fn error(&self) -> Option<String> {
        self.cell.get().and_then(|r| r.as_ref().err().cloned())
    }

    #[cfg(feature = "compress")]
    pub fn compress(&self, text: &str, rate: f32, force_keep: &[&str]) -> Result<CompressOutput> {
        let loaded = self.cell.get_or_init(|| {
            std::fs::create_dir_all(&self.cache_dir).map_err(|e| e.to_string())?;
            bf_compress::Compressor::load(&self.cache_dir).map_err(|e| format!("{e:#}"))
        });
        let c = loaded.as_ref().map_err(|e| anyhow!("compression model unavailable: {e}"))?;
        let out = c.compress(text, rate, force_keep)?;
        Ok(CompressOutput { text: out.text, original_tokens: out.original_tokens, kept_tokens: out.kept_tokens })
    }

    #[cfg(not(feature = "compress"))]
    pub fn compress(&self, _text: &str, _rate: f32, _force_keep: &[&str]) -> Result<CompressOutput> {
        Err(anyhow!("this bf-context was built without the compress feature"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "compress"))]
    #[test]
    fn disabled_build_reports_and_refuses() {
        let c = LazyCompressor::new(PathBuf::from("unused"));
        assert_eq!(c.state(), CompressorState::Disabled);
        assert!(c.compress("x", 0.5, &[]).is_err());
        assert_eq!(c.error(), None);
    }

    #[cfg(feature = "compress")]
    #[test]
    fn built_in_starts_idle() {
        let c = LazyCompressor::new(PathBuf::from("unused"));
        assert_eq!(c.state(), CompressorState::Idle);
    }
}
