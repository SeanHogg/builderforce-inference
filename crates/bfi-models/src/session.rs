//! One sequence on a host-tensor [`Model`]: the model plus its [`SeqState`], with the
//! two operations a chat loop needs between turns — append tokens and get the last
//! token's logits, and rewind to a cached prefix.
//!
//! This is what the browser engine's CPU backend drives (`bfi-web`, no WebGPU): the
//! SAME `Model` / `Backend` path the native `bfi` runs, sized to a KV budget instead of
//! the trained context. The GPU-resident twin is `Qwen3Resident`.

use anyhow::{ensure, Result};

use crate::arch::Model;
use crate::state::SeqState;

/// Error text prefix when a request would overrun the allocated KV cache.
/// `bfi-web` maps it to the `context_exceeded:` JS error (both backends).
pub const CONTEXT_EXCEEDED: &str = "context_exceeded";

pub struct Session {
    model: Box<dyn Model>,
    state: SeqState,
    params: u64,
}

impl Session {
    /// Allocate the sequence state the model's layout asks for (its `n_ctx` KV rows).
    pub fn new(model: Box<dyn Model>, params: u64) -> Self {
        let state = SeqState::allocate(&model.state_layout());
        Self { model, state, params }
    }

    pub fn architecture(&self) -> &'static str {
        self.model.architecture()
    }
    /// Allocated KV capacity.
    pub fn n_ctx(&self) -> usize {
        self.model.n_ctx()
    }
    pub fn n_vocab(&self) -> usize {
        self.model.n_vocab()
    }
    /// Total weight elements in the GGUF.
    pub fn params(&self) -> u64 {
        self.params
    }
    /// Tokens currently in the cache.
    pub fn n_past(&self) -> usize {
        self.state.n
    }

    /// Keep only the first `n` cached tokens. A KV cache rewinds by moving its fill
    /// mark (rows past it are overwritten); a recurrent state cannot, so a model with
    /// recurrent layers restarts from empty when asked to rewind at all.
    pub fn rewind(&mut self, n: usize) {
        if n >= self.state.n {
            return;
        }
        if self.model.state_layout().n_recurrent > 0 {
            self.state = SeqState::allocate(&self.model.state_layout());
        } else {
            self.state.n = n;
        }
    }

    /// Append `tokens` and return the last token's logits (`n_vocab` floats). Errors
    /// with [`CONTEXT_EXCEEDED`] when the cache would overflow.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        ensure!(!tokens.is_empty(), "forward: empty token list");
        ensure!(
            self.state.n + tokens.len() <= self.n_ctx(),
            "{CONTEXT_EXCEEDED}: {} cached + {} new tokens > {} context",
            self.state.n,
            tokens.len(),
            self.n_ctx()
        );
        let n_vocab = self.n_vocab() as u32;
        if let Some(&bad) = tokens.iter().find(|&&t| t >= n_vocab) {
            anyhow::bail!("token id {bad} >= n_vocab {n_vocab}");
        }
        let start = self.state.n as i32;
        let pos: Vec<i32> = (start..start + tokens.len() as i32).collect();
        Ok(self.model.forward(tokens, &pos, &mut self.state)?.values)
    }
}
