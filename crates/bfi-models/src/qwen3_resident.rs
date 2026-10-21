//! qwen3 on GPU-resident tensors (feature `webgpu`): the browser path.
//!
//! Same math as [`crate::qwen3`] (dense, Q/K RMSNorm per head, NEOX RoPE, GQA,
//! SwiGLU, optional untied `output.weight`), but weights, activations and the KV
//! cache never leave the device. [`Qwen3Resident::forward`] only enqueues kernels
//! and hands back the last token's logits as a device tensor; the caller awaits
//! [`bfi_core::resident::Gpu::read_f32`] — the single host sync per step, which is
//! what lets it run where the host cannot block (wasm32).
//!
//! Loading is streaming: [`Qwen3ResidentLoader::accept`] takes each tensor as
//! [`bfi_gguf::TensorStream`] completes it, uploads it packed, and drops the host
//! bytes. Only 1-D norm vectors are widened to f32 (a few KB per layer).

mod loader;

use anyhow::{anyhow, ensure, Result};
use bfi_core::resident::{DeviceTensor, Gpu, KvBuffer, Weight};

use crate::qwen3::{Qwen3Config, ARCH_NAME};
use crate::session::CONTEXT_EXCEEDED;

pub use loader::Qwen3ResidentLoader;

/// Most tokens per enqueued forward chunk during prefill. Long contexts use
/// smaller chunks so the attention scores (`chunk · n_head · n_ctx` f32) fit one
/// storage binding; the score buffer never caps the context.
pub const PREFILL_CHUNK: usize = 64;

/// Prefill chunks enqueued between host yields. The browser engine awaits a
/// tiny readback after this many chunks so an `AbortSignal` (or `unload`) is
/// seen mid-prompt instead of after the whole prompt has been enqueued.
pub const PREFILL_YIELD_CHUNKS: usize = 4;


struct Layer {
    attn_norm: DeviceTensor,
    wq: Weight,
    wk: Weight,
    wv: Weight,
    wo: Weight,
    q_norm: DeviceTensor,
    k_norm: DeviceTensor,
    ffn_norm: DeviceTensor,
    gate: Weight,
    up: Weight,
    down: Weight,
}

/// A loaded qwen3 with a single-sequence KV cache on the device.
pub struct Qwen3Resident {
    gpu: Gpu,
    cfg: Qwen3Config,
    n_ctx: usize,
    kv_f16: bool,
    max_binding_bytes: u64,
    prefill_chunk: usize,
    params: u64,
    inv_freq: DeviceTensor,
    tok_embd: Weight,
    output: Weight,
    output_norm: DeviceTensor,
    layers: Vec<Layer>,
    k: Vec<KvBuffer>,
    v: Vec<KvBuffer>,
    n_past: usize,
}

impl Qwen3Resident {
    /// KV cache is f16 (else f32).
    pub fn kv_f16(&self) -> bool {
        self.kv_f16
    }
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }
    pub fn architecture(&self) -> &'static str {
        ARCH_NAME
    }
    /// Allocated KV capacity (≤ the trained context).
    pub fn n_ctx(&self) -> usize {
        self.n_ctx
    }
    pub fn n_vocab(&self) -> usize {
        self.cfg.n_vocab
    }
    pub fn n_layer(&self) -> usize {
        self.cfg.n_layer
    }
    pub fn n_embd(&self) -> usize {
        self.cfg.n_embd
    }
    /// Total weight elements in the GGUF.
    pub fn params(&self) -> u64 {
        self.params
    }
    /// Tokens currently in the KV cache.
    pub fn n_past(&self) -> usize {
        self.n_past
    }

    /// (Re)allocate the KV cache for `n_ctx` tokens (clamped to the trained
    /// context), dropping any previous cache and resetting the sequence. On
    /// WebGPU an out-of-memory allocation does not fail here: the caller wraps
    /// this in `Gpu::checked` and retries a smaller context on error.
    pub fn allocate_kv(&mut self, n_ctx: usize) {
        self.release_kv();
        let n_ctx = n_ctx.min(self.cfg.n_ctx).max(1);
        let kv_len = n_ctx * self.cfg.n_head_kv * self.cfg.head_dim;
        let f16 = self.kv_f16;
        self.k = (0..self.cfg.n_layer).map(|_| self.gpu.empty_kv(kv_len, f16)).collect();
        self.v = (0..self.cfg.n_layer).map(|_| self.gpu.empty_kv(kv_len, f16)).collect();
        self.n_ctx = n_ctx;
        let score_row = (self.cfg.n_head * n_ctx * 4).max(1) as u64;
        self.prefill_chunk = ((self.max_binding_bytes / score_row) as usize).clamp(1, PREFILL_CHUNK);
    }

    /// Drop the KV cache (context 0) so its device memory can be reclaimed.
    pub fn release_kv(&mut self) {
        self.k.clear();
        self.v.clear();
        self.n_ctx = 0;
        self.n_past = 0;
    }

    /// Prompt tokens to enqueue between host yields during prefill
    /// ([`PREFILL_YIELD_CHUNKS`] chunks).
    pub fn prefill_yield_tokens(&self) -> usize {
        self.prefill_chunk * PREFILL_YIELD_CHUNKS
    }

    /// Keep only the first `n` cached tokens (prompt-prefix reuse across turns).
    pub fn rewind(&mut self, n: usize) {
        self.n_past = self.n_past.min(n);
    }

    /// Append `tokens` to the sequence and return the last token's logits
    /// (`[n_vocab]`, still on the device). Errors with [`CONTEXT_EXCEEDED`] when
    /// the cache would overflow.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<DeviceTensor> {
        self.enqueue(tokens, true)?
            .ok_or_else(|| anyhow!("forward produced no logits"))
    }

    /// Append `tokens` to the sequence without computing logits (the leading
    /// part of a prompt; [`Self::forward`] the rest).
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        self.enqueue(tokens, false).map(|_| ())
    }

    fn enqueue(&mut self, tokens: &[u32], want_logits: bool) -> Result<Option<DeviceTensor>> {
        ensure!(!tokens.is_empty(), "forward: empty token list");
        ensure!(
            self.n_past + tokens.len() <= self.n_ctx,
            "{CONTEXT_EXCEEDED}: {} cached + {} new tokens > {} context",
            self.n_past,
            tokens.len(),
            self.n_ctx
        );
        let n_vocab = self.cfg.n_vocab as u32;
        if let Some(&bad) = tokens.iter().find(|&&t| t >= n_vocab) {
            return Err(anyhow!("token id {bad} >= n_vocab {n_vocab}"));
        }
        let chunks: Vec<&[u32]> = tokens.chunks(self.prefill_chunk).collect();
        let last = chunks.len() - 1;
        let mut logits = None;
        for (ci, chunk) in chunks.into_iter().enumerate() {
            logits = self.forward_chunk(chunk, want_logits && ci == last);
        }
        Ok(logits)
    }

    fn forward_chunk(&mut self, toks: &[u32], want_logits: bool) -> Option<DeviceTensor> {
        let g = &self.gpu;
        let c = &self.cfg;
        let (t, e, d, hq, hkv, eps) = (toks.len(), c.n_embd, c.head_dim, c.n_head, c.n_head_kv, c.rms_eps);
        let n_past = self.n_past;
        let ids = g.upload_u32(toks);
        let pos: Vec<u32> = (n_past..n_past + t).map(|p| p as u32).collect();
        let pos = g.upload_u32(&pos);
        let kv_row = hkv * d;

        let x = g.get_rows(&self.tok_embd, &ids);
        for (i, l) in self.layers.iter().enumerate() {
            let h = g.rms_norm(&x, &l.attn_norm, t, e, eps);
            let q = g.matmul(&l.wq, &h, t);
            let k = g.matmul(&l.wk, &h, t);
            let v = g.matmul(&l.wv, &h, t);
            let q = g.rms_norm(&q, &l.q_norm, t * hq, d, eps);
            let k = g.rms_norm(&k, &l.k_norm, t * hkv, d, eps);
            g.rope_neox(&q, &pos, &self.inv_freq, t, hq, d);
            g.rope_neox(&k, &pos, &self.inv_freq, t, hkv, d);
            g.write_kv(&k, &self.k[i], n_past * kv_row, t * kv_row);
            g.write_kv(&v, &self.v[i], n_past * kv_row, t * kv_row);
            let a = g.attention(&q, &self.k[i], &self.v[i], t, n_past, hq, hkv, d);
            let o = g.matmul(&l.wo, &a, t);
            g.add_inplace(&x, &o);

            let h = g.rms_norm(&x, &l.ffn_norm, t, e, eps);
            let gate = g.matmul(&l.gate, &h, t);
            let up = g.matmul(&l.up, &h, t);
            let s = g.swiglu(&gate, &up);
            let down = g.matmul(&l.down, &s, t);
            g.add_inplace(&x, &down);
        }
        self.n_past += t;
        if !want_logits {
            return None;
        }
        let last = g.empty_f32(e);
        g.copy(&x, (t - 1) * e, &last, 0, e);
        let h = g.rms_norm(&last, &self.output_norm, 1, e, eps);
        Some(g.matmul(&self.output, &h, 1))
    }
}

#[cfg(test)]
mod tests;
