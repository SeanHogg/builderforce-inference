//! qwen3: dense Transformer, all full attention, Q/K-norm per head, GQA, SwiGLU.
//!
//! M2 correctness baseline (Qwen3-0.6B / 1.7B). Tensor names follow llama.cpp
//! GGUF (`blk.{i}.attn_q_norm.weight`, …).

use std::sync::Arc;

use anyhow::{anyhow, ensure, Context};
use bfi_core::{Backend, DType, RopeParams, Tensor};
use bfi_gguf::{GgufFile, ModelHeader};

use crate::arch::{ArchReg, Architecture, Model};
use crate::state::{Logits, SeqState, StateLayout};
use crate::tutil;

pub const ARCH_NAME: &str = "qwen3";

pub struct Qwen3Arch;

pub static QWEN3: Qwen3Arch = Qwen3Arch;

inventory::submit! {
    ArchReg(&QWEN3)
}

pub(crate) struct Qwen3Config {
    pub(crate) n_layer: usize,
    pub(crate) n_embd: usize,
    pub(crate) n_ff: usize,
    pub(crate) n_vocab: usize,
    pub(crate) n_ctx: usize,
    pub(crate) n_head: usize,
    pub(crate) n_head_kv: usize,
    pub(crate) head_dim: usize,
    pub(crate) rms_eps: f32,
    pub(crate) rope_freq_base: f32,
    pub(crate) n_rot: usize,
}

struct Qwen3Layer {
    attn_norm: Tensor,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    ffn_norm: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

pub(crate) struct Qwen3Model {
    backend: Arc<dyn Backend>,
    cfg: Qwen3Config,
    rope: RopeParams,
    tok_embd: Tensor,
    output_norm: Tensor,
    /// `None` = tied to `tok_embd` (Qwen3-0.6B, Bonsai 1.7B / 4B).
    output: Option<Tensor>,
    layers: Vec<Qwen3Layer>,
}

impl Architecture for Qwen3Arch {
    fn name(&self) -> &'static str {
        ARCH_NAME
    }

    fn from_gguf(
        &self,
        file: &GgufFile,
        backend: Arc<dyn Backend>,
    ) -> anyhow::Result<Box<dyn Model>> {
        let cfg = dense_config(file.header())?;
        let model = assemble(cfg, backend, |b, name| match file.header().tensor(name) {
            Ok(_) => load_weight(file, b, name).map(Some),
            Err(_) => Ok(None),
        })?;
        Ok(Box::new(model))
    }
}

/// The qwen3 config, refusing the MoE variant (dense qwen3 only). Every loader — the
/// file one above, the streaming CPU one (`qwen3_stream`) and the GPU-resident one —
/// starts here.
pub(crate) fn dense_config(header: &ModelHeader) -> anyhow::Result<Qwen3Config> {
    let cfg = parse_config(header)?;
    if let Some(n) = header.arch_u64("expert_count") {
        ensure!(
            n == 0,
            "qwen3moe (expert_count={n}) is not implemented; dense qwen3 only"
        );
    }
    Ok(cfg)
}

/// Build the model from named weights. `take(backend, name)` yields `Ok(None)` for a
/// tensor the GGUF does not carry; only `output.weight` may be absent (the lm_head is
/// then tied to `token_embd.weight`, shared rather than copied).
pub(crate) fn assemble(
    mut cfg: Qwen3Config,
    backend: Arc<dyn Backend>,
    mut take: impl FnMut(&dyn Backend, &str) -> anyhow::Result<Option<Tensor>>,
) -> anyhow::Result<Qwen3Model> {
    let mut require = |name: &str| -> anyhow::Result<Tensor> {
        take(backend.as_ref(), name)?.ok_or_else(|| anyhow!("missing tensor {name}"))
    };
    let tok_embd = require("token_embd.weight")?;
    ensure!(
        tok_embd.shape.len() >= 2,
        "token_embd.weight expected [n_vocab, n_embd], got {:?}",
        tok_embd.shape
    );
    cfg.n_vocab = tok_embd.shape[0];
    ensure!(
        tok_embd.shape[1] == cfg.n_embd,
        "token_embd n_embd {} != embedding_length {}",
        tok_embd.shape[1],
        cfg.n_embd
    );

    let mut layers = Vec::with_capacity(cfg.n_layer);
    for i in 0..cfg.n_layer {
        let p = format!("blk.{i}");
        let layer = Qwen3Layer {
            attn_norm: require(&format!("{p}.attn_norm.weight"))?,
            wq: require(&format!("{p}.attn_q.weight"))?,
            wk: require(&format!("{p}.attn_k.weight"))?,
            wv: require(&format!("{p}.attn_v.weight"))?,
            wo: require(&format!("{p}.attn_output.weight"))?,
            q_norm: require(&format!("{p}.attn_q_norm.weight"))?,
            k_norm: require(&format!("{p}.attn_k_norm.weight"))?,
            ffn_norm: require(&format!("{p}.ffn_norm.weight"))?,
            gate: require(&format!("{p}.ffn_gate.weight"))?,
            up: require(&format!("{p}.ffn_up.weight"))?,
            down: require(&format!("{p}.ffn_down.weight"))?,
        };
        layers.push(layer.checked(&cfg, i)?);
    }

    let output_norm = require("output_norm.weight")?;
    drop(require);
    let output = take(backend.as_ref(), "output.weight")?;
    if let Some(out) = &output {
        check_in_out(out, cfg.n_embd, cfg.n_vocab, 0, "output")
            .map_err(|e| anyhow!("output.weight: {e}"))?;
    }

    let rope = RopeParams {
        n_dims: cfg.head_dim,
        n_ctx: cfg.n_ctx,
        freq_base: cfg.rope_freq_base,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        n_rot: cfg.n_rot,
        sections: None,
        // llama.cpp `llama_model_rope_type`: QWEN3 → LLAMA_ROPE_TYPE_NEOX.
        neox: true,
    };

    Ok(Qwen3Model {
        backend,
        cfg,
        rope,
        tok_embd,
        output_norm,
        output,
        layers,
    })
}

impl Qwen3Model {
    /// Serve at most `n_ctx` tokens (the KV cache a [`SeqState`] allocates), never more
    /// than the trained context. RoPE keeps the trained context.
    pub(crate) fn cap_context(&mut self, n_ctx: usize) {
        self.cfg.n_ctx = n_ctx.min(self.cfg.n_ctx).max(1);
    }

    /// The lm_head: `output.weight`, or the token embedding when tied.
    fn lm_head(&self) -> &Tensor {
        self.output.as_ref().unwrap_or(&self.tok_embd)
    }
}

impl Model for Qwen3Model {
    fn architecture(&self) -> &'static str {
        ARCH_NAME
    }
    fn n_layer(&self) -> usize {
        self.cfg.n_layer
    }
    fn n_embd(&self) -> usize {
        self.cfg.n_embd
    }
    fn n_vocab(&self) -> usize {
        self.cfg.n_vocab
    }
    fn n_ctx(&self) -> usize {
        self.cfg.n_ctx
    }
    fn n_head(&self) -> usize {
        self.cfg.n_head
    }
    fn n_head_kv(&self) -> usize {
        self.cfg.n_head_kv
    }
    fn head_dim(&self) -> usize {
        self.cfg.head_dim
    }

    fn forward(
        &self,
        tokens: &[u32],
        pos: &[i32],
        state: &mut SeqState,
    ) -> anyhow::Result<Logits> {
        ensure!(!tokens.is_empty(), "forward: empty token list");
        ensure!(
            pos.len() == tokens.len(),
            "forward: pos len {} != tokens {}",
            pos.len(),
            tokens.len()
        );
        let t = tokens.len();
        ensure!(
            state.n + t <= self.cfg.n_ctx,
            "forward: KV overflow n={} + t={t} > n_ctx={}",
            state.n,
            self.cfg.n_ctx
        );

        let b = self.backend.as_ref();
        let ids: Vec<i32> = tokens.iter().map(|&tok| tok as i32).collect();
        let mut x = b.get_rows(&self.tok_embd, &ids)?;

        let n_head = self.cfg.n_head;
        let n_kv = self.cfg.n_head_kv;
        let d = self.cfg.head_dim;
        let q_dim = n_head * d;
        let eps = self.cfg.rms_eps;

        for (i, layer) in self.layers.iter().enumerate() {
            let residual = x.clone();
            x = b.rms_norm(&x, &layer.attn_norm, eps)?;

            let q = b.matmul_q(&layer.wq, &x)?;
            let k = b.matmul_q(&layer.wk, &x)?;
            let v = b.matmul_q(&layer.wv, &x)?;
            let mut q = tutil::reshape(q, vec![t, n_head, d])?;
            let mut k = tutil::reshape(k, vec![t, n_kv, d])?;
            let v = tutil::reshape(v, vec![t, n_kv, d])?;

            // Q/K RMSNorm per head (last dim = head_dim), then classic RoPE.
            q = b.rms_norm(&q, &layer.q_norm, eps)?;
            k = b.rms_norm(&k, &layer.k_norm, eps)?;
            b.rope(&mut q, pos, &self.rope)?;
            b.rope(&mut k, pos, &self.rope)?;

            tutil::write_rows(&mut state.k[i], state.n, &k)?;
            tutil::write_rows(&mut state.v[i], state.n, &v)?;
            let filled = state.n + t;
            let k_all = tutil::prefix_rows(&state.k[i], filled)?;
            let v_all = tutil::prefix_rows(&state.v[i], filled)?;

            let attn = b.attention(&q, &k_all, &v_all, true)?;
            let attn = tutil::reshape(attn, vec![t, q_dim])?;
            let attn_out = b.matmul_q(&layer.wo, &attn)?;
            x = b.add(&residual, &attn_out)?;

            let residual = x.clone();
            x = b.rms_norm(&x, &layer.ffn_norm, eps)?;
            let gate = b.matmul_q(&layer.gate, &x)?;
            let up = b.matmul_q(&layer.up, &x)?;
            let hidden = b.swiglu(&gate, &up)?;
            let down = b.matmul_q(&layer.down, &hidden)?;
            x = b.add(&residual, &down)?;
        }

        state.n += t;
        x = last_row(&x)?;
        x = b.rms_norm(&x, &self.output_norm, eps)?;
        let logits = b.matmul_q(self.lm_head(), &x)?;
        let values = tutil::as_f32(&logits)?.to_vec();
        ensure!(
            values.len() == self.cfg.n_vocab,
            "logits len {} != n_vocab {}",
            values.len(),
            self.cfg.n_vocab
        );
        Ok(Logits { values })
    }

    fn state_layout(&self) -> StateLayout {
        StateLayout {
            n_layer: self.cfg.n_layer,
            n_ctx: self.cfg.n_ctx,
            n_head_kv: self.cfg.n_head_kv,
            head_dim: self.cfg.head_dim,
            n_recurrent: 0,
            recurrent_shape: [0, 0, 0],
        }
    }
}

pub(crate) fn parse_config(h: &ModelHeader) -> anyhow::Result<Qwen3Config> {
    let n_layer = arch_usize(h, "block_count")?;
    let n_embd = arch_usize(h, "embedding_length")?;
    let n_ff = arch_usize(h, "feed_forward_length")?;
    let n_ctx = arch_usize(h, "context_length")?;
    let n_head = arch_usize(h, "attention.head_count")?;
    let n_head_kv = h
        .arch_u64("attention.head_count_kv")
        .map(|v| v as usize)
        .unwrap_or(n_head);
    ensure!(n_head > 0 && n_head_kv > 0, "head counts must be > 0");
    ensure!(
        n_head % n_head_kv == 0,
        "GQA: n_head {n_head} % n_head_kv {n_head_kv} != 0"
    );
    // Qwen3 head_dim is `attention.key_length` (often 128), NOT n_embd/n_head.
    let head_dim = h
        .arch_u64("attention.key_length")
        .map(|v| v as usize)
        .unwrap_or_else(|| n_embd / n_head);
    ensure!(head_dim > 0, "head_dim must be > 0");
    let rms_eps = h
        .arch_f64("attention.layer_norm_rms_epsilon")
        .unwrap_or(1e-6) as f32;
    let rope_freq_base = h.arch_f64("rope.freq_base").unwrap_or(1_000_000.0) as f32;
    let n_rot = h
        .arch_u64("rope.dimension_count")
        .map(|v| v as usize)
        .unwrap_or(head_dim);
    Ok(Qwen3Config {
        n_layer,
        n_embd,
        n_ff,
        n_vocab: 0,
        n_ctx,
        n_head,
        n_head_kv,
        head_dim,
        rms_eps,
        rope_freq_base,
        n_rot,
    })
}

fn arch_usize(h: &ModelHeader, field: &str) -> anyhow::Result<usize> {
    h.arch_u64(field)
        .map(|v| v as usize)
        .ok_or_else(|| anyhow!("missing GGUF metadata {}.{}", ARCH_NAME, field))
}

impl Qwen3Layer {
    fn checked(self, cfg: &Qwen3Config, i: usize) -> anyhow::Result<Self> {
        let q_out = cfg.n_head * cfg.head_dim;
        let kv_out = cfg.n_head_kv * cfg.head_dim;
        check_in_out(&self.wq, cfg.n_embd, q_out, i, "attn_q")?;
        check_in_out(&self.wk, cfg.n_embd, kv_out, i, "attn_k")?;
        check_in_out(&self.wv, cfg.n_embd, kv_out, i, "attn_v")?;
        check_in_out(&self.wo, q_out, cfg.n_embd, i, "attn_output")?;
        check_in_out(&self.gate, cfg.n_embd, cfg.n_ff, i, "ffn_gate")?;
        check_in_out(&self.up, cfg.n_embd, cfg.n_ff, i, "ffn_up")?;
        check_in_out(&self.down, cfg.n_ff, cfg.n_embd, i, "ffn_down")?;
        ensure!(
            last_dim(&self.q_norm) == cfg.head_dim,
            "blk.{i}.attn_q_norm last dim {} != head_dim {}",
            last_dim(&self.q_norm),
            cfg.head_dim
        );
        ensure!(
            last_dim(&self.k_norm) == cfg.head_dim,
            "blk.{i}.attn_k_norm last dim {} != head_dim {}",
            last_dim(&self.k_norm),
            cfg.head_dim
        );
        Ok(self)
    }
}

fn check_in_out(
    w: &Tensor,
    n_in: usize,
    n_out: usize,
    layer: usize,
    name: &str,
) -> anyhow::Result<()> {
    ensure!(
        w.shape.len() >= 2,
        "blk.{layer}.{name} expected rank >= 2, got {:?}",
        w.shape
    );
    let got_in = *w.shape.last().unwrap();
    let got_out = w.n_elems() / got_in;
    ensure!(
        got_in == n_in && got_out == n_out,
        "blk.{layer}.{name} shape {:?} (in={got_in} out={got_out}) != expected in={n_in} out={n_out}",
        w.shape
    );
    Ok(())
}

fn last_dim(t: &Tensor) -> usize {
    t.shape.last().copied().unwrap_or(0)
}

fn load_weight(file: &GgufFile, backend: &dyn Backend, name: &str) -> anyhow::Result<Tensor> {
    let info = file
        .tensor_info(name)
        .with_context(|| format!("missing tensor {name}"))?;
    weight_from_bytes(backend, name, info.ggml_type, &info.dims, file.tensor_bytes(name)?)
}

/// One GGUF tensor onto `backend`: F32 as is, a 1-D vector (a norm) widened to F32 (the
/// kernels read norms as F32), any other matrix uploaded PACKED (`Q1_0` stays 18 bytes
/// per 128 weights). Shared by the file loader and the streaming one.
pub(crate) fn weight_from_bytes(
    backend: &dyn Backend,
    name: &str,
    ggml_type: u32,
    dims: &[u64],
    bytes: &[u8],
) -> anyhow::Result<Tensor> {
    let shape = tutil::logical_shape(dims);
    if tutil::is_f32(ggml_type) {
        let data = tutil::f32_from_le_bytes(bytes)?;
        return backend.upload_f32(&shape, &data);
    }
    if dims.len() <= 1 {
        let ty = bfi_gguf::require_type(ggml_type, name)?;
        let n: usize = shape.iter().product();
        let mut data = vec![0.0f32; n];
        bfi_gguf::dequant_tensor(ty, bytes, &mut data).with_context(|| format!("dequant {name}"))?;
        return backend.upload_f32(&shape, &data);
    }
    let dtype = DType::from_ggml(ggml_type)
        .ok_or_else(|| anyhow!("unsupported ggml type {ggml_type} for tensor {name}"))?;
    backend.upload_quant(&shape, dtype, bytes)
}

fn last_row(x: &Tensor) -> anyhow::Result<Tensor> {
    let s = tutil::shape_of(x);
    ensure!(s.len() == 2, "expected [t, d], got {s:?}");
    let t = s[0];
    let d = s[1];
    ensure!(t > 0, "empty sequence");
    let data = tutil::as_f32(x)?;
    Ok(tutil::f32_tensor(
        vec![1, d],
        data[(t - 1) * d..t * d].to_vec(),
    ))
}
