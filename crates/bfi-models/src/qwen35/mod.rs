//! qwen35 (Bonsai): hybrid gated-delta (~75%) + full attention (~25%).
//!
//! Spec: [`SPEC.md`](./SPEC.md). qwen3 stays the dense all-full-attn baseline.

use std::sync::Arc;

use anyhow::{anyhow, ensure, Context};
use bfi_core::{Backend, DType, GatedDeltaParams, RopeParams, Tensor};
use bfi_gguf::{GgufFile, MetaValue, ModelHeader};

use crate::arch::{ArchReg, Architecture, Model};
use crate::state::{Logits, SeqState, StateLayout};
use crate::tutil;

pub const ARCH_NAME: &str = "qwen35";

pub struct Qwen35Arch;

pub static QWEN35: Qwen35Arch = Qwen35Arch;

inventory::submit! {
    ArchReg(&QWEN35)
}

struct Qwen35Config {
    n_layer: usize,
    n_embd: usize,
    n_ff: usize,
    n_vocab: usize,
    n_ctx: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    rms_eps: f32,
    rope_freq_base: f32,
    n_rot: usize,
    /// `true` = gated-delta (linear) layer.
    layer_recurrent: Vec<bool>,
}

enum AttnKind {
    Full {
        q_norm: Tensor,
        k_norm: Tensor,
    },
    Delta {
        conv: Option<Tensor>,
        a_gate: Option<Tensor>,
        beta: Option<Tensor>,
    },
}

struct Qwen35Layer {
    attn: AttnKind,
    attn_norm: Tensor,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    ffn_norm: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

struct Qwen35Model {
    backend: Arc<dyn Backend>,
    cfg: Qwen35Config,
    rope: RopeParams,
    tok_embd: Tensor,
    output_norm: Tensor,
    output: Tensor,
    layers: Vec<Qwen35Layer>,
    mtp: Vec<(String, Tensor)>,
}

/// Default 3+1 hybrid: full attention iff `(il + 1) % 4 == 0` (~25%).
pub(crate) fn default_is_recurrent(il: usize) -> bool {
    (il + 1) % 4 != 0
}

fn is_linear_kind(s: &str) -> bool {
    matches!(
        s.to_ascii_lowercase().as_str(),
        "linear_attention" | "recurrent" | "gated_delta" | "delta_net" | "ssm" | "linear"
    )
}

fn is_full_kind(s: &str) -> bool {
    matches!(
        s.to_ascii_lowercase().as_str(),
        "full_attention" | "attention" | "full" | "mha" | "softmax"
    )
}

fn parse_layer_kinds(h: &ModelHeader, n_layer: usize) -> Vec<bool> {
    let key = format!("{ARCH_NAME}.attention.layer_types");
    if let Some(v) = h.get(&key) {
        if let Some(ss) = v.as_string_array() {
            return (0..n_layer)
                .map(|i| match ss.get(i) {
                    Some(s) if is_full_kind(s) => false,
                    Some(s) if is_linear_kind(s) => true,
                    _ => default_is_recurrent(i),
                })
                .collect();
        }
        if let MetaValue::Array(items) = v {
            return (0..n_layer)
                .map(|i| match items.get(i).and_then(|x| x.as_u64()) {
                    Some(0) => false,
                    Some(_) => true,
                    None => default_is_recurrent(i),
                })
                .collect();
        }
    }
    (0..n_layer).map(default_is_recurrent).collect()
}

fn rope_sections(h: &ModelHeader, n_rot: usize) -> [usize; 4] {
    let key = format!("{ARCH_NAME}.rope.dimension_sections");
    if let Some(MetaValue::Array(items)) = h.get(&key) {
        let mut s = [0usize; 4];
        for (i, it) in items.iter().take(4).enumerate() {
            s[i] = it.as_u64().unwrap_or(0) as usize;
        }
        if s.iter().copied().sum::<usize>() > 0 {
            return s;
        }
    }
    [n_rot, 0, 0, 0]
}

impl Architecture for Qwen35Arch {
    fn name(&self) -> &'static str {
        ARCH_NAME
    }

    fn from_gguf(
        &self,
        file: &GgufFile,
        backend: Arc<dyn Backend>,
    ) -> anyhow::Result<Box<dyn Model>> {
        let header = file.header();
        let mut cfg = parse_config(header)?;

        let tok_embd = load_weight(file, backend.as_ref(), "token_embd.weight")?;
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
            layers.push(load_layer(file, backend.as_ref(), i, &cfg)?);
        }

        let output_norm = load_weight(file, backend.as_ref(), "output_norm.weight")?;
        let output = match file.header().tensor("output.weight") {
            Ok(_) => load_weight(file, backend.as_ref(), "output.weight")?,
            Err(_) => tok_embd.clone(),
        };

        let mtp = load_mtp(file, backend.as_ref(), cfg.n_layer)?;

        let sections = rope_sections(header, cfg.n_rot);
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
            sections: Some(sections),
        };

        Ok(Box::new(Qwen35Model {
            backend,
            cfg,
            rope,
            tok_embd,
            output_norm,
            output,
            layers,
            mtp,
        }))
    }
}

impl Model for Qwen35Model {
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
    fn mtp_loaded(&self) -> usize {
        self.mtp.len()
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
        let kv_dim = n_kv * d;
        let eps = self.cfg.rms_eps;

        for (i, layer) in self.layers.iter().enumerate() {
            let residual = x.clone();
            x = b.rms_norm(&x, &layer.attn_norm, eps)?;

            let q = b.matmul_q(&layer.wq, &x)?;
            let k = b.matmul_q(&layer.wk, &x)?;
            let v = b.matmul_q(&layer.wv, &x)?;

            let attn_out = match &layer.attn {
                AttnKind::Delta {
                    conv,
                    a_gate,
                    beta,
                } => {
                    let mut mixed = tutil::concat_last(&q, &k)?;
                    mixed = tutil::concat_last(&mixed, &v)?;
                    if let Some(ker) = conv {
                        mixed = b.conv1d_causal(&mixed, ker)?;
                    }
                    let mut parts = tutil::split_last(&mixed, &[q_dim, kv_dim, kv_dim])?;
                    ensure!(parts.len() == 3, "mixed QKV split");
                    let mut q = tutil::reshape(parts.remove(0), vec![t, n_head, d])?;
                    let mut k = tutil::reshape(parts.remove(0), vec![t, n_kv, d])?;
                    let v = tutil::reshape(parts.remove(0), vec![t, n_kv, d])?;
                    q = b.l2_norm(&q, eps)?;
                    k = b.l2_norm(&k, eps)?;
                    let g = match a_gate {
                        Some(w) => b.softplus(&b.matmul_q(w, &x)?)?,
                        None => tutil::zeros_f32(vec![t, n_kv]),
                    };
                    let beta_t = match beta {
                        Some(w) => b.sigmoid(&b.matmul_q(w, &x)?)?,
                        None => tutil::ones_f32(vec![t, n_kv]),
                    };
                    let rec = state.recurrent[i].get_or_insert_with(|| {
                        tutil::zeros_f32(vec![n_kv, d, d])
                    });
                    let attn = b.gated_delta_step(GatedDeltaParams {
                        q: &q,
                        k: &k,
                        v: &v,
                        g: &g,
                        beta: &beta_t,
                        state: rec,
                    })?;
                    let attn = tutil::reshape(attn, vec![t, q_dim])?;
                    b.matmul_q(&layer.wo, &attn)?
                }
                AttnKind::Full { q_norm, k_norm } => {
                    let mut q = tutil::reshape(q, vec![t, n_head, d])?;
                    let mut k = tutil::reshape(k, vec![t, n_kv, d])?;
                    let v = tutil::reshape(v, vec![t, n_kv, d])?;
                    q = b.rms_norm(&q, q_norm, eps)?;
                    k = b.rms_norm(&k, k_norm, eps)?;
                    b.rope_multi(&mut q, pos, &self.rope)?;
                    b.rope_multi(&mut k, pos, &self.rope)?;
                    tutil::write_rows(&mut state.k[i], state.n, &k)?;
                    tutil::write_rows(&mut state.v[i], state.n, &v)?;
                    let filled = state.n + t;
                    let k_all = tutil::prefix_rows(&state.k[i], filled)?;
                    let v_all = tutil::prefix_rows(&state.v[i], filled)?;
                    let attn = b.attention(&q, &k_all, &v_all, true)?;
                    let attn = tutil::reshape(attn, vec![t, q_dim])?;
                    b.matmul_q(&layer.wo, &attn)?
                }
            };
            x = b.add(&residual, &attn_out)?;

            let residual = x.clone();
            x = b.rms_norm(&x, &layer.ffn_norm, eps)?;
            let gate = b.matmul_q(&layer.gate, &x)?;
            let up = b.matmul_q(&layer.up, &x)?;
            let hidden = b.swiglu(&gate, &up)?;
            let down = b.matmul_q(&layer.down, &hidden)?;
            x = b.add(&residual, &down)?;
        }

        // MTP tensors live on `self.mtp` and are never forwarded.

        state.n += t;
        x = last_row(&x)?;
        x = b.rms_norm(&x, &self.output_norm, eps)?;
        let logits = b.matmul_q(&self.output, &x)?;
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
        let n_recurrent = self
            .cfg
            .layer_recurrent
            .iter()
            .filter(|r| **r)
            .count();
        StateLayout {
            n_layer: self.cfg.n_layer,
            n_ctx: self.cfg.n_ctx,
            n_head_kv: self.cfg.n_head_kv,
            head_dim: self.cfg.head_dim,
            n_recurrent,
            recurrent_shape: if n_recurrent > 0 {
                [self.cfg.n_head_kv, self.cfg.head_dim, self.cfg.head_dim]
            } else {
                [0, 0, 0]
            },
        }
    }
}

fn parse_config(h: &ModelHeader) -> anyhow::Result<Qwen35Config> {
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
    let layer_recurrent = parse_layer_kinds(h, n_layer);
    ensure!(
        layer_recurrent.len() == n_layer,
        "layer_types len {} != block_count {n_layer}",
        layer_recurrent.len()
    );
    Ok(Qwen35Config {
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
        layer_recurrent,
    })
}

fn arch_usize(h: &ModelHeader, field: &str) -> anyhow::Result<usize> {
    h.arch_u64(field)
        .map(|v| v as usize)
        .ok_or_else(|| anyhow!("missing GGUF metadata {}.{field}", ARCH_NAME))
}

fn load_layer(
    file: &GgufFile,
    backend: &dyn Backend,
    i: usize,
    cfg: &Qwen35Config,
) -> anyhow::Result<Qwen35Layer> {
    let p = format!("blk.{i}");
    let attn = if cfg.layer_recurrent[i] {
        AttnKind::Delta {
            conv: load_optional(
                file,
                backend,
                &[
                    &format!("{p}.ssm_conv1d.weight"),
                    &format!("{p}.conv1d.weight"),
                ],
            )?,
            a_gate: load_optional(
                file,
                backend,
                &[
                    &format!("{p}.attn_gate.weight"),
                    &format!("{p}.ssm_in.weight"),
                ],
            )?,
            beta: load_optional(
                file,
                backend,
                &[
                    &format!("{p}.ssm_dt.weight"),
                    &format!("{p}.attn_beta.weight"),
                ],
            )?,
        }
    } else {
        AttnKind::Full {
            q_norm: load_weight(file, backend, &format!("{p}.attn_q_norm.weight"))?,
            k_norm: load_weight(file, backend, &format!("{p}.attn_k_norm.weight"))?,
        }
    };
    Ok(Qwen35Layer {
        attn,
        attn_norm: load_weight(file, backend, &format!("{p}.attn_norm.weight"))?,
        wq: load_weight(file, backend, &format!("{p}.attn_q.weight"))?,
        wk: load_weight(file, backend, &format!("{p}.attn_k.weight"))?,
        wv: load_weight(file, backend, &format!("{p}.attn_v.weight"))?,
        wo: load_weight(file, backend, &format!("{p}.attn_output.weight"))?,
        ffn_norm: load_weight(file, backend, &format!("{p}.ffn_norm.weight"))?,
        gate: load_weight(file, backend, &format!("{p}.ffn_gate.weight"))?,
        up: load_weight(file, backend, &format!("{p}.ffn_up.weight"))?,
        down: load_weight(file, backend, &format!("{p}.ffn_down.weight"))?,
    }
    .checked(cfg, i)?)
}

impl Qwen35Layer {
    fn checked(self, cfg: &Qwen35Config, i: usize) -> anyhow::Result<Self> {
        let q_out = cfg.n_head * cfg.head_dim;
        let kv_out = cfg.n_head_kv * cfg.head_dim;
        check_in_out(&self.wq, cfg.n_embd, q_out, i, "attn_q")?;
        check_in_out(&self.wk, cfg.n_embd, kv_out, i, "attn_k")?;
        check_in_out(&self.wv, cfg.n_embd, kv_out, i, "attn_v")?;
        check_in_out(&self.wo, q_out, cfg.n_embd, i, "attn_output")?;
        check_in_out(&self.gate, cfg.n_embd, cfg.n_ff, i, "ffn_gate")?;
        check_in_out(&self.up, cfg.n_embd, cfg.n_ff, i, "ffn_up")?;
        check_in_out(&self.down, cfg.n_ff, cfg.n_embd, i, "ffn_down")?;
        match &self.attn {
            AttnKind::Full { q_norm, k_norm } => {
                ensure!(
                    last_dim(q_norm) == cfg.head_dim,
                    "blk.{i}.attn_q_norm last dim {} != head_dim {}",
                    last_dim(q_norm),
                    cfg.head_dim
                );
                ensure!(
                    last_dim(k_norm) == cfg.head_dim,
                    "blk.{i}.attn_k_norm last dim {} != head_dim {}",
                    last_dim(k_norm),
                    cfg.head_dim
                );
            }
            AttnKind::Delta {
                a_gate,
                beta,
                conv,
            } => {
                if let Some(w) = a_gate {
                    check_in_out(w, cfg.n_embd, cfg.n_head_kv, i, "attn_gate")?;
                }
                if let Some(w) = beta {
                    check_in_out(w, cfg.n_embd, cfg.n_head_kv, i, "ssm_dt")?;
                }
                if let Some(ker) = conv {
                    let channels = q_out + 2 * kv_out;
                    match ker.shape.as_slice() {
                        [_] => {}
                        &[ch, _] if ch == channels => {}
                        other => {
                            anyhow::bail!(
                                "blk.{i}.ssm_conv1d shape {other:?} vs mixed channels {channels}"
                            )
                        }
                    }
                }
            }
        }
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

fn load_optional(
    file: &GgufFile,
    backend: &dyn Backend,
    names: &[&str],
) -> anyhow::Result<Option<Tensor>> {
    for n in names {
        if file.header().tensor(n).is_ok() {
            return Ok(Some(load_weight(file, backend, n)?));
        }
    }
    Ok(None)
}

fn is_mtp_name(name: &str, n_layer: usize) -> bool {
    if name.starts_with("mtp.") || name.starts_with("nextn.") {
        return true;
    }
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some((idx, _)) = rest.split_once('.') {
            if let Ok(i) = idx.parse::<usize>() {
                return i >= n_layer;
            }
        }
    }
    false
}

fn load_mtp(
    file: &GgufFile,
    backend: &dyn Backend,
    n_layer: usize,
) -> anyhow::Result<Vec<(String, Tensor)>> {
    let mut out = Vec::new();
    for t in &file.header().tensors {
        if is_mtp_name(&t.name, n_layer) {
            let w = load_weight(file, backend, &t.name)?;
            out.push((t.name.clone(), w));
        }
    }
    Ok(out)
}

fn load_weight(file: &GgufFile, backend: &dyn Backend, name: &str) -> anyhow::Result<Tensor> {
    let info = file
        .tensor_info(name)
        .with_context(|| format!("missing tensor {name}"))?;
    let shape = tutil::logical_shape(&info.dims);
    let bytes = file.tensor_bytes(name)?;
    if tutil::is_f32(info.ggml_type) {
        let data = tutil::f32_from_le_bytes(bytes)?;
        backend.upload_f32(&shape, &data)
    } else {
        let dtype = DType::from_ggml(info.ggml_type).ok_or_else(|| {
            anyhow!(
                "unsupported ggml type {} for tensor {name}",
                info.ggml_type
            )
        })?;
        backend.upload_quant(&shape, dtype, bytes)
    }
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

#[cfg(test)]
mod layer_kind_tests {
    use super::*;

    #[test]
    fn default_pattern_is_three_linear_one_full() {
        let rec: Vec<bool> = (0..8).map(default_is_recurrent).collect();
        assert_eq!(
            rec,
            vec![true, true, true, false, true, true, true, false]
        );
        assert_eq!(rec.iter().filter(|r| **r).count(), 6);
        assert!(default_is_recurrent(0));
        assert!(!default_is_recurrent(3));
    }

    #[test]
    fn mtp_name_detection() {
        assert!(is_mtp_name("mtp.0.output_norm.weight", 4));
        assert!(is_mtp_name("nextn.0.attn_norm.weight", 4));
        assert!(is_mtp_name("blk.4.attn_q.weight", 4));
        assert!(!is_mtp_name("blk.3.attn_q.weight", 4));
        assert!(!is_mtp_name("token_embd.weight", 4));
    }
}
