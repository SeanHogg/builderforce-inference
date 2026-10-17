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

use std::collections::HashMap;

use anyhow::{anyhow, ensure, Context, Result};
use bfi_core::resident::{DeviceTensor, Gpu, KvBuffer, Weight};
use bfi_gguf::{dequant_tensor, require_type, ModelHeader, StreamedTensor};

use crate::kv_plan::KvShape;
use crate::qwen3::{parse_config, Qwen3Config, ARCH_NAME};

/// Most tokens per enqueued forward chunk during prefill. Long contexts use
/// smaller chunks so the attention scores (`chunk · n_head · n_ctx` f32) fit one
/// storage binding; the score buffer never caps the context.
pub const PREFILL_CHUNK: usize = 64;

/// Error text prefix when a request would overrun the allocated KV cache.
/// `bfi-web` maps it to the `context_exceeded:` JS error.
pub const CONTEXT_EXCEEDED: &str = "context_exceeded";


enum Slot {
    Matrix(Weight),
    Vector(DeviceTensor),
}

/// Collects streamed tensors into device buffers, then assembles the model.
pub struct Qwen3ResidentLoader {
    gpu: Gpu,
    cfg: Qwen3Config,
    tensors: HashMap<String, Slot>,
    device_bytes: u64,
    params: u64,
}

impl Qwen3ResidentLoader {
    pub fn new(gpu: Gpu, header: &ModelHeader) -> Result<Self> {
        let arch = header
            .architecture()
            .ok_or_else(|| anyhow!("GGUF missing metadata key general.architecture"))?;
        ensure!(
            arch == ARCH_NAME,
            "architecture '{arch}' is not supported by the WebGPU engine (qwen3 only)"
        );
        if let Some(n) = header.arch_u64("expert_count") {
            ensure!(n == 0, "qwen3moe (expert_count={n}) is not supported; dense qwen3 only");
        }
        let cfg = parse_config(header)?;
        Ok(Self {
            gpu,
            cfg,
            tensors: HashMap::new(),
            device_bytes: 0,
            params: 0,
        })
    }

    /// Upload one tensor. Matrices stay packed; an unsupported weight type fails
    /// with [`bfi_core::resident::ResidentError::UnsupportedType`] (downcastable).
    pub fn accept(&mut self, t: StreamedTensor) -> Result<()> {
        let info = t.info;
        self.params += info.n_elems();
        let slot = if info.dims.len() <= 1 {
            let ty = require_type(info.ggml_type, &info.name)?;
            let mut v = vec![0.0f32; info.n_elems() as usize];
            dequant_tensor(ty, &t.bytes, &mut v)
                .with_context(|| format!("dequant {}", info.name))?;
            self.device_bytes += (v.len() * 4) as u64;
            Slot::Vector(self.gpu.upload_f32(&v))
        } else {
            let w = self
                .gpu
                .upload_weight(&info.name, info.ggml_type, &info.dims, &t.bytes)?;
            self.device_bytes += w.device_bytes() as u64;
            Slot::Matrix(w)
        };
        self.tensors.insert(info.name, slot);
        Ok(())
    }

    /// Bytes uploaded so far.
    pub fn device_bytes(&self) -> u64 {
        self.device_bytes
    }

    /// Context length the GGUF was trained for (`qwen3.context_length`).
    pub fn trained_ctx(&self) -> usize {
        self.cfg.n_ctx
    }

    /// The KV-relevant shape, for [`crate::kv_plan::served_context`].
    pub fn kv_shape(&self) -> KvShape {
        KvShape {
            n_layer: self.cfg.n_layer,
            n_head_kv: self.cfg.n_head_kv,
            head_dim: self.cfg.head_dim,
            trained_ctx: self.cfg.n_ctx,
        }
    }

    /// Assemble the model with a KV cache of `n_ctx` tokens, stored as f16
    /// when `kv_f16` (half the memory) else f32. `max_binding_bytes` is the
    /// device's largest storage binding; it sizes the prefill chunk.
    pub fn finish(mut self, n_ctx: usize, kv_f16: bool, max_binding_bytes: u64) -> Result<Qwen3Resident> {
        let tok_embd = self.matrix("token_embd.weight")?;
        let mut cfg = self.cfg;
        cfg.n_vocab = tok_embd.n_out as usize;
        ensure!(
            tok_embd.n_in as usize == cfg.n_embd,
            "token_embd n_embd {} != embedding_length {}",
            tok_embd.n_in,
            cfg.n_embd
        );
        // Bonsai 8B ships an untied lm_head; 1.7B / 4B tie it to the embeddings.
        let output = match self.tensors.remove("output.weight") {
            Some(Slot::Matrix(w)) => w,
            Some(Slot::Vector(_)) => return Err(anyhow!("output.weight must be a matrix")),
            None => tok_embd.clone(),
        };
        ensure!(
            output.n_in as usize == cfg.n_embd && output.n_out as usize == cfg.n_vocab,
            "output.weight is {}x{}, expected {}x{}",
            output.n_out,
            output.n_in,
            cfg.n_vocab,
            cfg.n_embd
        );
        self.cfg = cfg;
        let output_norm = self.vector("output_norm.weight", self.cfg.n_embd)?;
        let mut layers = Vec::with_capacity(self.cfg.n_layer);
        for i in 0..self.cfg.n_layer {
            layers.push(self.layer(i)?);
        }

        let n_ctx = n_ctx.min(self.cfg.n_ctx).max(1);
        let kv_row = self.cfg.n_head_kv * self.cfg.head_dim;
        ensure!(!kv_f16 || kv_row % 2 == 0, "f16 KV needs an even n_head_kv·head_dim, got {kv_row}");
        let kv_len = n_ctx * kv_row;
        let k = (0..self.cfg.n_layer).map(|_| self.gpu.empty_kv(kv_len, kv_f16)).collect();
        let v = (0..self.cfg.n_layer).map(|_| self.gpu.empty_kv(kv_len, kv_f16)).collect();
        let score_row = (self.cfg.n_head * n_ctx * 4).max(1) as u64;
        let prefill_chunk = ((max_binding_bytes / score_row) as usize).clamp(1, PREFILL_CHUNK);
        let half = self.cfg.n_rot.min(self.cfg.head_dim) / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| 1.0 / self.cfg.rope_freq_base.powf((2 * i) as f32 / (2 * half) as f32))
            .collect();
        let inv_freq = self.gpu.upload_f32(&inv);

        Ok(Qwen3Resident {
            gpu: self.gpu,
            cfg: self.cfg,
            n_ctx,
            kv_f16,
            prefill_chunk,
            params: self.params,
            inv_freq,
            tok_embd,
            output,
            output_norm,
            layers,
            k,
            v,
            n_past: 0,
        })
    }

    fn matrix(&mut self, name: &str) -> Result<Weight> {
        match self.tensors.remove(name) {
            Some(Slot::Matrix(w)) => Ok(w),
            Some(Slot::Vector(_)) => Err(anyhow!("{name} must be a matrix")),
            None => Err(anyhow!("missing tensor {name}")),
        }
    }

    fn vector(&mut self, name: &str, len: usize) -> Result<DeviceTensor> {
        match self.tensors.remove(name) {
            Some(Slot::Vector(v)) if v.len == len => Ok(v),
            Some(Slot::Vector(v)) => Err(anyhow!("{name} has {} elements, expected {len}", v.len)),
            Some(Slot::Matrix(_)) => Err(anyhow!("{name} must be a vector")),
            None => Err(anyhow!("missing tensor {name}")),
        }
    }

    fn layer(&mut self, i: usize) -> Result<Layer> {
        let c = &self.cfg;
        let (e, d, q_out, kv_out, ff) = (
            c.n_embd,
            c.head_dim,
            c.n_head * c.head_dim,
            c.n_head_kv * c.head_dim,
            c.n_ff,
        );
        let p = format!("blk.{i}");
        let layer = Layer {
            attn_norm: self.vector(&format!("{p}.attn_norm.weight"), e)?,
            wq: self.matrix(&format!("{p}.attn_q.weight"))?,
            wk: self.matrix(&format!("{p}.attn_k.weight"))?,
            wv: self.matrix(&format!("{p}.attn_v.weight"))?,
            wo: self.matrix(&format!("{p}.attn_output.weight"))?,
            q_norm: self.vector(&format!("{p}.attn_q_norm.weight"), d)?,
            k_norm: self.vector(&format!("{p}.attn_k_norm.weight"), d)?,
            ffn_norm: self.vector(&format!("{p}.ffn_norm.weight"), e)?,
            gate: self.matrix(&format!("{p}.ffn_gate.weight"))?,
            up: self.matrix(&format!("{p}.ffn_up.weight"))?,
            down: self.matrix(&format!("{p}.ffn_down.weight"))?,
        };
        for (w, n_in, n_out, name) in [
            (&layer.wq, e, q_out, "attn_q"),
            (&layer.wk, e, kv_out, "attn_k"),
            (&layer.wv, e, kv_out, "attn_v"),
            (&layer.wo, q_out, e, "attn_output"),
            (&layer.gate, e, ff, "ffn_gate"),
            (&layer.up, e, ff, "ffn_up"),
            (&layer.down, ff, e, "ffn_down"),
        ] {
            ensure!(
                w.n_in as usize == n_in && w.n_out as usize == n_out,
                "blk.{i}.{name} is {}x{} (out x in), expected {n_out}x{n_in}",
                w.n_out,
                w.n_in
            );
        }
        Ok(layer)
    }
}

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

    /// Keep only the first `n` cached tokens (prompt-prefix reuse across turns).
    pub fn rewind(&mut self, n: usize) {
        self.n_past = self.n_past.min(n);
    }

    /// Append `tokens` to the sequence and return the last token's logits
    /// (`[n_vocab]`, still on the device). Errors with [`CONTEXT_EXCEEDED`] when
    /// the cache would overflow.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<DeviceTensor> {
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
            logits = self.forward_chunk(chunk, ci == last);
        }
        logits.ok_or_else(|| anyhow!("forward produced no logits"))
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
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use bfi_core::resident::{block_on, open_native};
    use bfi_core::{max_rel_err, CpuBackend};
    use bfi_gguf::writer::{write_gguf, TensorWrite};
    use bfi_gguf::{lookup_type, quantize_row_q1_0, GgufFile, MetaValue, TensorStream, TYPE_F32, TYPE_Q1_0};

    use crate::SeqState;

    fn pattern(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((((i * 7919 + seed * 104_729) % 2003) as f32) / 1001.5 - 1.0) * 0.5)
            .collect()
    }

    fn q1(name: &str, n_in: u64, n_out: u64, seed: usize) -> TensorWrite {
        let w = pattern((n_in * n_out) as usize, seed);
        let mut data = vec![0u8; w.len() / 128 * 18];
        for (b, c) in w.chunks(128).enumerate() {
            quantize_row_q1_0(c, &mut data[b * 18..b * 18 + 18]);
        }
        TensorWrite {
            name: name.into(),
            dims: vec![n_in, n_out],
            ty: lookup_type(TYPE_Q1_0).unwrap(),
            data,
        }
    }

    fn norm(name: &str, n: u64, seed: usize) -> TensorWrite {
        let data = pattern(n as usize, seed)
            .iter()
            .flat_map(|v| (1.0 + v * 0.2).to_le_bytes())
            .collect();
        TensorWrite {
            name: name.into(),
            dims: vec![n],
            ty: lookup_type(TYPE_F32).unwrap(),
            data,
        }
    }

    /// 1-bit Bonsai-shaped toy: every matrix Q1_0, untied lm_head, GQA 2q/1kv,
    /// head_dim 64, two layers.
    fn bonsai_toy() -> Vec<u8> {
        let (e, d, ff, vocab) = (128u64, 64u64, 256u64, 16u64);
        let mut meta = BTreeMap::new();
        let s = |v: &str| MetaValue::String(v.into());
        meta.insert("general.architecture".into(), s("qwen3"));
        meta.insert("qwen3.block_count".into(), MetaValue::U32(2));
        meta.insert("qwen3.embedding_length".into(), MetaValue::U32(e as u32));
        meta.insert("qwen3.feed_forward_length".into(), MetaValue::U32(ff as u32));
        meta.insert("qwen3.context_length".into(), MetaValue::U32(32));
        meta.insert("qwen3.attention.head_count".into(), MetaValue::U32(2));
        meta.insert("qwen3.attention.head_count_kv".into(), MetaValue::U32(1));
        meta.insert("qwen3.attention.key_length".into(), MetaValue::U32(d as u32));
        meta.insert("qwen3.attention.layer_norm_rms_epsilon".into(), MetaValue::F32(1e-6));
        meta.insert("qwen3.rope.freq_base".into(), MetaValue::F32(1_000_000.0));
        let mut t = vec![
            q1("token_embd.weight", e, vocab, 1),
            norm("output_norm.weight", e, 2),
            q1("output.weight", e, vocab, 3),
        ];
        for i in 0..2usize {
            let p = format!("blk.{i}");
            let k = 10 + i * 20;
            t.extend([
                norm(&format!("{p}.attn_norm.weight"), e, k),
                q1(&format!("{p}.attn_q.weight"), e, 2 * d, k + 1),
                q1(&format!("{p}.attn_k.weight"), e, d, k + 2),
                q1(&format!("{p}.attn_v.weight"), e, d, k + 3),
                q1(&format!("{p}.attn_output.weight"), 2 * d, e, k + 4),
                norm(&format!("{p}.attn_q_norm.weight"), d, k + 5),
                norm(&format!("{p}.attn_k_norm.weight"), d, k + 6),
                norm(&format!("{p}.ffn_norm.weight"), e, k + 7),
                q1(&format!("{p}.ffn_gate.weight"), e, ff, k + 8),
                q1(&format!("{p}.ffn_up.weight"), e, ff, k + 9),
                q1(&format!("{p}.ffn_down.weight"), ff, e, k + 10),
            ]);
        }
        write_gguf(&meta, &t).unwrap()
    }

    #[test]
    fn context_budget_is_bounded_by_kv_binding_and_training() {
        let Some(gpu) = bfi_core::gpu_or_skip(open_native(), "the resident qwen3 tests") else { return };
        let bytes = bonsai_toy();
        let mut stream = TensorStream::new();
        stream.push(&bytes).unwrap();
        stream.flush_header().unwrap();
        let loader = Qwen3ResidentLoader::new(gpu, stream.header().unwrap()).unwrap();
        // toy: 2 layers, 1 kv head × 64; trained ctx 32.
        let shape = loader.kv_shape();
        assert_eq!(shape, KvShape { n_layer: 2, n_head_kv: 1, head_dim: 64, trained_ctx: 32 });
        assert_eq!(crate::kv_plan::kv_bytes_per_token(&shape, false), 2 * 2 * 64 * 4);
        assert_eq!(crate::served_context(&shape, false, u64::MAX, u64::MAX), 32);
        assert_eq!(crate::served_context(&shape, false, 10 * 1024, u64::MAX), 10);
        assert_eq!(crate::served_context(&shape, true, 10 * 1024, u64::MAX), 20);
    }

    #[test]
    fn resident_matches_cpu_qwen3_prompt_then_decode_f32_kv() {
        resident_vs_cpu(false, 2e-3, u64::MAX);
    }

    #[test]
    fn resident_matches_cpu_qwen3_prompt_then_decode_f16_kv() {
        resident_vs_cpu(true, 1e-2, u64::MAX);
    }

    #[test]
    fn small_binding_shrinks_prefill_chunk_not_results() {
        // 2 heads · 32 ctx · 4 B = 256 B per score row → a 512 B binding forces
        // 2-token prefill chunks; logits must not change.
        resident_vs_cpu(false, 2e-3, 512);
    }

    fn resident_vs_cpu(kv_f16: bool, tol: f32, max_binding: u64) {
        let Some(gpu) = bfi_core::gpu_or_skip(open_native(), "the resident qwen3 tests") else { return };
        let bytes = bonsai_toy();

        let file = GgufFile::from_bytes("toy.gguf", bytes.clone()).unwrap();
        let cpu_model = crate::load_model(&file, Arc::new(CpuBackend)).unwrap();
        let mut state = SeqState::allocate(&cpu_model.state_layout());
        let prompt = [1u32, 5, 9, 3, 14];
        let pos: Vec<i32> = (0..prompt.len() as i32).collect();
        let want_prompt = cpu_model.forward(&prompt, &pos, &mut state).unwrap();
        let want_step = cpu_model.forward(&[7], &[prompt.len() as i32], &mut state).unwrap();

        let mut stream = TensorStream::new();
        stream.push(&bytes).unwrap();
        stream.flush_header().unwrap();
        let mut loader = Qwen3ResidentLoader::new(gpu.clone(), stream.header().unwrap()).unwrap();
        let mut s2 = TensorStream::new();
        for c in bytes.chunks(4096) {
            for t in s2.push(c).unwrap() {
                loader.accept(t).unwrap();
            }
        }
        for t in s2.flush_header().unwrap() {
            loader.accept(t).unwrap();
        }
        s2.finish().unwrap();
        let mut model = loader.finish(32, kv_f16, max_binding).unwrap();
        assert_eq!(model.kv_f16(), kv_f16);
        assert_eq!(model.n_vocab(), 16);

        let got = model.forward(&prompt).unwrap();
        let got = block_on(gpu.read_f32(&got)).unwrap();
        let err = max_rel_err(&got, want_prompt.as_slice());
        assert!(err <= tol, "prompt logits rel err {err}\n{got:?}\n{:?}", want_prompt.as_slice());

        let got = model.forward(&[7]).unwrap();
        let got = block_on(gpu.read_f32(&got)).unwrap();
        let err = max_rel_err(&got, want_step.as_slice());
        assert!(err <= tol, "decode logits rel err {err}");

        let over = model.forward(&vec![1u32; 32]).err().unwrap().to_string();
        assert!(over.starts_with(CONTEXT_EXCEEDED), "{over}");
    }
}
