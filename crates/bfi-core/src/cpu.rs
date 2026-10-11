use crate::backend::{Backend, GatedDeltaParams, RopeParams};
use crate::error::CoreError;
use crate::tensor::{n_elems_of, Storage, Tensor};
use crate::DType;
use anyhow::{anyhow, bail, Context};
use bfi_gguf::{dequant_tensor, lookup_type, GgmlType};
use rayon::prelude::*;

pub struct CpuBackend;

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn alloc(&self, shape: &[usize], dtype: DType) -> anyhow::Result<Tensor> {
        if dtype == DType::F32 {
            let n = n_elems_of(shape);
            return Ok(Tensor::f32(shape.to_vec(), vec![0.0; n]));
        }
        let n = nbytes(shape, dtype)?;
        Ok(Tensor {
            shape: shape.to_vec(),
            dtype,
            storage: Storage::Quant {
                ggml_id: dtype.ggml_id(),
                bytes: vec![0u8; n],
            },
        })
    }

    fn upload_f32(&self, shape: &[usize], data: &[f32]) -> anyhow::Result<Tensor> {
        let n = n_elems_of(shape);
        if data.len() != n {
            return Err(shape_err(
                "upload_f32",
                format!("data len {} != n_elems {n}", data.len()),
            ));
        }
        Ok(Tensor::f32(shape.to_vec(), data.to_vec()))
    }

    fn upload_quant(&self, shape: &[usize], dtype: DType, bytes: &[u8]) -> anyhow::Result<Tensor> {
        if dtype == DType::F32 {
            bail!("upload_quant: use upload_f32 for F32");
        }
        let n = nbytes(shape, dtype)?;
        if bytes.len() != n {
            return Err(shape_err(
                "upload_quant",
                format!("bytes {} != expected {n}", bytes.len()),
            ));
        }
        Ok(Tensor {
            shape: shape.to_vec(),
            dtype,
            storage: Storage::Quant {
                ggml_id: dtype.ggml_id(),
                bytes: bytes.to_vec(),
            },
        })
    }

    fn dequant(&self, t: &Tensor) -> anyhow::Result<Tensor> {
        match &t.storage {
            Storage::F32(v) => Ok(Tensor::f32(t.shape.clone(), v.clone())),
            Storage::Quant { ggml_id, bytes } => {
                let ty = ggml_ty(*ggml_id)?;
                let n = t.n_elems();
                let mut out = vec![0.0f32; n];
                dequant_tensor(ty, bytes, &mut out).map_err(|e| anyhow!("{e}"))?;
                Ok(Tensor::f32(t.shape.clone(), out))
            }
        }
    }

    fn matmul_q(&self, weights: &Tensor, x: &Tensor) -> anyhow::Result<Tensor> {
        let x_f = x.as_f32().context("matmul_q x")?;
        let (n_tokens, n_out, n_in) = matmul_dims(weights, x)?;
        let mut out = vec![0.0f32; n_tokens * n_out];
        match &weights.storage {
            Storage::F32(w) => matmul_f32(w, x_f, &mut out, n_tokens, n_out, n_in),
            Storage::Quant { ggml_id, bytes } => {
                let ty = ggml_ty(*ggml_id)?;
                matmul_quant(ty, bytes, x_f, &mut out, n_tokens, n_out, n_in)?;
            }
        }
        Ok(Tensor::f32(vec![n_tokens, n_out], out))
    }

    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> anyhow::Result<Tensor> {
        let x_f = x.as_f32()?;
        let w = weight.as_f32()?;
        let d = last_dim(&x.shape)?;
        if w.len() != d && w.len() != x_f.len() {
            return Err(shape_err(
                "rms_norm",
                format!("weight len {} != last dim {d}", w.len()),
            ));
        }
        if x_f.len() % d != 0 {
            return Err(shape_err("rms_norm", "n_elems not divisible by last dim"));
        }
        let n_rows = x_f.len() / d;
        let mut out = vec![0.0f32; x_f.len()];
        let w_per_row = w.len() == d;
        for r in 0..n_rows {
            let xs = &x_f[r * d..(r + 1) * d];
            let mean_sq = xs.iter().map(|v| v * v).sum::<f32>() / d as f32;
            let scale = (mean_sq + eps).sqrt().recip();
            let ws = if w_per_row {
                w
            } else {
                &w[r * d..(r + 1) * d]
            };
            for i in 0..d {
                out[r * d + i] = xs[i] * scale * ws[i];
            }
        }
        Ok(Tensor::f32(x.shape.clone(), out))
    }

    fn rope(&self, x: &mut Tensor, pos: &[i32], params: &RopeParams) -> anyhow::Result<()> {
        rope_impl(x, pos, params, false)
    }

    fn rope_multi(&self, x: &mut Tensor, pos: &[i32], params: &RopeParams) -> anyhow::Result<()> {
        rope_impl(x, pos, params, true)
    }

    fn softmax_inplace(&self, x: &mut Tensor) -> anyhow::Result<()> {
        let shape = x.shape.clone();
        let d = last_dim(&shape)?;
        let data = x.as_f32_mut()?;
        if d == 0 || data.is_empty() {
            return Ok(());
        }
        if data.len() % d != 0 {
            return Err(shape_err("softmax", "n_elems not divisible by last dim"));
        }
        for row in data.chunks_mut(d) {
            let mut max = f32::NEG_INFINITY;
            for &v in row.iter() {
                if v > max {
                    max = v;
                }
            }
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            if sum > 0.0 {
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
        }
        Ok(())
    }

    fn silu(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let x_f = x.as_f32()?;
        let out: Vec<f32> = x_f.par_iter().copied().map(silu_f).collect();
        Ok(Tensor::f32(x.shape.clone(), out))
    }

    fn swiglu(&self, gate: &Tensor, up: &Tensor) -> anyhow::Result<Tensor> {
        let g = gate.as_f32()?;
        let u = up.as_f32()?;
        if g.len() != u.len() {
            return Err(shape_err(
                "swiglu",
                format!("gate {} vs up {}", g.len(), u.len()),
            ));
        }
        let out: Vec<f32> = g
            .par_iter()
            .zip(u.par_iter())
            .map(|(&gv, &uv)| silu_f(gv) * uv)
            .collect();
        Ok(Tensor::f32(gate.shape.clone(), out))
    }

    fn sigmoid(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let x_f = x.as_f32()?;
        let out: Vec<f32> = x_f.par_iter().copied().map(sigmoid_f).collect();
        Ok(Tensor::f32(x.shape.clone(), out))
    }

    fn softplus(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let x_f = x.as_f32()?;
        let out: Vec<f32> = x_f.par_iter().copied().map(softplus_f).collect();
        Ok(Tensor::f32(x.shape.clone(), out))
    }

    fn l2_norm(&self, x: &Tensor, eps: f32) -> anyhow::Result<Tensor> {
        let x_f = x.as_f32()?;
        let d = last_dim(&x.shape)?;
        if d == 0 {
            return Ok(Tensor::f32(x.shape.clone(), x_f.to_vec()));
        }
        if x_f.len() % d != 0 {
            return Err(shape_err("l2_norm", "n_elems not divisible by last dim"));
        }
        let mut out = vec![0.0f32; x_f.len()];
        for (src, dst) in x_f.chunks(d).zip(out.chunks_mut(d)) {
            let n2: f32 = src.iter().map(|v| v * v).sum();
            let inv = (n2 + eps).sqrt().recip();
            for i in 0..d {
                dst[i] = src[i] * inv;
            }
        }
        Ok(Tensor::f32(x.shape.clone(), out))
    }

    fn conv1d_causal(&self, x: &Tensor, kernel: &Tensor) -> anyhow::Result<Tensor> {
        let x_f = x.as_f32()?;
        let k_f = kernel.as_f32()?;
        if x.shape.len() != 2 {
            return Err(shape_err(
                "conv1d_causal",
                format!("x expected [tokens, channels], got {:?}", x.shape),
            ));
        }
        let tokens = x.shape[0];
        let channels = x.shape[1];
        let (k_ch, k_len) = match kernel.shape.as_slice() {
            &[len] => (1usize, len),
            &[ch, len] => (ch, len),
            other => {
                return Err(shape_err(
                    "conv1d_causal",
                    format!("kernel expected [k_len] or [channels, k_len], got {other:?}"),
                ))
            }
        };
        if k_len == 0 {
            bail!("conv1d_causal: empty kernel");
        }
        if k_ch != 1 && k_ch != channels {
            return Err(shape_err(
                "conv1d_causal",
                format!("kernel channels {k_ch} vs x {channels}"),
            ));
        }
        let broadcast = k_ch == 1 && channels > 1 && kernel.shape.len() == 1;
        let mut out = vec![0.0f32; tokens * channels];
        for t in 0..tokens {
            for c in 0..channels {
                let mut acc = 0.0f32;
                for k in 0..k_len {
                    let src_t = t as isize - k as isize;
                    if src_t < 0 {
                        continue;
                    }
                    let xv = x_f[src_t as usize * channels + c];
                    let kv = if broadcast {
                        k_f[k]
                    } else if kernel.shape.len() == 1 {
                        k_f[k]
                    } else {
                        k_f[c * k_len + k]
                    };
                    acc += kv * xv;
                }
                out[t * channels + c] = acc;
            }
        }
        Ok(Tensor::f32(vec![tokens, channels], out))
    }

    fn gated_delta_step(&self, p: GatedDeltaParams<'_>) -> anyhow::Result<Tensor> {
        let q = p.q.as_f32()?;
        let k = p.k.as_f32()?;
        let v = p.v.as_f32()?;
        let g = p.g.as_f32()?;
        let beta = p.beta.as_f32()?;
        let (tq, hq, d) = parse_bth(p.q, "gated_delta q")?;
        let (tk, hkv, dk) = parse_bth(p.k, "gated_delta k")?;
        let (tv, hv, dv) = parse_bth(p.v, "gated_delta v")?;
        if tq != tk || tq != tv {
            return Err(shape_err("gated_delta", "q/k/v token counts differ"));
        }
        if d != dk || d != dv || hkv != hv {
            return Err(shape_err("gated_delta", "q/k/v head or dim mismatch"));
        }
        if hq % hkv != 0 {
            return Err(shape_err("gated_delta", format!("hq {hq} % hkv {hkv} != 0")));
        }
        if p.state.shape.as_slice() != [hkv, d, d] {
            return Err(shape_err(
                "gated_delta",
                format!("state expected [{hkv}, {d}, {d}], got {:?}", p.state.shape),
            ));
        }
        let state = p.state.as_f32_mut()?;
        let group = hq / hkv;
        let mut out = vec![0.0f32; tq * hq * d];

        for t in 0..tq {
            for h in 0..hkv {
                let decay = sigmoid_f(scalar_at(g, t, h, tq, hkv, "g")?);
                let b = scalar_at(beta, t, h, tq, hkv, "beta")?;
                let sbase = h * d * d;
                let kbase = (t * hkv + h) * d;
                let vbase = (t * hkv + h) * d;
                for i in 0..d {
                    let ki = k[kbase + i];
                    for j in 0..d {
                        let idx = sbase + i * d + j;
                        state[idx] = state[idx] * decay + ki * v[vbase + j] * b;
                    }
                }
                for gq in 0..group {
                    let qh = h * group + gq;
                    let qbase = (t * hq + qh) * d;
                    for j in 0..d {
                        let mut s = 0.0f32;
                        for i in 0..d {
                            s += q[qbase + i] * state[sbase + i * d + j];
                        }
                        out[qbase + j] = s;
                    }
                }
            }
        }
        Ok(Tensor::f32(vec![tq, hq, d], out))
    }

    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> anyhow::Result<Tensor> {
        let (tq, hq, d) = parse_bth(q, "attention q")?;
        let (tk, hkv, dk) = parse_bth(k, "attention k")?;
        let (tv, hv, dv) = parse_bth(v, "attention v")?;
        if d != dk || d != dv {
            return Err(shape_err("attention", "head dim mismatch"));
        }
        if tk != tv || hkv != hv {
            return Err(shape_err("attention", "k/v shape mismatch"));
        }
        if hq % hkv != 0 {
            return Err(shape_err("attention", format!("hq {hq} % hkv {hkv} != 0")));
        }
        let q_f = q.as_f32()?;
        let k_f = k.as_f32()?;
        let v_f = v.as_f32()?;
        let reps = hq / hkv;
        let scale = 1.0 / (d as f32).sqrt();
        let mut out = vec![0.0f32; tq * hq * d];

        for h in 0..hq {
            let hkv_i = h / reps;
            for t in 0..tq {
                let qv = &q_f[(t * hq + h) * d..(t * hq + h) * d + d];
                let mut scores = vec![0.0f32; tk];
                let qpos = if tq == tk { t } else { tk.saturating_sub(tq) + t };
                let mut max = f32::NEG_INFINITY;
                for s in 0..tk {
                    if causal && s > qpos {
                        scores[s] = f32::NEG_INFINITY;
                        continue;
                    }
                    let kv = &k_f[(s * hkv + hkv_i) * d..(s * hkv + hkv_i) * d + d];
                    let mut dot = 0.0f32;
                    for i in 0..d {
                        dot += qv[i] * kv[i];
                    }
                    scores[s] = dot * scale;
                    if scores[s] > max {
                        max = scores[s];
                    }
                }
                let mut sum = 0.0f32;
                for s in 0..tk {
                    if scores[s].is_finite() {
                        scores[s] = (scores[s] - max).exp();
                        sum += scores[s];
                    } else {
                        scores[s] = 0.0;
                    }
                }
                if sum > 0.0 {
                    for s in scores.iter_mut() {
                        *s /= sum;
                    }
                }
                let ov = &mut out[(t * hq + h) * d..(t * hq + h) * d + d];
                for s in 0..tk {
                    let w = scores[s];
                    if w == 0.0 {
                        continue;
                    }
                    let vv = &v_f[(s * hkv + hkv_i) * d..(s * hkv + hkv_i) * d + d];
                    for i in 0..d {
                        ov[i] += w * vv[i];
                    }
                }
            }
        }
        Ok(Tensor::f32(vec![tq, hq, d], out))
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> anyhow::Result<Tensor> {
        binary(a, b, |x, y| x + y, "add")
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> anyhow::Result<Tensor> {
        binary(a, b, |x, y| x * y, "mul")
    }

    fn get_rows(&self, weight: &Tensor, ids: &[i32]) -> anyhow::Result<Tensor> {
        if weight.shape.len() < 2 {
            return Err(shape_err(
                "get_rows",
                format!("weight expected [n_vocab, n_embd], got {:?}", weight.shape),
            ));
        }
        let n_vocab = weight.shape[0];
        let n_embd = weight.n_elems() / n_vocab;
        if n_embd * n_vocab != weight.n_elems() {
            return Err(shape_err("get_rows", "weight n_elems not n_vocab * n_embd"));
        }
        let mut out = vec![0.0f32; ids.len() * n_embd];
        match &weight.storage {
            Storage::F32(w) => {
                for (r, &id) in ids.iter().enumerate() {
                    let i = row_index(id, n_vocab)?;
                    out[r * n_embd..(r + 1) * n_embd]
                        .copy_from_slice(&w[i * n_embd..(i + 1) * n_embd]);
                }
            }
            Storage::Quant { ggml_id, bytes } => {
                let ty = ggml_ty(*ggml_id)?;
                let row_bytes = ty
                    .bytes_for_elems(n_embd as u64)
                    .ok_or_else(|| anyhow!("get_rows: n_embd {n_embd} not a multiple of block"))?
                    as usize;
                for (r, &id) in ids.iter().enumerate() {
                    let i = row_index(id, n_vocab)?;
                    let src = &bytes[i * row_bytes..(i + 1) * row_bytes];
                    dequant_tensor(ty, src, &mut out[r * n_embd..(r + 1) * n_embd])
                        .map_err(|e| anyhow!("{e}"))?;
                }
            }
        }
        Ok(Tensor::f32(vec![ids.len(), n_embd], out))
    }
}

fn shape_err(op: &'static str, msg: impl Into<String>) -> anyhow::Error {
    CoreError::Shape {
        op,
        msg: msg.into(),
    }
    .into()
}

fn ggml_ty(id: u32) -> anyhow::Result<&'static GgmlType> {
    lookup_type(id).ok_or_else(|| CoreError::UnknownType(id).into())
}

fn nbytes(shape: &[usize], dtype: DType) -> anyhow::Result<usize> {
    let n = n_elems_of(shape) as u64;
    let ty = ggml_ty(dtype.ggml_id())?;
    ty.bytes_for_elems(n)
        .map(|b| b as usize)
        .ok_or_else(|| anyhow!("n_elems {n} is not a multiple of {} block {}", ty.name, ty.block_elems))
}

fn last_dim(shape: &[usize]) -> anyhow::Result<usize> {
    shape
        .last()
        .copied()
        .ok_or_else(|| shape_err("shape", "empty shape"))
}

fn matmul_dims(weights: &Tensor, x: &Tensor) -> anyhow::Result<(usize, usize, usize)> {
    if x.shape.is_empty() || weights.shape.is_empty() {
        return Err(shape_err("matmul_q", "empty shape"));
    }
    let n_in = *x.shape.last().unwrap_or(&0);
    if n_in == 0 {
        return Err(shape_err("matmul_q", "n_in is 0"));
    }
    if x.n_elems() % n_in != 0 {
        return Err(shape_err("matmul_q", "x n_elems not divisible by n_in"));
    }
    let n_tokens = x.n_elems() / n_in;
    let w_in = *weights.shape.last().unwrap_or(&0);
    if w_in != n_in {
        return Err(shape_err(
            "matmul_q",
            format!("weight in-dim {w_in} != x in-dim {n_in}"),
        ));
    }
    if weights.n_elems() % w_in != 0 {
        return Err(shape_err("matmul_q", "weights n_elems not divisible by n_in"));
    }
    let n_out = weights.n_elems() / w_in;
    Ok((n_tokens, n_out, n_in))
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn matmul_f32(w: &[f32], x: &[f32], out: &mut [f32], n_tokens: usize, n_out: usize, n_in: usize) {
    if n_tokens <= 1 {
        let xrow = x;
        out.par_iter_mut().enumerate().for_each(|(o, dst)| {
            let wr = &w[o * n_in..(o + 1) * n_in];
            *dst = dot(wr, xrow);
        });
        return;
    }
    out.par_chunks_mut(n_out).enumerate().for_each(|(t, row)| {
        let xr = &x[t * n_in..(t + 1) * n_in];
        for o in 0..n_out {
            let wr = &w[o * n_in..(o + 1) * n_in];
            row[o] = dot(wr, xr);
        }
    });
}

fn matmul_quant(
    ty: &GgmlType,
    bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    n_tokens: usize,
    n_out: usize,
    n_in: usize,
) -> anyhow::Result<()> {
    if n_in % ty.block_elems != 0 {
        let mut w = vec![0.0f32; n_out * n_in];
        dequant_tensor(ty, bytes, &mut w).map_err(|e| anyhow!("{e}"))?;
        matmul_f32(&w, x, out, n_tokens, n_out, n_in);
        return Ok(());
    }
    let row_blocks = n_in / ty.block_elems;
    let row_bytes = row_blocks * ty.block_bytes;
    let cols: Vec<Vec<f32>> = (0..n_out)
        .into_par_iter()
        .map(|o| {
            let wrow = &bytes[o * row_bytes..(o + 1) * row_bytes];
            let mut col = vec![0.0f32; n_tokens];
            for t in 0..n_tokens {
                let xr = &x[t * n_in..(t + 1) * n_in];
                col[t] = vec_dot_row(ty, wrow, xr);
            }
            col
        })
        .collect();
    for o in 0..n_out {
        for t in 0..n_tokens {
            out[t * n_out + o] = cols[o][t];
        }
    }
    Ok(())
}

fn vec_dot_row(ty: &GgmlType, wrow: &[u8], x: &[f32]) -> f32 {
    let n_blocks = wrow.len() / ty.block_bytes;
    if let Some(vd) = ty.vec_dot {
        let mut s = 0.0f32;
        for b in 0..n_blocks {
            let blk = &wrow[b * ty.block_bytes..(b + 1) * ty.block_bytes];
            let xr = &x[b * ty.block_elems..(b + 1) * ty.block_elems];
            s += vd(blk, xr);
        }
        s
    } else {
        let mut tmp = vec![0.0f32; x.len()];
        for i in 0..n_blocks {
            (ty.dequant_row)(
                &wrow[i * ty.block_bytes..(i + 1) * ty.block_bytes],
                &mut tmp[i * ty.block_elems..(i + 1) * ty.block_elems],
            );
        }
        dot(&tmp, x)
    }
}

fn rope_impl(
    x: &mut Tensor,
    pos: &[i32],
    params: &RopeParams,
    multi: bool,
) -> anyhow::Result<()> {
    let _ = (params.n_ctx, params.beta_fast, params.beta_slow, params.ext_factor);
    let (n_tokens, n_heads, head_dim) = match x.shape.as_slice() {
        &[tok, heads, dim] => (tok, heads, dim),
        &[tok, dim] => (tok, 1, dim),
        other => {
            return Err(shape_err(
                "rope",
                format!("expected [tokens, heads, dim], got {other:?}"),
            ))
        }
    };
    if pos.len() != n_tokens {
        return Err(shape_err(
            "rope",
            format!("pos len {} != n_tokens {n_tokens}", pos.len()),
        ));
    }
    let data = x.as_f32_mut()?;
    let n_rot = if params.n_rot > 0 {
        params.n_rot.min(head_dim)
    } else {
        params.n_dims.min(head_dim)
    };
    let attn = if params.attn_factor == 0.0 {
        1.0
    } else {
        params.attn_factor
    };
    for t in 0..n_tokens {
        let p = pos[t] as f32;
        for h in 0..n_heads {
            let row = &mut data[(t * n_heads + h) * head_dim..(t * n_heads + h) * head_dim + head_dim];
            if multi {
                if let Some(sections) = params.sections {
                    apply_rope_multi(row, p, n_rot, sections, params.freq_base, params.freq_scale, attn);
                } else {
                    apply_rope_pairs(row, p, n_rot, params.freq_base, params.freq_scale, attn);
                }
            } else {
                apply_rope_pairs(row, p, n_rot, params.freq_base, params.freq_scale, attn);
            }
        }
    }
    Ok(())
}

fn apply_rope_pairs(row: &mut [f32], pos: f32, n_rot: usize, freq_base: f32, freq_scale: f32, attn: f32) {
    let n_rot = n_rot.min(row.len());
    let n_rot = n_rot - (n_rot % 2);
    if n_rot == 0 {
        return;
    }
    for i in (0..n_rot).step_by(2) {
        let theta = (pos * freq_scale) / freq_base.powf(i as f32 / n_rot as f32);
        let (sin, cos) = theta.sin_cos();
        let cos = cos * attn;
        let sin = sin * attn;
        let x0 = row[i];
        let x1 = row[i + 1];
        row[i] = x0 * cos - x1 * sin;
        row[i + 1] = x0 * sin + x1 * cos;
    }
}

fn apply_rope_multi(
    row: &mut [f32],
    pos: f32,
    n_rot: usize,
    sections: [usize; 4],
    freq_base: f32,
    freq_scale: f32,
    attn: f32,
) {
    let sum: usize = sections.iter().sum();
    let pair_counts = if n_rot >= 2 && sum == n_rot / 2 {
        sections
    } else if sum == n_rot {
        [
            sections[0] / 2,
            sections[1] / 2,
            sections[2] / 2,
            sections[3] / 2,
        ]
    } else {
        sections
    };
    let mut offset = 0usize;
    for &npairs in &pair_counts {
        let ne = npairs.saturating_mul(2);
        if ne == 0 {
            continue;
        }
        if offset + ne > row.len() {
            break;
        }
        apply_rope_pairs(&mut row[offset..offset + ne], pos, ne, freq_base, freq_scale, attn);
        offset += ne;
    }
}

fn sigmoid_f(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn silu_f(x: f32) -> f32 {
    x * sigmoid_f(x)
}

fn softplus_f(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

fn parse_bth(t: &Tensor, what: &str) -> anyhow::Result<(usize, usize, usize)> {
    match t.shape.as_slice() {
        &[h, d] => Ok((1, h, d)),
        &[tok, h, d] => Ok((tok, h, d)),
        other => Err(shape_err(
            "shape",
            format!("{what}: expected [t,h,d] or [h,d], got {other:?}"),
        )),
    }
}

fn scalar_at(data: &[f32], t: usize, h: usize, n_tokens: usize, n_heads: usize, what: &str) -> anyhow::Result<f32> {
    let n = data.len();
    if n == n_tokens * n_heads {
        return Ok(data[t * n_heads + h]);
    }
    if n == n_tokens * n_heads * 1 {
        return Ok(data[t * n_heads + h]);
    }
    if n_heads > 0 && n % (n_tokens * n_heads) == 0 {
        let inner = n / (n_tokens * n_heads);
        return Ok(data[(t * n_heads + h) * inner]);
    }
    if n == 1 {
        return Ok(data[0]);
    }
    Err(shape_err(
        "gated_delta",
        format!("{what} len {n} does not match tokens={n_tokens} heads={n_heads}"),
    ))
}

fn binary(
    a: &Tensor,
    b: &Tensor,
    op: fn(f32, f32) -> f32,
    name: &'static str,
) -> anyhow::Result<Tensor> {
    let aa = a.as_f32()?;
    let bb = b.as_f32()?;
    if aa.len() == bb.len() {
        let out: Vec<f32> = aa.iter().zip(bb.iter()).map(|(&x, &y)| op(x, y)).collect();
        return Ok(Tensor::f32(a.shape.clone(), out));
    }
    if bb.len() == 1 {
        let s = bb[0];
        let out: Vec<f32> = aa.iter().map(|&x| op(x, s)).collect();
        return Ok(Tensor::f32(a.shape.clone(), out));
    }
    if aa.len() % bb.len() == 0 {
        let out: Vec<f32> = aa
            .iter()
            .enumerate()
            .map(|(i, &x)| op(x, bb[i % bb.len()]))
            .collect();
        return Ok(Tensor::f32(a.shape.clone(), out));
    }
    Err(shape_err(
        name,
        format!("len {} vs {}", aa.len(), bb.len()),
    ))
}

fn row_index(id: i32, n_vocab: usize) -> anyhow::Result<usize> {
    let i = usize::try_from(id).map_err(|_| CoreError::RowId { id, n: n_vocab })?;
    if i >= n_vocab {
        return Err(CoreError::RowId { id, n: n_vocab }.into());
    }
    Ok(i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        select_backend, select_backend_from, Backend, DType, RopeParams, Storage, Tensor,
    };

    fn cpu() -> CpuBackend {
        CpuBackend
    }

    fn rope_params(n_rot: usize) -> RopeParams {
        RopeParams {
            n_dims: n_rot,
            n_ctx: 128,
            freq_base: 10_000.0,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            n_rot,
            sections: None,
        }
    }

    fn l2(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn rms_norm_ones() {
        let x = Tensor::f32(vec![2, 3], vec![1.0; 6]);
        let w = Tensor::f32(vec![3], vec![1.0; 3]);
        let y = cpu().rms_norm(&x, &w, 0.0).unwrap();
        for v in y.as_f32().unwrap() {
            assert!((v - 1.0).abs() < 1e-5, "{v}");
        }
    }

    #[test]
    fn matmul_f32_identity() {
        let w = Tensor::f32(
            vec![3, 3],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        );
        let x = Tensor::f32(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = cpu().matmul_q(&w, &x).unwrap();
        assert_eq!(y.shape, vec![2, 3]);
        assert_eq!(y.as_f32().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn matmul_pq2_0_finite() {
        let row = [0.5f32; 128];
        let mut bytes = vec![0u8; 34];
        bfi_gguf::quantize_row_pq2_0(&row, &mut bytes);
        let w = Tensor {
            shape: vec![1, 128],
            dtype: DType::PQ2_0,
            storage: Storage::Quant {
                ggml_id: bfi_gguf::TYPE_PQ2_0,
                bytes,
            },
        };
        let x = Tensor::f32(vec![1, 128], vec![1.0; 128]);
        let y = cpu().matmul_q(&w, &x).unwrap();
        let v = y.as_f32().unwrap();
        assert_eq!(v.len(), 1);
        assert!(v[0].is_finite(), "{}", v[0]);
    }

    #[test]
    fn matmul_ptq1_0_finite() {
        // Zero qs/qh + f16 d=2.0 (0x4000) dequantizes to -2.0 per elem.
        let mut bytes = vec![0u8; 28];
        bytes[26..28].copy_from_slice(&0x4000u16.to_le_bytes());
        let w = cpu()
            .upload_quant(&[1, 128], DType::PTQ1_0, &bytes)
            .unwrap();
        let x = Tensor::f32(vec![1, 128], vec![1.0; 128]);
        let y = cpu().matmul_q(&w, &x).unwrap();
        let v = y.as_f32().unwrap();
        assert_eq!(v.len(), 1);
        assert!(v[0].is_finite(), "{}", v[0]);
        assert!((v[0] + 256.0).abs() < 1e-2, "{}", v[0]);
    }

    #[test]
    fn alloc_f32_is_zeros_with_shape() {
        let t = cpu().alloc(&[2, 4], DType::F32).unwrap();
        assert_eq!(t.shape, vec![2, 4]);
        assert_eq!(t.dtype, DType::F32);
        assert_eq!(t.as_f32().unwrap(), &[0.0; 8]);
    }

    #[test]
    fn rope_preserves_l2() {
        let data = vec![3.0, 4.0, 1.0, -2.0];
        let mut x = Tensor::f32(vec![1, 1, 4], data.clone());
        let before = l2(&data);
        cpu().rope(&mut x, &[5], &rope_params(4)).unwrap();
        let after = l2(x.as_f32().unwrap());
        assert!((before - after).abs() < 1e-5, "{before} vs {after}");
    }

    #[test]
    fn softmax_zeros() {
        let mut x = Tensor::f32(vec![2], vec![0.0, 0.0]);
        cpu().softmax_inplace(&mut x).unwrap();
        let y = x.as_f32().unwrap();
        assert!((y[0] - 0.5).abs() < 1e-6);
        assert!((y[1] - 0.5).abs() < 1e-6);
        let z = cpu()
            .softmax(&Tensor::f32(vec![2], vec![0.0, 0.0]))
            .unwrap();
        assert!((z.as_f32().unwrap()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn silu_zero_and_one() {
        let y = cpu()
            .silu(&Tensor::f32(vec![3], vec![0.0, 1.0, -1.0]))
            .unwrap();
        let v = y.as_f32().unwrap();
        assert!((v[0] - 0.0).abs() < 1e-6);
        // silu(x) = x / (1 + exp(-x))
        let s1 = 1.0 / (1.0 + (-1.0f32).exp());
        assert!((v[1] - s1).abs() < 1e-6);
        let sm1 = -1.0 / (1.0 + (1.0f32).exp());
        assert!((v[2] - sm1).abs() < 1e-6);
        let gate = Tensor::f32(vec![2], vec![1.0, 0.0]);
        let up = Tensor::f32(vec![2], vec![2.0, 3.0]);
        let sw = cpu().swiglu(&gate, &up).unwrap();
        let sv = sw.as_f32().unwrap();
        assert!((sv[0] - s1 * 2.0).abs() < 1e-6);
        assert!((sv[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn attention_t1_eq_v() {
        let q = Tensor::f32(vec![1, 1, 2], vec![1.0, 0.0]);
        let k = Tensor::f32(vec![1, 1, 2], vec![0.0, 1.0]);
        let v = Tensor::f32(vec![1, 1, 2], vec![7.0, 9.0]);
        let o = cpu().attention(&q, &k, &v, true).unwrap();
        assert_eq!(o.as_f32().unwrap(), &[7.0, 9.0]);
    }

    #[test]
    fn add_mul_get_rows_smoke() {
        let b = cpu();
        let a = Tensor::f32(vec![2], vec![1.0, 2.0]);
        let c = Tensor::f32(vec![2], vec![3.0, 4.0]);
        let sum = b.add(&a, &c).unwrap();
        assert_eq!(sum.as_f32().unwrap(), &[4.0, 6.0]);
        let prod = b.mul(&a, &c).unwrap();
        assert_eq!(prod.as_f32().unwrap(), &[3.0, 8.0]);
        let w = Tensor::f32(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rows = b.get_rows(&w, &[2, 0]).unwrap();
        assert_eq!(rows.shape, vec![2, 2]);
        assert_eq!(rows.as_f32().unwrap(), &[5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn select_backend_is_cpu() {
        assert_eq!(select_backend().name(), "cpu");
        assert_eq!(select_backend_from(Some("cuda")).name(), "cpu");
        assert_eq!(select_backend_from(Some("cpu")).name(), "cpu");
    }

    #[test]
    fn upload_matches_upload_f32() {
        let b = cpu();
        let via_alias = b.upload(&[2], &[1.5, -0.5]).unwrap();
        let via_named = b.upload_f32(&[2], &[1.5, -0.5]).unwrap();
        assert_eq!(via_alias.as_f32().unwrap(), via_named.as_f32().unwrap());
        assert_eq!(via_alias.shape, vec![2]);
    }

    #[test]
    fn dtype_ggml_roundtrip() {
        for id in [
            bfi_gguf::TYPE_F32,
            bfi_gguf::TYPE_F16,
            bfi_gguf::TYPE_BF16,
            bfi_gguf::TYPE_Q4_0,
            bfi_gguf::TYPE_Q8_0,
            bfi_gguf::TYPE_Q4_K,
            bfi_gguf::TYPE_Q5_K,
            bfi_gguf::TYPE_Q6_K,
            bfi_gguf::TYPE_PQ2_0,
            bfi_gguf::TYPE_PTQ1_0,
        ] {
            let dt = DType::from_ggml(id).unwrap();
            assert_eq!(dt.ggml_id(), id);
        }
        assert!(DType::from_ggml(99).is_none());
    }

    #[test]
    fn rope_multi_none_matches_rope() {
        let data = vec![0.5, -1.25, 2.0, 0.25];
        let mut a = Tensor::f32(vec![1, 1, 4], data.clone());
        let mut b = Tensor::f32(vec![1, 1, 4], data);
        let p = rope_params(4);
        cpu().rope(&mut a, &[3], &p).unwrap();
        cpu().rope_multi(&mut b, &[3], &p).unwrap();
        let aa = a.as_f32().unwrap();
        let bb = b.as_f32().unwrap();
        for (x, y) in aa.iter().zip(bb) {
            assert!((x - y).abs() < 1e-6);
        }
    }
}
