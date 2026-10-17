//! f32 activation kernels for a GPU-resident decoder layer.
//!
//! Row kernels (RMSNorm, softmax) use one 64-lane workgroup per row; the rest
//! are one lane per element. Must match `CpuBackend` (the reference) within
//! f32 rounding.

use cubecl::prelude::*;

// Masked scores use -3.0e38 instead of -inf, written `f32::new(-3.0e38)`: a
// mutable local initialised from a literal *expression* stays a host f32 in
// CubeCL 0.10 and cannot later take a runtime value. Instead of -inf: `exp(-3e38 - max)` underflows to
// exactly 0 without inf arithmetic (some WebGPU compilers fold -inf badly).

#[cube]
fn reduce_sum64(smem: &mut SharedMemory<f32>, u: u32) {
    let i = usize::cast_from(u);
    if u < 32u32 {
        smem[i] += smem[i + 32];
    }
    sync_cube();
    if u < 16u32 {
        smem[i] += smem[i + 16];
    }
    sync_cube();
    if u < 8u32 {
        smem[i] += smem[i + 8];
    }
    sync_cube();
    if u < 4u32 {
        smem[i] += smem[i + 4];
    }
    sync_cube();
    if u < 2u32 {
        smem[i] += smem[i + 2];
    }
    sync_cube();
    if u < 1u32 {
        smem[i] += smem[i + 1];
    }
    sync_cube();
}

#[cube]
fn max_into(smem: &mut SharedMemory<f32>, i: usize, j: usize) {
    let b = smem[j];
    if b > smem[i] {
        smem[i] = b;
    }
}

#[cube]
fn reduce_max64(smem: &mut SharedMemory<f32>, u: u32) {
    let i = usize::cast_from(u);
    if u < 32u32 {
        max_into(smem, i, i + 32);
    }
    sync_cube();
    if u < 16u32 {
        max_into(smem, i, i + 16);
    }
    sync_cube();
    if u < 8u32 {
        max_into(smem, i, i + 8);
    }
    sync_cube();
    if u < 4u32 {
        max_into(smem, i, i + 4);
    }
    sync_cube();
    if u < 2u32 {
        max_into(smem, i, i + 2);
    }
    sync_cube();
    if u < 1u32 {
        max_into(smem, i, i + 1);
    }
    sync_cube();
}

/// `y[r, :] = x[r, :] / sqrt(mean(x[r, :]²) + eps) * w`. One workgroup per row.
#[cube(launch_unchecked)]
pub fn rms_norm_rows(x: &Array<f32>, w: &Array<f32>, y: &mut Array<f32>, d: u32, eps: f32) {
    let mut smem = SharedMemory::<f32>::new(64usize);
    let u = UNIT_POS;
    let base = CUBE_POS_X * d;
    let iters = (d + 63u32 - u) / 64u32;
    let mut ss = 0.0f32;
    for k in 0..iters {
        let v = x[usize::cast_from(base + u + k * 64u32)];
        ss += v * v;
    }
    smem[usize::cast_from(u)] = ss;
    sync_cube();
    reduce_sum64(&mut smem, u);
    let scale = 1.0f32 / f32::sqrt(smem[0] / f32::cast_from(d) + eps);
    for k in 0..iters {
        let e = u + k * 64u32;
        let i = usize::cast_from(base + e);
        y[i] = x[i] * scale * w[usize::cast_from(e)];
    }
}

/// NEOX RoPE in place on `x[t, h, :]`: dim `i` pairs with `i + half`,
/// `theta = pos[t] · inv_freq[i]` for `i < half = n_rot / 2`.
#[cube(launch_unchecked)]
pub fn rope_neox(
    x: &mut Array<f32>,
    pos: &Array<u32>,
    inv_freq: &Array<f32>,
    n_heads: u32,
    head_dim: u32,
    half: u32,
    total: u32,
) {
    let gid = CUBE_POS_X * 64u32 + UNIT_POS;
    if gid < total {
        let i = gid % half;
        let th = gid / half;
        let t = th / n_heads;
        let base = th * head_dim;
        let theta = f32::cast_from(pos[usize::cast_from(t)]) * inv_freq[usize::cast_from(i)];
        let c = f32::cos(theta);
        let s = f32::sin(theta);
        let a = usize::cast_from(base + i);
        let b = usize::cast_from(base + i + half);
        let x0 = x[a];
        let x1 = x[b];
        x[a] = x0 * c - x1 * s;
        x[b] = x0 * s + x1 * c;
    }
}

/// Causal scores: `sc[r, s] = scale · q[r] · k[s, kv(r)]` for `s ≤ n_past + t(r)`,
/// `-3e38` (masked) after. Rows `r = t · hq + h`; grid `(ceil(s_len/64), rows)`.
/// KV cache layout `[ctx, hkv, d]`.
#[cube(launch_unchecked)]
pub fn attn_scores(
    q: &Array<f32>,
    kc: &Array<f32>,
    sc: &mut Array<f32>,
    hq: u32,
    hkv: u32,
    d: u32,
    s_len: u32,
    n_past: u32,
    scale: f32,
) {
    let s = CUBE_POS_X * 64u32 + UNIT_POS;
    let r = CUBE_POS_Y;
    if s < s_len {
        let t = r / hq;
        let kvh = (r % hq) / (hq / hkv);
        let mut v = f32::new(-3.0e38);
        if s <= n_past + t {
            let qb = r * d;
            let kb = (s * hkv + kvh) * d;
            let mut dot = 0.0f32;
            for i in 0..d {
                dot += q[usize::cast_from(qb + i)] * kc[usize::cast_from(kb + i)];
            }
            v = dot * scale;
        }
        sc[usize::cast_from(r * s_len + s)] = v;
    }
}

/// Row softmax over `s_len` entries in place. One workgroup per row.
#[cube(launch_unchecked)]
pub fn softmax_rows(sc: &mut Array<f32>, s_len: u32) {
    let mut smem = SharedMemory::<f32>::new(64usize);
    let u = UNIT_POS;
    let base = CUBE_POS_X * s_len;
    let iters = (s_len + 63u32 - u) / 64u32;
    let mut m = f32::new(-3.0e38);
    for k in 0..iters {
        let v = sc[usize::cast_from(base + u + k * 64u32)];
        if v > m {
            m = v;
        }
    }
    smem[usize::cast_from(u)] = m;
    sync_cube();
    reduce_max64(&mut smem, u);
    let mx = smem[0];
    sync_cube();
    let mut sum = 0.0f32;
    for k in 0..iters {
        let i = usize::cast_from(base + u + k * 64u32);
        let e = f32::exp(sc[i] - mx);
        sc[i] = e;
        sum += e;
    }
    smem[usize::cast_from(u)] = sum;
    sync_cube();
    reduce_sum64(&mut smem, u);
    let inv = 1.0f32 / smem[0];
    for k in 0..iters {
        let i = usize::cast_from(base + u + k * 64u32);
        sc[i] = sc[i] * inv;
    }
}

/// `out[r, i] = Σ_{s ≤ n_past + t(r)} p[r, s] · v[s, kv(r), i]`. Never touches
/// cache rows past the causal limit (they may be uninitialised).
#[cube(launch_unchecked)]
pub fn attn_values(
    sc: &Array<f32>,
    vc: &Array<f32>,
    out: &mut Array<f32>,
    hq: u32,
    hkv: u32,
    d: u32,
    s_len: u32,
    n_past: u32,
    total: u32,
) {
    let gid = CUBE_POS_X * 64u32 + UNIT_POS;
    if gid < total {
        let i = gid % d;
        let r = gid / d;
        let t = r / hq;
        let kvh = (r % hq) / (hq / hkv);
        let n_valid = n_past + t + 1u32;
        let pb = r * s_len;
        let mut acc = 0.0f32;
        for s in 0..n_valid {
            acc += sc[usize::cast_from(pb + s)] * vc[usize::cast_from((s * hkv + kvh) * d + i)];
        }
        out[usize::cast_from(gid)] = acc;
    }
}

/// `y = silu(g) · up`.
#[cube(launch_unchecked)]
pub fn swiglu(g: &Array<f32>, up: &Array<f32>, y: &mut Array<f32>, n: u32) {
    let gid = CUBE_POS_X * 64u32 + UNIT_POS;
    if gid < n {
        let i = usize::cast_from(gid);
        let v = g[i];
        y[i] = v / (1.0f32 + f32::exp(0.0f32 - v)) * up[i];
    }
}

/// `a += b`.
#[cube(launch_unchecked)]
pub fn add_inplace(a: &mut Array<f32>, b: &Array<f32>, n: u32) {
    let gid = CUBE_POS_X * 64u32 + UNIT_POS;
    if gid < n {
        let i = usize::cast_from(gid);
        a[i] = a[i] + b[i];
    }
}

/// `dst[dst_off + i] = src[src_off + i]` for `i < n` (KV append, last-row slice).
#[cube(launch_unchecked)]
pub fn copy_f32(src: &Array<f32>, dst: &mut Array<f32>, src_off: u32, dst_off: u32, n: u32) {
    let gid = CUBE_POS_X * 64u32 + UNIT_POS;
    if gid < n {
        dst[usize::cast_from(dst_off + gid)] = src[usize::cast_from(src_off + gid)];
    }
}
