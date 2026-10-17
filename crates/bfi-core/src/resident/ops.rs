//! Kernel launches for resident tensors. Every op only *enqueues* work; nothing
//! here waits on the GPU.

use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use bfi_gguf::{TYPE_F16, TYPE_F32, TYPE_Q1_0, TYPE_Q8_0};

use super::{DeviceTensor, Gpu, KvBuffer, Weight};
use crate::kernels::{elementwise_cubes, gemm, gemm_cubes, lanes, ops, rows};

type R = WgpuRuntime;

// Safety: every DeviceTensor / Weight records its exact element count at creation.
fn arr(t: &DeviceTensor) -> ArrayArg<R> {
    unsafe { ArrayArg::from_raw_parts(t.handle.clone(), t.len.max(1)) }
}

fn warr(w: &Weight) -> ArrayArg<R> {
    unsafe { ArrayArg::from_raw_parts(w.handle.clone(), w.words.max(1)) }
}

impl Gpu {
    /// `y[t, :] = W · x[t, :]` for `n_tokens` rows of `x` (`[n_tokens, n_in]`).
    pub fn matmul(&self, w: &Weight, x: &DeviceTensor, n_tokens: usize) -> DeviceTensor {
        debug_assert_eq!(x.len, n_tokens * w.n_in as usize);
        let y = self.empty_f32(n_tokens * w.n_out as usize);
        let (count, row_stride) = gemm_cubes(w.n_out, n_tokens as u32);
        let c = &self.client;
        macro_rules! launch {
            ($k:path) => {
                unsafe {
                    $k(
                        c,
                        count,
                        lanes(),
                        warr(w),
                        arr(x),
                        arr(&y),
                        w.n_out,
                        w.n_in,
                        n_tokens as u32,
                        row_stride,
                    )
                }
            };
        }
        match w.ggml_type {
            TYPE_Q1_0 => launch!(gemm::gemm_q1_0::launch_unchecked::<R>),
            TYPE_Q8_0 => launch!(gemm::gemm_q8_0::launch_unchecked::<R>),
            TYPE_F16 => launch!(gemm::gemm_f16::launch_unchecked::<R>),
            TYPE_F32 => launch!(gemm::gemm_f32::launch_unchecked::<R>),
            other => unreachable!("upload_weight rejects GGML type {other}"),
        };
        y
    }

    /// `out[t, :] = W[ids[t], :]` (W is `[n_vocab, n_embd]`, i.e. `n_out` rows of `n_in`).
    pub fn get_rows(&self, w: &Weight, ids: &DeviceTensor) -> DeviceTensor {
        let total = ids.len * w.n_in as usize;
        let out = self.empty_f32(total);
        let count = elementwise_cubes(total as u32);
        let c = &self.client;
        macro_rules! launch {
            ($k:path) => {
                unsafe { $k(c, count, lanes(), warr(w), arr(ids), arr(&out), w.n_in, total as u32) }
            };
        }
        match w.ggml_type {
            TYPE_Q1_0 => launch!(rows::rows_q1_0::launch_unchecked::<R>),
            TYPE_Q8_0 => launch!(rows::rows_q8_0::launch_unchecked::<R>),
            TYPE_F16 => launch!(rows::rows_f16::launch_unchecked::<R>),
            TYPE_F32 => launch!(rows::rows_f32::launch_unchecked::<R>),
            other => unreachable!("upload_weight rejects GGML type {other}"),
        };
        out
    }

    /// RMSNorm over `rows` rows of width `d`; `w` has `d` elements.
    pub fn rms_norm(&self, x: &DeviceTensor, w: &DeviceTensor, rows: usize, d: usize, eps: f32) -> DeviceTensor {
        let y = self.empty_f32(rows * d);
        unsafe {
            ops::rms_norm_rows::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(rows.max(1) as u32, 1, 1),
                lanes(),
                arr(x),
                arr(w),
                arr(&y),
                d as u32,
                eps,
            );
        }
        y
    }

    /// NEOX RoPE in place on `x` (`[n_tokens, n_heads, head_dim]`).
    /// `pos` holds `n_tokens` u32 positions; `inv_freq` holds `n_rot / 2` values.
    pub fn rope_neox(
        &self,
        x: &DeviceTensor,
        pos: &DeviceTensor,
        inv_freq: &DeviceTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) {
        let half = inv_freq.len;
        let total = (n_tokens * n_heads * half) as u32;
        unsafe {
            ops::rope_neox::launch_unchecked::<R>(
                &self.client,
                elementwise_cubes(total),
                lanes(),
                arr(x),
                arr(pos),
                arr(inv_freq),
                n_heads as u32,
                head_dim as u32,
                half as u32,
                total,
            );
        }
    }

    /// `dst[dst_off..dst_off + n] = src[src_off..src_off + n]`.
    pub fn copy(&self, src: &DeviceTensor, src_off: usize, dst: &DeviceTensor, dst_off: usize, n: usize) {
        unsafe {
            ops::copy_f32::launch_unchecked::<R>(
                &self.client,
                elementwise_cubes(n as u32),
                lanes(),
                arr(src),
                arr(dst),
                src_off as u32,
                dst_off as u32,
                n as u32,
            );
        }
    }

    /// Append `n` f32 elements of `src` to `kv` at logical element `dst_off`
    /// (f16 caches pack pairs; `dst_off` and `n` must then be even).
    pub fn write_kv(&self, src: &DeviceTensor, kv: &KvBuffer, dst_off: usize, n: usize) {
        if !kv.f16 {
            self.copy(src, 0, &kv.words, dst_off, n);
            return;
        }
        debug_assert!(dst_off % 2 == 0 && n % 2 == 0, "f16 KV rows must be even");
        let n_words = (n / 2) as u32;
        unsafe {
            ops::pack_f16::launch_unchecked::<R>(
                &self.client,
                elementwise_cubes(n_words),
                lanes(),
                arr(src),
                arr(&kv.words),
                (dst_off / 2) as u32,
                n_words,
            );
        }
    }

    /// Causal GQA attention of `q` (`[n_tokens, hq, d]`, positions
    /// `n_past..n_past + n_tokens`) over a KV cache (`[ctx, hkv, d]`, f32 or f16)
    /// that already holds rows `0..n_past + n_tokens`. Returns `[n_tokens, hq, d]`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        q: &DeviceTensor,
        k: &KvBuffer,
        v: &KvBuffer,
        n_tokens: usize,
        n_past: usize,
        hq: usize,
        hkv: usize,
        d: usize,
    ) -> DeviceTensor {
        let rows = n_tokens * hq;
        let s_len = n_past + n_tokens;
        let scores = self.empty_f32(rows * s_len);
        let scale = 1.0 / (d as f32).sqrt();
        let score_cubes = CubeCount::Static((s_len as u32).div_ceil(64).max(1), rows as u32, 1);
        macro_rules! scores {
            ($k:path) => {
                unsafe {
                    $k(
                        &self.client,
                        score_cubes,
                        lanes(),
                        arr(q),
                        arr(&k.words),
                        arr(&scores),
                        hq as u32,
                        hkv as u32,
                        d as u32,
                        s_len as u32,
                        n_past as u32,
                        scale,
                    )
                }
            };
        }
        if k.f16 {
            scores!(ops::attn_scores_f16::launch_unchecked::<R>);
        } else {
            scores!(ops::attn_scores::launch_unchecked::<R>);
        }
        unsafe {
            ops::softmax_rows::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(rows as u32, 1, 1),
                lanes(),
                arr(&scores),
                s_len as u32,
            );
        }
        let total = rows * d;
        let out = self.empty_f32(total);
        macro_rules! values {
            ($k:path) => {
                unsafe {
                    $k(
                        &self.client,
                        elementwise_cubes(total as u32),
                        lanes(),
                        arr(&scores),
                        arr(&v.words),
                        arr(&out),
                        hq as u32,
                        hkv as u32,
                        d as u32,
                        s_len as u32,
                        n_past as u32,
                        total as u32,
                    )
                }
            };
        }
        if v.f16 {
            values!(ops::attn_values_f16::launch_unchecked::<R>);
        } else {
            values!(ops::attn_values::launch_unchecked::<R>);
        }
        out
    }

    /// `silu(g) · up`.
    pub fn swiglu(&self, g: &DeviceTensor, up: &DeviceTensor) -> DeviceTensor {
        let y = self.empty_f32(g.len);
        unsafe {
            ops::swiglu::launch_unchecked::<R>(
                &self.client,
                elementwise_cubes(g.len as u32),
                lanes(),
                arr(g),
                arr(up),
                arr(&y),
                g.len as u32,
            );
        }
        y
    }

    /// `a += b`.
    pub fn add_inplace(&self, a: &DeviceTensor, b: &DeviceTensor) {
        unsafe {
            ops::add_inplace::launch_unchecked::<R>(
                &self.client,
                elementwise_cubes(a.len as u32),
                lanes(),
                arr(a),
                arr(b),
                a.len as u32,
            );
        }
    }
}
