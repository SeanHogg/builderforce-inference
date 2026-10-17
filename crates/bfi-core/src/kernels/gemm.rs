//! Dequant + GEMM/GEMV over packed weights: `y[t, row] = Σ_e W[row, e] · x[t, e]`.
//!
//! Geometry (see [`super::gemm_cubes`]): one 64-lane workgroup per output row,
//! up to [`super::TOKEN_TILE`] tokens per workgroup, so a weight row is read once
//! per 8 tokens during prefill and once per token at decode. Lanes stride the row
//! so neighbouring lanes read neighbouring bytes; a shared-memory tree sums lanes.
//!
//! Barriers sit only in uniform control flow (top level of the kernel), which
//! WGSL's uniformity analysis requires.

use cubecl::prelude::*;

use super::bytes::{byte_at, half_at, weight_f16, weight_f32, weight_q8_0};

/// Tokens in this workgroup's tile.
#[cube]
fn tile_len(tok0: u32, n_tokens: u32) -> u32 {
    let mut n = 0u32;
    if tok0 < n_tokens {
        n = n_tokens - tok0;
        if n > 8u32 {
            n = 8u32;
        }
    }
    n
}

/// Sum the 64 lane partials of each of the 8 tile slots into slot `t * 64`.
#[cube]
fn reduce_tile(smem: &mut SharedMemory<f32>, u: u32) {
    if u < 32u32 {
        for t in 0..8u32 {
            let i = usize::cast_from(t * 64u32 + u);
            smem[i] += smem[i + 32];
        }
    }
    sync_cube();
    if u < 16u32 {
        for t in 0..8u32 {
            let i = usize::cast_from(t * 64u32 + u);
            smem[i] += smem[i + 16];
        }
    }
    sync_cube();
    if u < 8u32 {
        for t in 0..8u32 {
            let i = usize::cast_from(t * 64u32 + u);
            smem[i] += smem[i + 8];
        }
    }
    sync_cube();
    if u < 4u32 {
        for t in 0..8u32 {
            let i = usize::cast_from(t * 64u32 + u);
            smem[i] += smem[i + 4];
        }
    }
    sync_cube();
    if u < 2u32 {
        for t in 0..8u32 {
            let i = usize::cast_from(t * 64u32 + u);
            smem[i] += smem[i + 2];
        }
    }
    sync_cube();
    if u < 1u32 {
        for t in 0..8u32 {
            let i = usize::cast_from(t * 64u32 + u);
            smem[i] += smem[i + 1];
        }
    }
    sync_cube();
}

#[cube]
fn write_tile(
    smem: &SharedMemory<f32>,
    y: &mut Array<f32>,
    u: u32,
    row: u32,
    n_out: u32,
    tok0: u32,
    ntile: u32,
) {
    if u < ntile {
        if row < n_out {
            y[usize::cast_from((tok0 + u) * n_out + row)] = smem[usize::cast_from(u * 64u32)];
        }
    }
}

/// 1-bit Bonsai `Q1_0`. Each lane takes whole sign bytes (8 weights, one scale):
/// `acc += d · (Σ_{bit set} x − Σ_{bit clear} x)`.
#[cube(launch_unchecked)]
pub fn gemm_q1_0(
    w: &Array<u32>,
    x: &Array<f32>,
    y: &mut Array<f32>,
    n_out: u32,
    n_in: u32,
    n_tokens: u32,
    row_stride: u32,
) {
    let mut smem = SharedMemory::<f32>::new(512usize);
    let u = UNIT_POS;
    let row = CUBE_POS_Z * row_stride + CUBE_POS_X;
    let tok0 = CUBE_POS_Y * 8u32;
    let ntile = tile_len(tok0, n_tokens);
    let row_bytes = (n_in / 128u32) * 18u32;
    let n_sign_bytes = n_in / 8u32;
    let iters = (n_sign_bytes + 63u32 - u) / 64u32;
    for t in 0..8u32 {
        let mut acc = 0.0f32;
        if row < n_out {
            if t < ntile {
                let xb = (tok0 + t) * n_in;
                for k in 0..iters {
                    let b = u + k * 64u32;
                    let base = row * row_bytes + (b / 16u32) * 18u32;
                    let d = half_at(w, base);
                    let bits = byte_at(w, base + 2u32 + b % 16u32);
                    let xo = xb + b * 8u32;
                    let mut s = 0.0f32;
                    for j in 0..8u32 {
                        let v = x[usize::cast_from(xo + j)];
                        if ((bits >> j) & 1u32) == 1u32 {
                            s += v;
                        } else {
                            s -= v;
                        }
                    }
                    acc += d * s;
                }
            }
        }
        smem[usize::cast_from(t * 64u32 + u)] = acc;
    }
    sync_cube();
    reduce_tile(&mut smem, u);
    write_tile(&smem, y, u, row, n_out, tok0, ntile);
}

#[cube(launch_unchecked)]
pub fn gemm_f32(
    w: &Array<u32>,
    x: &Array<f32>,
    y: &mut Array<f32>,
    n_out: u32,
    n_in: u32,
    n_tokens: u32,
    row_stride: u32,
) {
    let mut smem = SharedMemory::<f32>::new(512usize);
    let u = UNIT_POS;
    let row = CUBE_POS_Z * row_stride + CUBE_POS_X;
    let tok0 = CUBE_POS_Y * 8u32;
    let ntile = tile_len(tok0, n_tokens);
    let iters = (n_in + 63u32 - u) / 64u32;
    for t in 0..8u32 {
        let mut acc = 0.0f32;
        if row < n_out {
            if t < ntile {
                let xb = (tok0 + t) * n_in;
                for k in 0..iters {
                    let e = u + k * 64u32;
                    acc += weight_f32(w, row * n_in + e) * x[usize::cast_from(xb + e)];
                }
            }
        }
        smem[usize::cast_from(t * 64u32 + u)] = acc;
    }
    sync_cube();
    reduce_tile(&mut smem, u);
    write_tile(&smem, y, u, row, n_out, tok0, ntile);
}

#[cube(launch_unchecked)]
pub fn gemm_f16(
    w: &Array<u32>,
    x: &Array<f32>,
    y: &mut Array<f32>,
    n_out: u32,
    n_in: u32,
    n_tokens: u32,
    row_stride: u32,
) {
    let mut smem = SharedMemory::<f32>::new(512usize);
    let u = UNIT_POS;
    let row = CUBE_POS_Z * row_stride + CUBE_POS_X;
    let tok0 = CUBE_POS_Y * 8u32;
    let ntile = tile_len(tok0, n_tokens);
    let iters = (n_in + 63u32 - u) / 64u32;
    for t in 0..8u32 {
        let mut acc = 0.0f32;
        if row < n_out {
            if t < ntile {
                let xb = (tok0 + t) * n_in;
                for k in 0..iters {
                    let e = u + k * 64u32;
                    acc += weight_f16(w, row * n_in + e) * x[usize::cast_from(xb + e)];
                }
            }
        }
        smem[usize::cast_from(t * 64u32 + u)] = acc;
    }
    sync_cube();
    reduce_tile(&mut smem, u);
    write_tile(&smem, y, u, row, n_out, tok0, ntile);
}

#[cube(launch_unchecked)]
pub fn gemm_q8_0(
    w: &Array<u32>,
    x: &Array<f32>,
    y: &mut Array<f32>,
    n_out: u32,
    n_in: u32,
    n_tokens: u32,
    row_stride: u32,
) {
    let mut smem = SharedMemory::<f32>::new(512usize);
    let u = UNIT_POS;
    let row = CUBE_POS_Z * row_stride + CUBE_POS_X;
    let tok0 = CUBE_POS_Y * 8u32;
    let ntile = tile_len(tok0, n_tokens);
    let iters = (n_in + 63u32 - u) / 64u32;
    for t in 0..8u32 {
        let mut acc = 0.0f32;
        if row < n_out {
            if t < ntile {
                let xb = (tok0 + t) * n_in;
                for k in 0..iters {
                    let e = u + k * 64u32;
                    acc += weight_q8_0(w, row, e, n_in) * x[usize::cast_from(xb + e)];
                }
            }
        }
        smem[usize::cast_from(t * 64u32 + u)] = acc;
    }
    sync_cube();
    reduce_tile(&mut smem, u);
    write_tile(&smem, y, u, row, n_out, tok0, ntile);
}
