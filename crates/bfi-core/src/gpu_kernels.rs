//! CubeCL fused GEMV kernels for PQ2_0 / PTQ1_0 (M1) and the host launcher
//! for Q1_0, whose `#[cube]` kernel lives in [`crate::kernels::gemm`] so the
//! native backend and the WebGPU resident path share one implementation.
//!
//! Compiled only when a GPU feature (`cuda`/`rocm`/`vulkan`/`metal`) is on.

use cubecl::prelude::*;

use crate::fused_gemv::{fused_gemv_pq2_0, fused_gemv_ptq1_0, fused_gemv_q1_0};
use crate::kernels::{gemm::gemm_q1_0, gemm_cubes, lanes};

#[cube(launch_unchecked)]
fn gemv_pq2(
    qs: &Array<u32>,
    scales: &Array<f32>,
    x: &Array<f32>,
    out: &mut Array<f32>,
    n_out: u32,
    n_blocks: u32,
) {
    // CubeCL 0.10: ABSOLUTE_POS and Array indices are usize.
    let n_out_us = usize::cast_from(n_out);
    let n_blocks_us = usize::cast_from(n_blocks);
    if ABSOLUTE_POS < n_out_us {
        let row = ABSOLUTE_POS;
        let mut acc = 0.0f32;
        for b in 0..n_blocks {
            let b_us = usize::cast_from(b);
            let d = scales[row * n_blocks_us + b_us];
            let qs_base = (row * n_blocks_us + b_us) * 8;
            let x_base = b_us * 128;
            for j in 0..128u32 {
                let j_us = usize::cast_from(j);
                let word = qs[qs_base + j_us / 16];
                let shift = (j % 16) * 2;
                let q = (word >> shift) & 3;
                let w = (f32::cast_from(q) - 1.0) * d;
                acc += w * x[x_base + j_us];
            }
        }
        out[row] = acc;
    }
}

#[cube]
fn pow3_of(n: u32) -> u32 {
    let mut p = 1u32;
    for _ in 0..n {
        p *= 3u32;
    }
    p
}

#[cube]
fn trit_of(byte: u32, n: u32) -> u32 {
    let wrapped = (byte * pow3_of(n)) & 255;
    (wrapped * 3) >> 8
}

#[cube(launch_unchecked)]
fn gemv_ptq1(
    qs: &Array<u32>,
    qh: &Array<u32>,
    scales: &Array<f32>,
    x: &Array<f32>,
    out: &mut Array<f32>,
    n_out: u32,
    n_blocks: u32,
) {
    let n_out_us = usize::cast_from(n_out);
    let n_blocks_us = usize::cast_from(n_blocks);
    if ABSOLUTE_POS < n_out_us {
        let row = ABSOLUTE_POS;
        let mut acc = 0.0f32;
        for b in 0..n_blocks {
            let b_us = usize::cast_from(b);
            let d = scales[row * n_blocks_us + b_us];
            let qs_base = (row * n_blocks_us + b_us) * 6;
            let x_base = b_us * 128;
            let mut i = 0;
            for n in 0..5u32 {
                for m in 0..16u32 {
                    let m_us = usize::cast_from(m);
                    let word = qs[qs_base + m_us / 4];
                    let byte = (word >> ((m % 4) * 8)) & 255;
                    let w = (f32::cast_from(trit_of(byte, n)) - 1.0) * d;
                    acc += w * x[x_base + i];
                    i += 1;
                }
            }
            for n in 0..5u32 {
                for m in 0..8u32 {
                    let m_us = usize::cast_from(m);
                    let word = qs[qs_base + 4 + m_us / 4];
                    let byte = (word >> ((m % 4) * 8)) & 255;
                    let w = (f32::cast_from(trit_of(byte, n)) - 1.0) * d;
                    acc += w * x[x_base + i];
                    i += 1;
                }
            }
            let qh_word = qh[row * n_blocks_us + b_us];
            for n in 0..4u32 {
                for h in 0..2u32 {
                    let byte = (qh_word >> (h * 8)) & 255;
                    let w = (f32::cast_from(trit_of(byte, n)) - 1.0) * d;
                    acc += w * x[x_base + i];
                    i += 1;
                }
            }
        }
        out[row] = acc;
    }
}

fn as_u8_u32(v: &[u32]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn as_u8_f32(v: &[f32]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn f32_from_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn launch_gemv_pq2<R: Runtime>(
    client: &ComputeClient<R>,
    qs: &[u32],
    scales: &[f32],
    x: &[f32],
    n_out: u32,
    n_blocks: u32,
) -> Vec<f32> {
    let qs_h = client.create_from_slice(as_u8_u32(qs));
    let sc_h = client.create_from_slice(as_u8_f32(scales));
    let x_h = client.create_from_slice(as_u8_f32(x));
    let n = n_out as usize;
    let out_h = client.empty(n * 4);
    let cube_dim = 64u32;
    let cubes = (n_out + cube_dim - 1) / cube_dim;
    unsafe {
        gemv_pq2::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes.max(1), 1, 1),
            CubeDim::new_1d(cube_dim),
            ArrayArg::from_raw_parts(qs_h, qs.len()),
            ArrayArg::from_raw_parts(sc_h, scales.len()),
            ArrayArg::from_raw_parts(x_h, x.len()),
            ArrayArg::from_raw_parts(out_h.clone(), n),
            n_out,
            n_blocks,
        );
    }
    let bytes = client.read_one_unchecked(out_h);
    f32_from_bytes(&bytes)[..n].to_vec()
}

pub fn launch_gemv_ptq1<R: Runtime>(
    client: &ComputeClient<R>,
    qs: &[u32],
    qh: &[u32],
    scales: &[f32],
    x: &[f32],
    n_out: u32,
    n_blocks: u32,
) -> Vec<f32> {
    let qs_h = client.create_from_slice(as_u8_u32(qs));
    let qh_h = client.create_from_slice(as_u8_u32(qh));
    let sc_h = client.create_from_slice(as_u8_f32(scales));
    let x_h = client.create_from_slice(as_u8_f32(x));
    let n = n_out as usize;
    let out_h = client.empty(n * 4);
    let cube_dim = 64u32;
    let cubes = (n_out + cube_dim - 1) / cube_dim;
    unsafe {
        gemv_ptq1::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes.max(1), 1, 1),
            CubeDim::new_1d(cube_dim),
            ArrayArg::from_raw_parts(qs_h, qs.len()),
            ArrayArg::from_raw_parts(qh_h, qh.len()),
            ArrayArg::from_raw_parts(sc_h, scales.len()),
            ArrayArg::from_raw_parts(x_h, x.len()),
            ArrayArg::from_raw_parts(out_h.clone(), n),
            n_out,
            n_blocks,
        );
    }
    let bytes = client.read_one_unchecked(out_h);
    f32_from_bytes(&bytes)[..n].to_vec()
}

/// Raw GGUF bytes padded to whole `u32` words (kernels read words, unpack bytes).
pub fn padded_words(bytes: &[u8]) -> Vec<u8> {
    let mut v = bytes.to_vec();
    v.resize(bytes.len().div_ceil(4) * 4, 0);
    v
}

/// Q1_0 GEMV: weights uploaded exactly as packed (18 B / 128 weights).
pub fn launch_gemv_q1<R: Runtime>(
    client: &ComputeClient<R>,
    weights: &[u8],
    x: &[f32],
    n_out: u32,
    n_in: u32,
) -> Vec<f32> {
    let words = padded_words(weights);
    let n_words = words.len() / 4;
    let w_h = client.create_from_slice(&words);
    let x_h = client.create_from_slice(as_u8_f32(x));
    let n = n_out as usize;
    let out_h = client.empty(n * 4);
    let (count, row_stride) = gemm_cubes(n_out, 1);
    unsafe {
        gemm_q1_0::launch_unchecked::<R>(
            client,
            count,
            lanes(),
            ArrayArg::from_raw_parts(w_h, n_words),
            ArrayArg::from_raw_parts(x_h, x.len()),
            ArrayArg::from_raw_parts(out_h.clone(), n),
            n_out,
            n_in,
            1u32,
            row_stride,
        );
    }
    let bytes = client.read_one_unchecked(out_h);
    f32_from_bytes(&bytes)[..n].to_vec()
}

pub fn gemv_q1_or_cpu<R: Runtime>(
    client: &ComputeClient<R>,
    weights: &[u8],
    x: &[f32],
    n_out: usize,
    n_in: usize,
) -> anyhow::Result<Vec<f32>> {
    if n_in % 128 != 0 || weights.len() != n_out * (n_in / 128) * 18 {
        anyhow::bail!("q1_0 gemv: bytes {} do not match {n_out}x{n_in}", weights.len());
    }
    let launched = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        launch_gemv_q1::<R>(client, weights, x, n_out as u32, n_in as u32)
    }));
    match launched {
        Ok(y) => Ok(y),
        Err(_) => fused_gemv_q1_0(weights, x, n_out, n_in),
    }
}

/// Host packing + kernel launch. Falls back to the portable fused GEMV if the
/// launch panics (missing compiler feature, device reset).
pub fn gemv_pq2_or_cpu<R: Runtime>(
    client: &ComputeClient<R>,
    weights: &[u8],
    x: &[f32],
    n_out: usize,
    n_in: usize,
) -> anyhow::Result<Vec<f32>> {
    let (qs, scales) = pack_pq2(weights, n_out, n_in)?;
    let n_blocks = (n_in / 128) as u32;
    let launched = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        launch_gemv_pq2::<R>(client, &qs, &scales, x, n_out as u32, n_blocks)
    }));
    match launched {
        Ok(y) => Ok(y),
        Err(_) => fused_gemv_pq2_0(weights, x, n_out, n_in),
    }
}

pub fn gemv_ptq1_or_cpu<R: Runtime>(
    client: &ComputeClient<R>,
    weights: &[u8],
    x: &[f32],
    n_out: usize,
    n_in: usize,
) -> anyhow::Result<Vec<f32>> {
    let (qs, qh, scales) = pack_ptq1(weights, n_out, n_in)?;
    let n_blocks = (n_in / 128) as u32;
    let launched = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        launch_gemv_ptq1::<R>(client, &qs, &qh, &scales, x, n_out as u32, n_blocks)
    }));
    match launched {
        Ok(y) => Ok(y),
        Err(_) => fused_gemv_ptq1_0(weights, x, n_out, n_in),
    }
}

pub fn pack_pq2(
    bytes: &[u8],
    n_out: usize,
    n_in: usize,
) -> anyhow::Result<(Vec<u32>, Vec<f32>)> {
    if n_in % 128 != 0 {
        anyhow::bail!("pq2 pack: n_in {n_in} % 128 != 0");
    }
    let n_blocks = n_in / 128;
    let row_bytes = n_blocks * 34;
    if bytes.len() != n_out * row_bytes {
        anyhow::bail!(
            "pq2 pack: bytes {} != {n_out}*{row_bytes}",
            bytes.len()
        );
    }
    let mut qs = Vec::with_capacity(n_out * n_blocks * 8);
    let mut scales = Vec::with_capacity(n_out * n_blocks);
    for row in 0..n_out {
        for b in 0..n_blocks {
            let off = (row * n_blocks + b) * 34;
            let bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
            scales.push(bfi_gguf::f16_to_f32(bits));
            for w in 0..8 {
                let i = off + 2 + w * 4;
                qs.push(u32::from_le_bytes([
                    bytes[i],
                    bytes[i + 1],
                    bytes[i + 2],
                    bytes[i + 3],
                ]));
            }
        }
    }
    Ok((qs, scales))
}

pub fn pack_ptq1(
    bytes: &[u8],
    n_out: usize,
    n_in: usize,
) -> anyhow::Result<(Vec<u32>, Vec<u32>, Vec<f32>)> {
    if n_in % 128 != 0 {
        anyhow::bail!("ptq1 pack: n_in {n_in} % 128 != 0");
    }
    let n_blocks = n_in / 128;
    let row_bytes = n_blocks * 28;
    if bytes.len() != n_out * row_bytes {
        anyhow::bail!(
            "ptq1 pack: bytes {} != {n_out}*{row_bytes}",
            bytes.len()
        );
    }
    let mut qs = Vec::with_capacity(n_out * n_blocks * 6);
    let mut qh = Vec::with_capacity(n_out * n_blocks);
    let mut scales = Vec::with_capacity(n_out * n_blocks);
    for row in 0..n_out {
        for b in 0..n_blocks {
            let off = (row * n_blocks + b) * 28;
            for w in 0..6 {
                let i = off + w * 4;
                qs.push(u32::from_le_bytes([
                    bytes[i],
                    bytes[i + 1],
                    bytes[i + 2],
                    bytes[i + 3],
                ]));
            }
            qh.push(u32::from_le_bytes([bytes[off + 24], bytes[off + 25], 0, 0]));
            let bits = u16::from_le_bytes([bytes[off + 26], bytes[off + 27]]);
            scales.push(bfi_gguf::f16_to_f32(bits));
        }
    }
    Ok((qs, qh, scales))
}
