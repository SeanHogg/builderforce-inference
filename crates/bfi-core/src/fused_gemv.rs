//! Portable fused dequant+GEMV for PQ2_0 / PTQ1_0.
//!
//! This is the CPU reference of the M1 CubeCL kernel: one call, no materialised
//! dequant buffer handed back to the caller. The GPU path in `gpu.rs` must match
//! this to max relative abs err ≤ 1e-3.

use anyhow::{bail, Context, Result};
use bfi_gguf::{lookup_type, vec_dot_pq2_0, vec_dot_ptq1_0, TYPE_PQ2_0, TYPE_PTQ1_0};
use rayon::prelude::*;

pub fn fused_gemv_pq2_0(
    weights: &[u8],
    x: &[f32],
    n_out: usize,
    n_in: usize,
) -> Result<Vec<f32>> {
    gemv_typed(TYPE_PQ2_0, weights, x, n_out, n_in, vec_dot_pq2_0)
}

pub fn fused_gemv_ptq1_0(
    weights: &[u8],
    x: &[f32],
    n_out: usize,
    n_in: usize,
) -> Result<Vec<f32>> {
    gemv_typed(TYPE_PTQ1_0, weights, x, n_out, n_in, vec_dot_ptq1_0)
}

fn gemv_typed(
    type_id: u32,
    weights: &[u8],
    x: &[f32],
    n_out: usize,
    n_in: usize,
    dot: fn(&[u8], &[f32]) -> f32,
) -> Result<Vec<f32>> {
    let ty = lookup_type(type_id).context("unknown ggml type")?;
    if ty.block_elems == 0 || n_in % ty.block_elems != 0 {
        bail!(
            "fused gemv: n_in {n_in} is not a multiple of {} block {}",
            ty.name,
            ty.block_elems
        );
    }
    if x.len() != n_in {
        bail!("fused gemv: x len {} != n_in {n_in}", x.len());
    }
    let n_blocks = n_in / ty.block_elems;
    let row_bytes = n_blocks * ty.block_bytes;
    if weights.len() != n_out * row_bytes {
        bail!(
            "fused gemv: weights {} != n_out {n_out} * row {row_bytes}",
            weights.len()
        );
    }
    let mut out = vec![0.0f32; n_out];
    out.par_iter_mut().enumerate().for_each(|(o, dst)| {
        let row = &weights[o * row_bytes..(o + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..n_blocks {
            let blk = &row[b * ty.block_bytes..(b + 1) * ty.block_bytes];
            let xs = &x[b * ty.block_elems..(b + 1) * ty.block_elems];
            acc += dot(blk, xs);
        }
        *dst = acc;
    });
    Ok(out)
}

/// `max_i |got - ref| / max(1e-6, max_i |ref|)`. PRD M1: ≤ 1e-3.
pub fn max_rel_err(got: &[f32], reference: &[f32]) -> f32 {
    if got.len() != reference.len() || got.is_empty() {
        return f32::INFINITY;
    }
    let scale = reference.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    got.iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
        / scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use bfi_gguf::{dequant_tensor, quantize_row_pq2_0};

    fn dequant_dot(ty_id: u32, bytes: &[u8], x: &[f32], n_out: usize, n_in: usize) -> Vec<f32> {
        let ty = lookup_type(ty_id).unwrap();
        let mut w = vec![0.0f32; n_out * n_in];
        dequant_tensor(ty, bytes, &mut w).unwrap();
        (0..n_out)
            .map(|o| {
                w[o * n_in..(o + 1) * n_in]
                    .iter()
                    .zip(x.iter())
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect()
    }

    #[test]
    fn fused_pq2_matches_dequant_dot() {
        let n_out = 4;
        let n_in = 128;
        let mut bytes = Vec::with_capacity(n_out * 34);
        for o in 0..n_out {
            let mut row = vec![0.25f32 * (o as f32 + 1.0); 128];
            row[o] = 1.5;
            let mut block = vec![0u8; 34];
            quantize_row_pq2_0(&row, &mut block);
            bytes.extend_from_slice(&block);
        }
        let x: Vec<f32> = (0..n_in).map(|i| 0.01 * (i as f32 + 1.0)).collect();
        let y_fused = fused_gemv_pq2_0(&bytes, &x, n_out, n_in).unwrap();
        let y_ref = dequant_dot(TYPE_PQ2_0, &bytes, &x, n_out, n_in);
        let err = max_rel_err(&y_fused, &y_ref);
        assert!(err <= 1e-3, "rel err {err} fused={y_fused:?} ref={y_ref:?}");
    }

    #[test]
    fn fused_ptq1_matches_dequant_dot() {
        let n_out = 4;
        let n_in = 128;
        let mut bytes = Vec::with_capacity(n_out * 28);
        for o in 0..n_out {
            let mut block = vec![0u8; 28];
            // d = 2.0 as f16 (0x4000); zero qs/qh → every weight is -d.
            let d_bits = if o % 2 == 0 { 0x4000u16 } else { 0x3c00u16 };
            block[26..28].copy_from_slice(&d_bits.to_le_bytes());
            bytes.extend_from_slice(&block);
        }
        let x = vec![1.0f32; n_in];
        let y_fused = fused_gemv_ptq1_0(&bytes, &x, n_out, n_in).unwrap();
        let y_ref = dequant_dot(TYPE_PTQ1_0, &bytes, &x, n_out, n_in);
        let err = max_rel_err(&y_fused, &y_ref);
        assert!(err <= 1e-3, "rel err {err} fused={y_fused:?} ref={y_ref:?}");
    }
}
