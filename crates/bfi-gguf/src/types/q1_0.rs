//! Q1_0 / `block_q1_0`: 128 sign bits + one f16 scale = 18 bytes (1.125 bpw).
//!
//! PrismML 1-bit Bonsai (`prism-ml/Bonsai-{1.7B,4B,8B}-gguf`, `Bonsai-<N>-Q1_0.gguf`).
//!
//! Source of truth: `PrismML-Eng/llama.cpp`, branch `prism`, commit
//! `0082ac24a89ae62132041b40e2c0170e278a98f4` (fetched 2026-10-17):
//! - `ggml/include/ggml.h`: `GGML_TYPE_Q1_0 = 41` (upstream id slot; `Q2_0 = 42`).
//! - `ggml/src/ggml-common.h`: `#define QK1_0 128`,
//!   `typedef struct { ggml_half d; uint8_t qs[QK1_0 / 8]; } block_q1_0;`
//!   with `static_assert(sizeof(block_q1_0) == sizeof(ggml_half) + QK1_0 / 8)`.
//! - `ggml/src/ggml-quants.c`: `dequantize_row_q1_0` — bit `j` is
//!   `(qs[j / 8] >> (j % 8)) & 1` (LSB first); set ⇒ `+d`, clear ⇒ `-d`.
//!   `quantize_row_q1_0_ref` stores `d = mean(|x|)` and sets the bit when `x >= 0`.
//!
//! Ported from that fork. MIT:
//!
//! Copyright (c) 2023-2026 The ggml authors.
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//! The above copyright notice and this permission notice shall be included in
//! all copies or substantial portions of the Software.

use crate::f16::{f32_to_f16, read_f16_le};

/// Weights per block (`QK1_0`).
pub const QK1_0: usize = 128;
/// Bytes per block: f16 scale + 16 bytes of sign bits.
pub const BLOCK_Q1_0_BYTES: usize = 2 + QK1_0 / 8;

/// `dequantize_row_q1_0` for one block.
pub fn dequantize_row_q1_0(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), BLOCK_Q1_0_BYTES);
    debug_assert_eq!(out.len(), QK1_0);
    let d = read_f16_le(&block[0..2]);
    let qs = &block[2..BLOCK_Q1_0_BYTES];
    for j in 0..QK1_0 {
        let bit = (qs[j / 8] >> (j % 8)) & 1;
        out[j] = if bit != 0 { d } else { -d };
    }
}

/// Dot of one block with 128 activations, without materialising the block:
/// `d * (Σ_{bit set} y - Σ_{bit clear} y)`.
pub fn vec_dot_q1_0(block: &[u8], y: &[f32]) -> f32 {
    debug_assert_eq!(block.len(), BLOCK_Q1_0_BYTES);
    let d = read_f16_le(&block[0..2]);
    let qs = &block[2..BLOCK_Q1_0_BYTES];
    let mut s = 0.0f32;
    for (b, &byte) in qs.iter().enumerate() {
        let ys = &y[b * 8..b * 8 + 8];
        for (k, &v) in ys.iter().enumerate() {
            if (byte >> k) & 1 != 0 {
                s += v;
            } else {
                s -= v;
            }
        }
    }
    s * d
}

/// `quantize_row_q1_0_ref` for one block. Tests and fixtures only; loaders never encode.
pub fn quantize_row_q1_0(x: &[f32], block: &mut [u8]) {
    assert_eq!(x.len(), QK1_0);
    assert_eq!(block.len(), BLOCK_Q1_0_BYTES);
    let d = x.iter().map(|v| v.abs()).sum::<f32>() / QK1_0 as f32;
    block[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
    let qs = &mut block[2..BLOCK_Q1_0_BYTES];
    qs.fill(0);
    for (j, &v) in x.iter().enumerate() {
        if v >= 0.0 {
            qs[j / 8] |= 1 << (j % 8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(d: f32, qs: [u8; 16]) -> [u8; 18] {
        let mut b = [0u8; 18];
        b[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        b[2..].copy_from_slice(&qs);
        b
    }

    #[test]
    fn registry_row_matches_fork() {
        assert_eq!(crate::TYPE_Q1_0, 41);
        let ty = crate::lookup_type(crate::TYPE_Q1_0).unwrap();
        assert_eq!(ty.name, "Q1_0");
        assert_eq!(ty.block_elems, 128);
        assert_eq!(ty.block_bytes, 18);
        assert_eq!(ty.bytes_for_elems(4096), Some(32 * 18));
        assert_eq!(ty.bytes_for_elems(100), None);
    }

    #[test]
    fn all_clear_is_minus_d_all_set_is_plus_d() {
        let mut out = [0.0f32; 128];
        dequantize_row_q1_0(&block(0.5, [0x00; 16]), &mut out);
        assert!(out.iter().all(|&v| v == -0.5));
        dequantize_row_q1_0(&block(0.5, [0xFF; 16]), &mut out);
        assert!(out.iter().all(|&v| v == 0.5));
    }

    #[test]
    fn bit_order_is_lsb_first() {
        // qs[0] = 0b0000_0101 → weights 0 and 2 set; qs[15] bit 7 → weight 127 set.
        let mut qs = [0u8; 16];
        qs[0] = 0b0000_0101;
        qs[15] = 0b1000_0000;
        let mut out = [0.0f32; 128];
        dequantize_row_q1_0(&block(2.0, qs), &mut out);
        assert_eq!(out[0], 2.0);
        assert_eq!(out[1], -2.0);
        assert_eq!(out[2], 2.0);
        assert_eq!(out[3], -2.0);
        assert_eq!(out[8], -2.0);
        assert_eq!(out[126], -2.0);
        assert_eq!(out[127], 2.0);
    }

    #[test]
    fn vec_dot_matches_dequant_dot() {
        let mut qs = [0u8; 16];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = (i as u8).wrapping_mul(37) ^ 0x5A;
        }
        let b = block(0.125, qs);
        let y: Vec<f32> = (0..128).map(|i| (i as f32 - 60.0) * 0.01).collect();
        let mut w = [0.0f32; 128];
        dequantize_row_q1_0(&b, &mut w);
        let want: f32 = w.iter().zip(&y).map(|(a, b)| a * b).sum();
        let got = vec_dot_q1_0(&b, &y);
        assert!((want - got).abs() < 1e-5, "{want} vs {got}");
    }

    #[test]
    fn quantize_ref_is_sign_and_mean_abs() {
        let x: Vec<f32> = (0..128).map(|i| if i % 3 == 0 { -1.0 } else { 1.0 }).collect();
        let mut b = [0u8; 18];
        quantize_row_q1_0(&x, &mut b);
        let mut out = [0.0f32; 128];
        dequantize_row_q1_0(&b, &mut out);
        for (o, v) in out.iter().zip(&x) {
            assert_eq!(*o, *v);
        }
    }
}
