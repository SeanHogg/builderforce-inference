//! PQ2_0 / `block_pq2_0`: 128 weights in 34 bytes.
//!
//! Ported from the PrismML llama.cpp fork (`ggml-quants.c`). MIT:
//!
//! Copyright (c) 2023-2025 The ggml authors.
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//! The above copyright notice and this permission notice shall be included in
//! all copies or substantial portions of the Software.

use crate::f16::{f16_to_f32, read_f16_le};

/// ```c
/// #define QK_Q2_0 128
/// typedef struct {
///     ggml_half d;
///     uint8_t qs[QK_Q2_0 / 4];
/// } block_pq2_0;
/// ```
///
/// Little-end first: weight `j = (qs[j/4] >> (2*(j%4))) & 3`.
/// Codes `{0,1,2,3}` map to `{-1, 0, +1, +2}` times `d`.
pub fn dequantize_row_pq2_0(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), 34);
    debug_assert_eq!(out.len(), 128);
    let d = read_f16_le(&block[0..2]);
    let qs = &block[2..34];
    for j in 0..128 {
        let q = (qs[j / 4] >> (2 * (j % 4))) & 3;
        out[j] = (q as i8 as f32 - 1.0) * d;
    }
}

pub fn vec_dot_pq2_0(block: &[u8], y: &[f32]) -> f32 {
    let mut tmp = [0.0f32; 128];
    dequantize_row_pq2_0(block, &mut tmp);
    tmp.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

/// Encode 128 f32 weights into one PQ2_0 block (nearest of {-1,0,+1,+2} * d).
/// Used by tests; production loaders never encode.
pub fn quantize_row_pq2_0(x: &[f32], block: &mut [u8]) {
    assert_eq!(x.len(), 128);
    assert_eq!(block.len(), 34);
    let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let d = if amax == 0.0 { 1.0 } else { amax / 2.0 };
    let bits = crate::f16::f32_to_f16(d);
    block[0..2].copy_from_slice(&bits.to_le_bytes());
    let d = f16_to_f32(bits);
    let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
    let mut qs = [0u8; 32];
    for j in 0..128 {
        let q = (x[j] * inv + 1.0).round().clamp(0.0, 3.0) as u8;
        qs[j / 4] |= q << (2 * (j % 4));
    }
    block[2..34].copy_from_slice(&qs);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::f16::f32_to_f16;

    fn block_pq2(d: f32, fill_qs: u8) -> [u8; 34] {
        let mut b = [0u8; 34];
        b[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        b[2..].fill(fill_qs);
        b
    }

    #[test]
    fn pq2_maps_codes_to_minus1_0_plus1_plus2() {
        let mut out = [0.0f32; 128];
        dequantize_row_pq2_0(&block_pq2(1.0, 0x00), &mut out);
        assert!(out.iter().all(|&v| (v + 1.0).abs() < 1e-4), "00 → -1");
        dequantize_row_pq2_0(&block_pq2(1.0, 0x55), &mut out);
        assert!(out.iter().all(|&v| v.abs() < 1e-4), "01 → 0");
        dequantize_row_pq2_0(&block_pq2(1.0, 0xAA), &mut out);
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-4), "10 → +1");
        dequantize_row_pq2_0(&block_pq2(1.0, 0xFF), &mut out);
        assert!(out.iter().all(|&v| (v - 2.0).abs() < 1e-4), "11 → +2");
    }

    #[test]
    fn pq2_little_end_first_packing() {
        let mut b = block_pq2(1.0, 0);
        // weight 0 = code 2 (+1), weight 1 = code 3 (+2), rest 0 (-1)
        b[2] = 0b00_00_11_10;
        let mut out = [0.0f32; 128];
        dequantize_row_pq2_0(&b, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-4);
        assert!((out[1] - 2.0).abs() < 1e-4);
        assert!((out[2] + 1.0).abs() < 1e-4);
    }

    #[test]
    fn pq2_prd_formula_scale_and_index() {
        // PRD §3.1: value = (q - 1) * d, q = (qs[j/4] >> (2*(j%4))) & 3
        let mut b = [0u8; 34];
        b[0..2].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
        // j=4 lives in qs[1], shift 0: code 2 → (2-1)*0.5 = 0.5
        b[2 + 1] = 0b0000_0010;
        let mut out = [0.0f32; 128];
        dequantize_row_pq2_0(&b, &mut out);
        assert!((out[4] - 0.5).abs() < 1e-4, "{}", out[4]);
        assert!((out[0] + 0.5).abs() < 1e-4, "unset → code 0 → -d");
    }

    #[test]
    fn pq2_roundtrip_constant() {
        let x = [0.5f32; 128];
        let mut block = [0u8; 34];
        quantize_row_pq2_0(&x, &mut block);
        let mut out = [0.0f32; 128];
        dequantize_row_pq2_0(&block, &mut out);
        for v in out {
            assert!((v - 0.5).abs() < 0.02, "{v}");
        }
    }

    #[test]
    fn pq2_block_is_34_bytes() {
        assert_eq!(crate::TYPE_PQ2_0, 142);
        let ty = crate::lookup_type(crate::TYPE_PQ2_0).unwrap();
        assert_eq!(ty.block_bytes, 34);
        assert_eq!(ty.block_elems, 128);
    }
}
