//! PTQ1_0 / `block_ptq1_0`: 128 weights in 28 bytes.
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

use crate::f16::read_f16_le;

/// ```c
/// #define QK_TQ1_0 128
/// typedef struct {
///     uint8_t qs[(QK_TQ1_0 - 4 * QK_TQ1_0 / 64) / 5]; /* 24 */
///     uint8_t qh[4 * QK_TQ1_0 / 64];                  /* 2  */
///     ggml_half d;
/// } block_ptq1_0;
/// ```
///
/// Decoding follows the fork's `dequantize_row_ptq1_0`, with stages `{32, 16, 8}`
/// and `pow3 = {1,3,9,27,81,243}`:
/// - The 24-byte `qs` is consumed in stages. A 32-byte stage doesn't fit, so the
///   16-byte stage covers bytes 0..16 and the 8-byte stage covers bytes 16..24.
/// - Within a stage of `c` bytes starting at byte `b`, the output order is
///   `for n in 0..5 { for m in 0..c { trit(qs[b+m], n) } }`.
/// - Then `for n in 0..4 { for h in 0..2 { trit(qh[h], n) } }`.
/// - `trit(byte, n) = ((u16)((u8)(byte * pow3[n])) * 3) >> 8`.
///   The value is `(trit - 1) * d`.
const POW3: [u16; 6] = [1, 3, 9, 27, 81, 243];

#[inline]
fn trit(byte: u8, n: usize) -> u8 {
    let wrapped = (byte as u16).wrapping_mul(POW3[n]) as u8;
    ((wrapped as u16 * 3) >> 8) as u8
}

pub fn dequantize_row_ptq1_0(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), 28);
    debug_assert_eq!(out.len(), 128);
    let qs = &block[0..24];
    let qh = &block[24..26];
    let d = read_f16_le(&block[26..28]);
    let mut i = 0usize;
    let mut b = 0usize;
    for c in [32usize, 16, 8] {
        if b + c > qs.len() {
            continue;
        }
        for n in 0..5 {
            for m in 0..c {
                out[i] = (trit(qs[b + m], n) as i8 as f32 - 1.0) * d;
                i += 1;
            }
        }
        b += c;
    }
    for n in 0..4 {
        for h in 0..2 {
            out[i] = (trit(qh[h], n) as i8 as f32 - 1.0) * d;
            i += 1;
        }
    }
    debug_assert_eq!(i, 128);
}

pub fn vec_dot_ptq1_0(block: &[u8], y: &[f32]) -> f32 {
    let mut tmp = [0.0f32; 128];
    dequantize_row_ptq1_0(block, &mut tmp);
    tmp.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::f16::f32_to_f16;

    /// Independent copy of PRD §3.1 `trit` so a regression in `trit` is visible.
    fn prd_trit(byte: u8, n: usize) -> u8 {
        let pow3 = [1u16, 3, 9, 27, 81, 243];
        let wrapped = (byte as u16).wrapping_mul(pow3[n]) as u8;
        ((wrapped as u16 * 3) >> 8) as u8
    }

    fn prd_dequant(block: &[u8; 28]) -> [f32; 128] {
        let qs = &block[0..24];
        let qh = &block[24..26];
        let d = crate::f16::f16_to_f32(u16::from_le_bytes([block[26], block[27]]));
        let mut out = [0.0f32; 128];
        let mut i = 0usize;
        // 32-byte stage does not fit in 24-byte qs; 16 then 8.
        for (b, c) in [(0usize, 16usize), (16, 8)] {
            for n in 0..5 {
                for m in 0..c {
                    out[i] = (prd_trit(qs[b + m], n) as i8 as f32 - 1.0) * d;
                    i += 1;
                }
            }
        }
        for n in 0..4 {
            for h in 0..2 {
                out[i] = (prd_trit(qh[h], n) as i8 as f32 - 1.0) * d;
                i += 1;
            }
        }
        assert_eq!(i, 128);
        out
    }

    #[test]
    fn ptq1_zero_qs_is_minus_d() {
        let mut b = [0u8; 28];
        b[26..28].copy_from_slice(&f32_to_f16(2.0).to_le_bytes());
        let mut out = [0.0f32; 128];
        dequantize_row_ptq1_0(&b, &mut out);
        assert_eq!(out.len(), 128);
        for v in out {
            assert!((v + 2.0).abs() < 1e-3, "{v}");
        }
    }

    #[test]
    fn ptq1_writes_128_weights() {
        let mut b = [0u8; 28];
        b[0..24].fill(1);
        b[24..26].fill(1);
        b[26..28].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        let mut out = [f32::NAN; 128];
        dequantize_row_ptq1_0(&b, &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn ptq1_matches_prd_reference_on_synthetic_block() {
        let mut b = [0u8; 28];
        for i in 0..24 {
            b[i] = (i as u8).wrapping_mul(13).wrapping_add(7);
        }
        b[24] = 0xA5;
        b[25] = 0x5A;
        b[26..28].copy_from_slice(&f32_to_f16(1.25).to_le_bytes());
        let mut out = [0.0f32; 128];
        dequantize_row_ptq1_0(&b, &mut out);
        let expect = prd_dequant(&b);
        for i in 0..128 {
            assert_eq!(out[i].to_bits(), expect[i].to_bits(), "idx {i}");
        }
    }

    #[test]
    fn ptq1_stage_skip_32_then_16_then_8() {
        // qs[0]=29 → trit(n=0)=0, trit(n=1)=1. Distinguishes 16-byte stages
        // from a single 24-byte stage (which would put qs[16] at index 16).
        let mut b = [0u8; 28];
        b[0] = 29;
        b[16] = 0;
        b[26..28].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        let mut out = [0.0f32; 128];
        dequantize_row_ptq1_0(&b, &mut out);
        assert!((out[0] + 1.0).abs() < 1e-4, "n=0 m=0 trit=0 → -1, got {}", out[0]);
        assert!(out[16].abs() < 1e-4, "n=1 m=0 trit=1 → 0, got {}", out[16]);
        // 8-byte stage starts at weight 80 = trit(qs[16], 0) = 0 → -1
        assert!((out[80] + 1.0).abs() < 1e-4, "8-byte stage start, got {}", out[80]);
    }

    #[test]
    fn ptq1_qh_occupies_last_eight_weights() {
        let mut b = [0u8; 28];
        b[24] = 171; // trit(n=0) = 2
        b[26..28].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        let mut out = [0.0f32; 128];
        dequantize_row_ptq1_0(&b, &mut out);
        assert!((out[120] - 1.0).abs() < 1e-4, "qh[0] n=0 → +1, got {}", out[120]);
        assert!((out[121] + 1.0).abs() < 1e-4, "qh[1] n=0 is 0 → -1");
        for i in 0..120 {
            assert!((out[i] + 1.0).abs() < 1e-4, "qs zero at {i}");
        }
    }

    #[test]
    fn ptq1_scale_is_last_two_bytes() {
        let mut b = [0u8; 28];
        b[26..28].copy_from_slice(&f32_to_f16(4.0).to_le_bytes());
        let mut out = [0.0f32; 128];
        dequantize_row_ptq1_0(&b, &mut out);
        assert!((out[0] + 4.0).abs() < 1e-2);
    }

    #[test]
    fn ptq1_block_is_28_bytes() {
        assert_eq!(crate::TYPE_PTQ1_0, 143);
        let ty = crate::lookup_type(crate::TYPE_PTQ1_0).unwrap();
        assert_eq!(ty.block_bytes, 28);
        assert_eq!(ty.block_elems, 128);
    }
}
