use crate::f16::read_f16_le;

use super::k_scale::get_scale_min_k4;

/// Q5_K: 256 weights, 176 bytes.
pub fn dequant_q5_k(block: &[u8], out: &mut [f32]) {
    let d = read_f16_le(&block[0..2]);
    let min = read_f16_le(&block[2..4]);
    let scales = &block[4..16];
    let qh = &block[16..48];
    let mut ql = &block[48..176];
    let mut is = 0;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;
    for group in 0..4 {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d1 = d * sc1 as f32;
        let m1f = min * m1 as f32;
        let d2 = d * sc2 as f32;
        let m2f = min * m2 as f32;
        let base = group * 64;
        for l in 0..32 {
            let extra_l = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
            let extra_h = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
            let byte = ql[l];
            out[base + l] = d1 * ((byte & 0x0F) as f32 + extra_l) - m1f;
            out[base + l + 32] = d2 * ((byte >> 4) as f32 + extra_h) - m2f;
        }
        ql = &ql[32..];
        is += 2;
        u1 = u1.wrapping_shl(2);
        u2 = u2.wrapping_shl(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q5_k_zero_block_is_zero() {
        let b = [0u8; 176];
        let mut out = [1.0f32; 256];
        dequant_q5_k(&b, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }
}
