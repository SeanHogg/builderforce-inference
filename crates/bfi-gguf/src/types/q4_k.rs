use crate::f16::read_f16_le;

use super::k_scale::get_scale_min_k4;

/// Q4_K: 256 weights, 144 bytes (`d`, `dmin`, 12 scale bytes, 128 qs).
pub fn dequant_q4_k(block: &[u8], out: &mut [f32]) {
    let d = read_f16_le(&block[0..2]);
    let min = read_f16_le(&block[2..4]);
    let scales = &block[4..16];
    let mut q = &block[16..144];
    let mut is = 0;
    for group in 0..4 {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d1 = d * sc1 as f32;
        let m1f = min * m1 as f32;
        let d2 = d * sc2 as f32;
        let m2f = min * m2 as f32;
        let base = group * 64;
        for l in 0..32 {
            let byte = q[l];
            out[base + l] = d1 * (byte & 0x0F) as f32 - m1f;
            out[base + l + 32] = d2 * (byte >> 4) as f32 - m2f;
        }
        q = &q[32..];
        is += 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_k_zero_block_is_zero() {
        let b = [0u8; 144];
        let mut out = [1.0f32; 256];
        dequant_q4_k(&b, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }
}
