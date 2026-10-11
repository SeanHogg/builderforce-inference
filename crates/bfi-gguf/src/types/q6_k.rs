use crate::f16::read_f16_le;

/// Q6_K: 256 weights, 210 bytes (`ql[128]`, `qh[64]`, `scales[16]`, `d`).
pub fn dequant_q6_k(block: &[u8], out: &mut [f32]) {
    let ql_all = &block[0..128];
    let qh_all = &block[128..192];
    let sc = &block[192..208];
    let d = read_f16_le(&block[208..210]);
    for n in 0..2 {
        let ql = &ql_all[n * 64..];
        let qh = &qh_all[n * 32..];
        let scales = &sc[n * 8..];
        let base = n * 128;
        for l in 0..32 {
            let q1 = ((ql[l] & 0x0F) as i32 | ((qh[l] as i32 & 3) << 4)) - 32;
            let q2 = ((ql[l + 32] & 0x0F) as i32 | (((qh[l] as i32 >> 2) & 3) << 4)) - 32;
            let q3 = ((ql[l] >> 4) as i32 | (((qh[l] as i32 >> 4) & 3) << 4)) - 32;
            let q4 = ((ql[l + 32] >> 4) as i32 | (((qh[l] as i32 >> 6) & 3) << 4)) - 32;
            let s0 = scales[l / 16] as i8 as f32;
            let s1 = scales[l / 16 + 2] as i8 as f32;
            let s2 = scales[l / 16 + 4] as i8 as f32;
            let s3 = scales[l / 16 + 6] as i8 as f32;
            out[base + l] = d * s0 * q1 as f32;
            out[base + 32 + l] = d * s1 * q2 as f32;
            out[base + 64 + l] = d * s2 * q3 as f32;
            out[base + 96 + l] = d * s3 * q4 as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q6_k_zero_d_is_zero() {
        let b = [0u8; 210];
        let mut out = [1.0f32; 256];
        dequant_q6_k(&b, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }
}
