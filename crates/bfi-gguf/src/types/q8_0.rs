use crate::f16::read_f16_le;

/// Q8_0: `ggml_half d` + 32 int8 qs. value = qs[i] * d.
pub fn dequant_q8_0(block: &[u8], out: &mut [f32]) {
    let d = read_f16_le(&block[0..2]);
    for i in 0..32 {
        out[i] = (block[2 + i] as i8 as f32) * d;
    }
}

pub fn vec_dot_q8_0(block: &[u8], y: &[f32]) -> f32 {
    let mut tmp = [0.0f32; 32];
    dequant_q8_0(block, &mut tmp);
    tmp.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::f16::f32_to_f16;

    #[test]
    fn q8_0_identity_at_d1() {
        let mut b = [0u8; 34];
        b[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        b[2] = 5u8;
        b[3] = (-3i8) as u8;
        let mut out = [0.0f32; 32];
        dequant_q8_0(&b, &mut out);
        assert!((out[0] - 5.0).abs() < 1e-3);
        assert!((out[1] + 3.0).abs() < 1e-3);
    }
}
