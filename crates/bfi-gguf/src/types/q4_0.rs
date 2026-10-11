use crate::f16::read_f16_le;

/// Q4_0: `ggml_half d` + 16 bytes of 4-bit qs. 32 weights. value = (nibble - 8) * d.
pub fn dequant_q4_0(block: &[u8], out: &mut [f32]) {
    let d = read_f16_le(&block[0..2]);
    let qs = &block[2..18];
    for j in 0..16 {
        let byte = qs[j];
        out[j] = ((byte & 0x0F) as i8 as f32 - 8.0) * d;
        out[j + 16] = ((byte >> 4) as i8 as f32 - 8.0) * d;
    }
}

pub fn vec_dot_q4_0(block: &[u8], y: &[f32]) -> f32 {
    let mut tmp = [0.0f32; 32];
    dequant_q4_0(block, &mut tmp);
    tmp.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::f16::f32_to_f16;

    #[test]
    fn q4_0_nibble_minus_eight() {
        let mut b = [0u8; 18];
        b[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        let mut out = [0.0f32; 32];
        dequant_q4_0(&b, &mut out);
        assert!(out.iter().all(|&v| (v + 8.0).abs() < 1e-3));
    }
}
