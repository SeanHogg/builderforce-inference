use crate::f16::read_f16_le;

/// F16: one weight, 2 bytes, IEEE-754 binary16.
pub fn dequant_f16(block: &[u8], out: &mut [f32]) {
    out[0] = read_f16_le(block);
}
