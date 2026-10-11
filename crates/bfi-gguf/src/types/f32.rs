/// F32: one weight, 4 bytes, little-endian IEEE-754.
pub fn dequant_f32(block: &[u8], out: &mut [f32]) {
    out[0] = f32::from_le_bytes(block[0..4].try_into().unwrap());
}
