/// BF16: one weight, 2 bytes, top 16 bits of IEEE-754 f32.
pub fn dequant_bf16(block: &[u8], out: &mut [f32]) {
    let bits = u16::from_le_bytes([block[0], block[1]]);
    out[0] = crate::f16::bf16_to_f32(bits);
}
