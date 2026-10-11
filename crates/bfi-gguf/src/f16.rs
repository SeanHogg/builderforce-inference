/// IEEE-754 binary16 → f32. Uses the `half` crate so the conversion matches
/// every other GGUF reader that goes through `ggml_fp16_to_fp32`.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[inline]
pub fn f32_to_f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

/// bfloat16 is the top 16 bits of an IEEE-754 f32.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline]
pub fn f32_to_bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

pub fn read_f16_le(bytes: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]))
}
