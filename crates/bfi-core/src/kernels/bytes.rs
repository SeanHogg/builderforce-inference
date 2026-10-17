//! In-shader unpacking of GGUF bytes stored as little-endian `u32` words.
//!
//! GGUF blocks are not 4-byte aligned (`Q1_0` = 18 B, `Q8_0` = 34 B), and WGSL
//! has no byte loads or `f16` without the optional `shader-f16` feature, so both
//! are done with integer ops here.

use cubecl::prelude::*;

/// Byte `off` of a word array (little-endian).
#[cube]
pub fn byte_at(w: &Array<u32>, off: u32) -> u32 {
    let word = w[usize::cast_from(off / 4u32)];
    (word >> ((off % 4u32) * 8u32)) & 255u32
}

/// IEEE binary16 bits → f32 (subnormals kept; inf/NaN are not produced by GGUF scales).
#[cube]
pub fn half_to_f32(h: u32) -> f32 {
    let exp = (h >> 10u32) & 31u32;
    let mant = h & 1023u32;
    // Subnormal / zero by default; normal numbers rebuild the f32 bit pattern.
    let mut out = f32::cast_from(mant) * 0.000_000_059_604_645f32;
    if exp != 0u32 {
        let bits = ((exp + 112u32) << 23u32) | (mant << 13u32);
        out = f32::reinterpret(bits);
    }
    if ((h >> 15u32) & 1u32) == 1u32 {
        out = 0.0f32 - out;
    }
    out
}

/// Little-endian f16 starting at byte `off`.
#[cube]
pub fn half_at(w: &Array<u32>, off: u32) -> f32 {
    half_to_f32(byte_at(w, off) | (byte_at(w, off + 1u32) << 8u32))
}

/// F32 element `idx`.
#[cube]
pub fn weight_f32(w: &Array<u32>, idx: u32) -> f32 {
    f32::reinterpret(w[usize::cast_from(idx)])
}

/// F16 element `idx` (two per word).
#[cube]
pub fn weight_f16(w: &Array<u32>, idx: u32) -> f32 {
    let word = w[usize::cast_from(idx / 2u32)];
    half_to_f32((word >> ((idx % 2u32) * 16u32)) & 65535u32)
}

/// `Q8_0` element `(row, e)`: `block_q8_0 { f16 d; i8 qs[32]; }` (34 B).
#[cube]
pub fn weight_q8_0(w: &Array<u32>, row: u32, e: u32, n_in: u32) -> f32 {
    let base = (row * (n_in / 32u32) + e / 32u32) * 34u32;
    let d = half_at(w, base);
    let q = byte_at(w, base + 2u32 + e % 32u32);
    let mut v = f32::cast_from(q);
    if q >= 128u32 {
        v -= 256.0f32;
    }
    v * d
}

/// `Q1_0` element `(row, e)`: `block_q1_0 { f16 d; u8 qs[16]; }` (18 B),
/// bit `j` LSB-first, set ⇒ `+d`, clear ⇒ `-d` (PrismML `dequantize_row_q1_0`).
#[cube]
pub fn weight_q1_0(w: &Array<u32>, row: u32, e: u32, n_in: u32) -> f32 {
    let base = (row * (n_in / 128u32) + e / 128u32) * 18u32;
    let d = half_at(w, base);
    let bits = byte_at(w, base + 2u32 + (e % 128u32) / 8u32);
    let mut v = 0.0f32 - d;
    if ((bits >> (e % 8u32)) & 1u32) == 1u32 {
        v = d;
    }
    v
}
