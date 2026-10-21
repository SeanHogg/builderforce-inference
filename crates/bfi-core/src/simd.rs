//! The CPU backend's three hot loops — the f32 dot product, `y += w·x`, and the 1-bit
//! `Q1_0` row dot — with a wasm SIMD128 path for the browser engine's CPU backend.
//!
//! Natively (and on wasm32 built without `+simd128`) every function here is exactly the
//! scalar code `cpu.rs` always ran, so native numerics do not move. On
//! `wasm32` + `simd128` (the workspace's `.cargo/config.toml` turns it on) the f32 dot
//! and the `Q1_0` row dot accumulate in four lanes; a lane-order scalar twin
//! ([`lanes`]) computes the same sums in the same order, so the tests can demand the
//! SIMD result BIT-identical to it, and the twin within float rounding of the plain
//! sequential reference. `axpy` is element-wise, so its SIMD path is bit-identical to
//! the scalar one outright.

use bfi_gguf::{f16_to_f32, vec_dot_q1_0};

/// `Q1_0` block: f16 scale + 16 bytes of sign bits for 128 weights.
const Q1_0_BLOCK_BYTES: usize = 18;
#[cfg_attr(all(target_arch = "wasm32", target_feature = "simd128", not(test)), allow(dead_code))]
const Q1_0_BLOCK_ELEMS: usize = 128;

/// `Σ a[i]·b[i]` over the shorter of the two slices.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        wasm::dot_f32(a, b)
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        scalar_dot_f32(a, b)
    }
}

/// `out[i] += w · x[i]` (attention's value accumulation).
#[inline]
pub fn axpy(out: &mut [f32], w: f32, x: &[f32]) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        wasm::axpy(out, w, x)
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        scalar_axpy(out, w, x)
    }
}

/// Dot of one packed `Q1_0` row (`n` blocks of 18 bytes) with `x` (`n · 128` floats).
#[inline]
pub fn dot_q1_0_row(row: &[u8], x: &[f32]) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        wasm::dot_q1_0_row(row, x)
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        scalar_dot_q1_0_row(row, x)
    }
}

/// The sequential reference (what `cpu.rs` computed before this module existed).
#[cfg_attr(all(target_arch = "wasm32", target_feature = "simd128", not(test)), allow(dead_code))]
pub(crate) fn scalar_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg_attr(all(target_arch = "wasm32", target_feature = "simd128", not(test)), allow(dead_code))]
pub(crate) fn scalar_axpy(out: &mut [f32], w: f32, x: &[f32]) {
    for (o, v) in out.iter_mut().zip(x) {
        *o += w * v;
    }
}

/// The sequential reference: bfi-gguf's per-block `vec_dot_q1_0`, summed block by block.
#[cfg_attr(all(target_arch = "wasm32", target_feature = "simd128", not(test)), allow(dead_code))]
pub(crate) fn scalar_dot_q1_0_row(row: &[u8], x: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for (b, blk) in row.chunks_exact(Q1_0_BLOCK_BYTES).enumerate() {
        s += vec_dot_q1_0(blk, &x[b * Q1_0_BLOCK_ELEMS..(b + 1) * Q1_0_BLOCK_ELEMS]);
    }
    s
}

/// `SIGN_FLIP[byte][k]`: `0` when bit `k` of `byte` is set (weight `+d`), the f32 sign
/// bit when clear (`-d`). XOR-ing an activation with it is exact negation, so
/// `Σ (y ^ flip)` is `Σ_set y − Σ_clear y` with no branch.
#[cfg_attr(not(all(target_arch = "wasm32", target_feature = "simd128")), allow(dead_code))]
static SIGN_FLIP: [[u32; 8]; 256] = build_sign_flip();

const fn build_sign_flip() -> [[u32; 8]; 256] {
    let mut t = [[0u32; 8]; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut k = 0;
        while k < 8 {
            if (byte >> k) & 1 == 0 {
                t[byte][k] = 0x8000_0000;
            }
            k += 1;
        }
        byte += 1;
    }
    t
}

/// Scalar twins of the SIMD kernels: the same four-lane accumulation, in the same order.
/// The SIMD paths must equal these bit for bit; these must equal the sequential
/// references to float rounding.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod lanes {
    use super::{f16_to_f32, SIGN_FLIP, Q1_0_BLOCK_BYTES};

    /// Lanes `l0..l3` reduced as `(l0 + l1) + (l2 + l3)` — `wasm::hsum`'s order.
    fn hsum(l: [f32; 4]) -> f32 {
        (l[0] + l[1]) + (l[2] + l[3])
    }

    pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let mut acc = [0.0f32; 4];
        let body = n / 4 * 4;
        let mut i = 0;
        while i < body {
            for k in 0..4 {
                acc[k] += a[i + k] * b[i + k];
            }
            i += 4;
        }
        let mut s = hsum(acc);
        for j in body..n {
            s += a[j] * b[j];
        }
        s
    }

    pub fn dot_q1_0_row(row: &[u8], x: &[f32]) -> f32 {
        let mut s = 0.0f32;
        for (b, blk) in row.chunks_exact(Q1_0_BLOCK_BYTES).enumerate() {
            let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let y = &x[b * 128..b * 128 + 128];
            let mut lo = [0.0f32; 4];
            let mut hi = [0.0f32; 4];
            for (j, &byte) in blk[2..].iter().enumerate() {
                let flip = &SIGN_FLIP[byte as usize];
                for k in 0..4 {
                    lo[k] += f32::from_bits(y[j * 8 + k].to_bits() ^ flip[k]);
                    hi[k] += f32::from_bits(y[j * 8 + 4 + k].to_bits() ^ flip[4 + k]);
                }
            }
            let mut sum = [0.0f32; 4];
            for k in 0..4 {
                sum[k] = lo[k] + hi[k];
            }
            s += hsum(sum) * d;
        }
        s
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use core::arch::wasm32::*;

    use super::{f16_to_f32, SIGN_FLIP, Q1_0_BLOCK_BYTES};

    #[inline(always)]
    fn hsum(v: v128) -> f32 {
        (f32x4_extract_lane::<0>(v) + f32x4_extract_lane::<1>(v))
            + (f32x4_extract_lane::<2>(v) + f32x4_extract_lane::<3>(v))
    }

    /// Unaligned 4-float load (wasm `v128.load` accepts any alignment).
    #[inline(always)]
    unsafe fn load(p: *const f32) -> v128 {
        v128_load(p as *const v128)
    }

    pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let body = n / 4 * 4;
        let mut acc = f32x4_splat(0.0);
        let mut i = 0;
        // SAFETY: `i + 4 <= body <= n` for both slices.
        unsafe {
            while i < body {
                acc = f32x4_add(acc, f32x4_mul(load(a.as_ptr().add(i)), load(b.as_ptr().add(i))));
                i += 4;
            }
        }
        let mut s = hsum(acc);
        for j in body..n {
            s += a[j] * b[j];
        }
        s
    }

    pub fn axpy(out: &mut [f32], w: f32, x: &[f32]) {
        let n = out.len().min(x.len());
        let body = n / 4 * 4;
        let wv = f32x4_splat(w);
        let mut i = 0;
        // SAFETY: `i + 4 <= body <= n` for both slices.
        unsafe {
            while i < body {
                let o = out.as_mut_ptr().add(i);
                let r = f32x4_add(load(o), f32x4_mul(wv, load(x.as_ptr().add(i))));
                v128_store(o as *mut v128, r);
                i += 4;
            }
        }
        for j in body..n {
            out[j] += w * x[j];
        }
    }

    pub fn dot_q1_0_row(row: &[u8], x: &[f32]) -> f32 {
        let n_blocks = row.len() / Q1_0_BLOCK_BYTES;
        assert!(x.len() >= n_blocks * 128, "Q1_0 row of {n_blocks} blocks needs {} activations", n_blocks * 128);
        let mut s = 0.0f32;
        for b in 0..n_blocks {
            let blk = &row[b * Q1_0_BLOCK_BYTES..(b + 1) * Q1_0_BLOCK_BYTES];
            let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let y = x[b * 128..].as_ptr();
            let mut lo = f32x4_splat(0.0);
            let mut hi = f32x4_splat(0.0);
            // SAFETY: the block's 128 activations are in bounds (asserted above); each
            // `SIGN_FLIP` row is 8 contiguous u32s.
            unsafe {
                for j in 0..16 {
                    let flip = SIGN_FLIP[blk[2 + j] as usize].as_ptr();
                    let f_lo = v128_load(flip as *const v128);
                    let f_hi = v128_load(flip.add(4) as *const v128);
                    lo = f32x4_add(lo, v128_xor(load(y.add(j * 8)), f_lo));
                    hi = f32x4_add(hi, v128_xor(load(y.add(j * 8 + 4)), f_hi));
                }
            }
            s += hsum(f32x4_add(lo, hi)) * d;
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bfi_gguf::quantize_row_q1_0;

    /// Deterministic pseudo-random floats in [-1, 1).
    fn floats(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2_654_435_761).max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn q1_0_row(n_blocks: usize, seed: u32) -> Vec<u8> {
        let w = floats(n_blocks * 128, seed);
        let mut row = vec![0u8; n_blocks * Q1_0_BLOCK_BYTES];
        for (b, blk) in row.chunks_exact_mut(Q1_0_BLOCK_BYTES).enumerate() {
            quantize_row_q1_0(&w[b * 128..(b + 1) * 128], blk);
        }
        row
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn lane_twins_match_the_sequential_references() {
        for n in [0, 1, 3, 4, 7, 128, 129, 2048] {
            let (a, b) = (floats(n, 1), floats(n, 2));
            let (want, got) = (scalar_dot_f32(&a, &b), lanes::dot_f32(&a, &b));
            assert!(close(want, got), "dot n={n}: {want} vs {got}");
        }
        for blocks in [1, 2, 16, 32] {
            let (row, x) = (q1_0_row(blocks, 3), floats(blocks * 128, 4));
            let (want, got) = (scalar_dot_q1_0_row(&row, &x), lanes::dot_q1_0_row(&row, &x));
            assert!(close(want, got), "q1_0 blocks={blocks}: {want} vs {got}");
        }
    }

    #[test]
    fn sign_flip_is_exact_negation_of_clear_bits() {
        assert_eq!(SIGN_FLIP[0xFF], [0; 8]);
        assert_eq!(SIGN_FLIP[0x00], [0x8000_0000; 8]);
        assert_eq!(SIGN_FLIP[0b0000_0101][0], 0);
        assert_eq!(SIGN_FLIP[0b0000_0101][1], 0x8000_0000);
        let v = 0.375f32;
        assert_eq!(f32::from_bits(v.to_bits() ^ 0x8000_0000), -v);
    }

    #[test]
    fn dispatch_matches_the_active_path() {
        // Natively the dispatchers ARE the sequential references; on wasm32 + simd128
        // they must equal the lane twins exactly (see `simd_tests`).
        let (a, b) = (floats(515, 5), floats(515, 6));
        let (row, x) = (q1_0_row(16, 7), floats(16 * 128, 8));
        let mut o1 = floats(515, 9);
        let mut o2 = o1.clone();
        axpy(&mut o1, 0.3, &a);
        scalar_axpy(&mut o2, 0.3, &a);
        assert_eq!(o1, o2, "axpy is element-wise: identical on every path");
        if cfg!(all(target_arch = "wasm32", target_feature = "simd128")) {
            assert_eq!(dot_f32(&a, &b).to_bits(), lanes::dot_f32(&a, &b).to_bits());
            assert_eq!(dot_q1_0_row(&row, &x).to_bits(), lanes::dot_q1_0_row(&row, &x).to_bits());
        } else {
            assert_eq!(dot_f32(&a, &b).to_bits(), scalar_dot_f32(&a, &b).to_bits());
            assert_eq!(dot_q1_0_row(&row, &x).to_bits(), scalar_dot_q1_0_row(&row, &x).to_bits());
        }
    }
}

/// The SIMD-vs-scalar proof, run where the SIMD path exists: `wasm-pack test --node
/// crates/bfi-core` (wasm32 + simd128 from `.cargo/config.toml`).
#[cfg(all(test, target_arch = "wasm32", target_feature = "simd128"))]
mod simd_tests {
    use super::*;
    use bfi_gguf::quantize_row_q1_0;
    use wasm_bindgen_test::wasm_bindgen_test;

    fn floats(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2_654_435_761).max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[wasm_bindgen_test]
    fn simd_f32_dot_is_bit_identical_to_its_lane_twin() {
        for n in [0, 1, 3, 4, 5, 8, 127, 128, 1031, 4096] {
            let (a, b) = (floats(n, 11), floats(n, 12));
            assert_eq!(dot_f32(&a, &b).to_bits(), lanes::dot_f32(&a, &b).to_bits(), "n={n}");
            let r = scalar_dot_f32(&a, &b);
            assert!((r - dot_f32(&a, &b)).abs() <= 1e-5 * r.abs().max(1.0), "n={n}");
        }
    }

    #[wasm_bindgen_test]
    fn simd_q1_0_row_is_bit_identical_to_its_lane_twin() {
        for blocks in [1, 2, 16, 32, 64] {
            let w = floats(blocks * 128, 13);
            let mut row = vec![0u8; blocks * Q1_0_BLOCK_BYTES];
            for (b, blk) in row.chunks_exact_mut(Q1_0_BLOCK_BYTES).enumerate() {
                quantize_row_q1_0(&w[b * 128..(b + 1) * 128], blk);
            }
            let x = floats(blocks * 128, 14);
            assert_eq!(dot_q1_0_row(&row, &x).to_bits(), lanes::dot_q1_0_row(&row, &x).to_bits(), "blocks={blocks}");
            let r = scalar_dot_q1_0_row(&row, &x);
            assert!((r - dot_q1_0_row(&row, &x)).abs() <= 1e-5 * r.abs().max(1.0), "blocks={blocks}");
        }
    }

    #[wasm_bindgen_test]
    fn simd_axpy_is_bit_identical_to_scalar() {
        let x = floats(1029, 15);
        let mut a = floats(1029, 16);
        let mut b = a.clone();
        axpy(&mut a, -0.7, &x);
        scalar_axpy(&mut b, -0.7, &x);
        assert_eq!(a, b);
    }
}
