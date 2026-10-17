//! `#[cube]` kernels shared by every CubeCL runtime (CUDA, HIP, wgpu/Vulkan,
//! wgpu/Metal, and WebGPU in the browser).
//!
//! Quantized weights stay exactly as GGUF packs them: one device buffer of raw
//! block bytes per tensor, read as `u32` words and unpacked in-shader
//! ([`bytes`]). Nothing here widens a weight matrix to f32.
//!
//! - [`gemm`]: `y[t, row] = Σ W[row, e] · x[t, e]` per weight type (one workgroup
//!   per row, 64 lanes, a tile of 8 tokens per workgroup).
//! - [`rows`]: embedding lookup (`get_rows`) per weight type.
//! - [`ops`]: f32 activations — RMSNorm, NEOX RoPE, causal attention, SwiGLU, add, copy.

pub mod bytes;
pub mod gemm;
pub mod ops;
pub mod rows;

use cubecl::prelude::*;

/// Lanes per workgroup for every kernel in this module.
pub const LANES: u32 = 64;
/// Tokens a GEMM workgroup computes per weight-row read.
pub const TOKEN_TILE: u32 = 8;
/// WebGPU guarantees 65 535 workgroups per dimension; stay under it.
const MAX_CUBES_PER_DIM: u32 = 32_768;

/// Launch geometry for [`gemm`] kernels: rows fold into X·Z (vocab-sized
/// outputs exceed one dimension), token tiles go on Y. Returns the count and the
/// `row_stride` scalar the kernel needs (`row = z * row_stride + x`).
pub fn gemm_cubes(n_out: u32, n_tokens: u32) -> (CubeCount, u32) {
    let x = n_out.clamp(1, MAX_CUBES_PER_DIM);
    let z = n_out.div_ceil(x).max(1);
    let y = n_tokens.div_ceil(TOKEN_TILE).max(1);
    (CubeCount::Static(x, y, z), x)
}

/// One lane per element, 1-D. Elementwise sizes here stay far below 65 535 × 64.
pub fn elementwise_cubes(n: u32) -> CubeCount {
    CubeCount::Static(n.div_ceil(LANES).max(1), 1, 1)
}

pub fn lanes() -> CubeDim {
    CubeDim::new_1d(LANES)
}
