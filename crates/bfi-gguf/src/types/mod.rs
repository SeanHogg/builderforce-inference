//! GGML type registry. Layouts are data, not a pile of `match` arms.
//!
//! Adding a type = one table row below + one file in this directory.
//! PQ2_0 (id 142) and PTQ1_0 (id 143) match the PrismML llama.cpp fork
//! (`ggml-common.h` / `ggml-quants.c`). Ported files keep the MIT notice.

mod bf16;
mod f16;
mod f32;
mod k_scale;
mod pq2_0;
mod ptq1_0;
mod q4_0;
mod q4_k;
mod q5_k;
mod q6_k;
mod q8_0;

pub use bf16::dequant_bf16;
pub use f16::dequant_f16;
pub use f32::dequant_f32;
pub use pq2_0::{dequantize_row_pq2_0, quantize_row_pq2_0, vec_dot_pq2_0};
pub use ptq1_0::{dequantize_row_ptq1_0, vec_dot_ptq1_0};
pub use q4_0::{dequant_q4_0, vec_dot_q4_0};
pub use q4_k::dequant_q4_k;
pub use q5_k::dequant_q5_k;
pub use q6_k::dequant_q6_k;
pub use q8_0::{dequant_q8_0, vec_dot_q8_0};

use crate::error::{GgufError, Result};

pub const TYPE_F32: u32 = 0;
pub const TYPE_F16: u32 = 1;
pub const TYPE_Q4_0: u32 = 2;
pub const TYPE_Q8_0: u32 = 8;
pub const TYPE_Q4_K: u32 = 12;
pub const TYPE_Q5_K: u32 = 13;
pub const TYPE_Q6_K: u32 = 14;
pub const TYPE_BF16: u32 = 30;
/// PrismML packed 2-bit. Identical layout to llama.cpp `Q2_0` (id 42).
pub const TYPE_PQ2_0: u32 = 142;
/// PrismML packed ternary. Identical layout to llama.cpp `TQ1_0` (id 34) with QK=128.
pub const TYPE_PTQ1_0: u32 = 143;

/// One GGML / GGUF tensor type.
#[derive(Clone, Copy)]
pub struct GgmlType {
    pub id: u32,
    pub name: &'static str,
    pub block_elems: usize,
    pub block_bytes: usize,
    pub dequant_row: fn(&[u8], &mut [f32]),
    pub vec_dot: Option<fn(&[u8], &[f32]) -> f32>,
}

impl std::fmt::Debug for GgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgmlType")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("block_elems", &self.block_elems)
            .field("block_bytes", &self.block_bytes)
            .finish()
    }
}

impl GgmlType {
    pub fn bytes_for_elems(&self, n: u64) -> Option<u64> {
        let be = self.block_elems as u64;
        if n % be != 0 {
            return None;
        }
        Some((n / be) * self.block_bytes as u64)
    }
}

const fn t(
    id: u32,
    name: &'static str,
    block_elems: usize,
    block_bytes: usize,
    dequant_row: fn(&[u8], &mut [f32]),
    vec_dot: Option<fn(&[u8], &[f32]) -> f32>,
) -> GgmlType {
    GgmlType {
        id,
        name,
        block_elems,
        block_bytes,
        dequant_row,
        vec_dot,
    }
}

/// Types we load in M1/M2. Anything else is an error that names the id.
pub static TYPES: &[GgmlType] = &[
    t(
        TYPE_F32,
        "F32",
        1,
        4,
        crate::types::f32::dequant_f32,
        None,
    ),
    t(
        TYPE_F16,
        "F16",
        1,
        2,
        crate::types::f16::dequant_f16,
        None,
    ),
    t(
        TYPE_Q4_0,
        "Q4_0",
        32,
        18,
        crate::types::q4_0::dequant_q4_0,
        Some(crate::types::q4_0::vec_dot_q4_0),
    ),
    t(
        TYPE_Q8_0,
        "Q8_0",
        32,
        34,
        crate::types::q8_0::dequant_q8_0,
        Some(crate::types::q8_0::vec_dot_q8_0),
    ),
    t(TYPE_Q4_K, "Q4_K", 256, 144, crate::types::q4_k::dequant_q4_k, None),
    t(TYPE_Q5_K, "Q5_K", 256, 176, crate::types::q5_k::dequant_q5_k, None),
    t(TYPE_Q6_K, "Q6_K", 256, 210, crate::types::q6_k::dequant_q6_k, None),
    t(
        TYPE_BF16,
        "BF16",
        1,
        2,
        crate::types::bf16::dequant_bf16,
        None,
    ),
    t(
        TYPE_PQ2_0,
        "PQ2_0",
        128,
        34,
        crate::types::pq2_0::dequantize_row_pq2_0,
        Some(crate::types::pq2_0::vec_dot_pq2_0),
    ),
    t(
        TYPE_PTQ1_0,
        "PTQ1_0",
        128,
        28,
        crate::types::ptq1_0::dequantize_row_ptq1_0,
        Some(crate::types::ptq1_0::vec_dot_ptq1_0),
    ),
];

pub fn lookup_type(id: u32) -> Option<&'static GgmlType> {
    TYPES.iter().find(|t| t.id == id)
}

pub fn type_by_name(name: &str) -> Option<&'static GgmlType> {
    TYPES.iter().find(|t| t.name.eq_ignore_ascii_case(name))
}

pub fn require_type(id: u32, tensor: &str) -> Result<&'static GgmlType> {
    lookup_type(id).ok_or_else(|| GgufError::UnknownGgmlType {
        id,
        tensor: tensor.to_string(),
    })
}

/// Dequantize a whole tensor (or a contiguous row-group) into `out`.
pub fn dequant_tensor(ty: &GgmlType, data: &[u8], out: &mut [f32]) -> Result<()> {
    if data.len() % ty.block_bytes != 0 {
        return Err(GgufError::UnevenBlocks {
            tensor: ty.name.to_string(),
            ty: ty.name.to_string(),
            block_bytes: ty.block_bytes,
            len: data.len(),
        });
    }
    let n_blocks = data.len() / ty.block_bytes;
    let need = n_blocks * ty.block_elems;
    if out.len() < need {
        return Err(GgufError::Truncated);
    }
    for i in 0..n_blocks {
        let src = &data[i * ty.block_bytes..(i + 1) * ty.block_bytes];
        let dst = &mut out[i * ty.block_elems..(i + 1) * ty.block_elems];
        (ty.dequant_row)(src, dst);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_exactly_the_m2_types() {
        let names: Vec<&str> = TYPES.iter().map(|t| t.name).collect();
        assert_eq!(
            names,
            [
                "F32", "F16", "Q4_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K", "BF16", "PQ2_0", "PTQ1_0"
            ]
        );
    }

    #[test]
    fn lookup_and_require_name_the_tensor() {
        assert!(lookup_type(TYPE_PQ2_0).is_some());
        assert!(lookup_type(42).is_none());
        let err = require_type(99, "blk.0.attn_q.weight").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("99"), "{msg}");
        assert!(msg.contains("blk.0.attn_q.weight"), "{msg}");
    }

    #[test]
    fn type_by_name_is_case_insensitive() {
        assert_eq!(type_by_name("pq2_0").unwrap().id, TYPE_PQ2_0);
        assert_eq!(type_by_name("PTQ1_0").unwrap().id, TYPE_PTQ1_0);
        assert!(type_by_name("Q2_K").is_none());
    }

    #[test]
    fn bytes_for_elems_rejects_partial_blocks() {
        let pq = lookup_type(TYPE_PQ2_0).unwrap();
        assert_eq!(pq.bytes_for_elems(128), Some(34));
        assert_eq!(pq.bytes_for_elems(256), Some(68));
        assert_eq!(pq.bytes_for_elems(127), None);
        let pt = lookup_type(TYPE_PTQ1_0).unwrap();
        assert_eq!(pt.bytes_for_elems(128), Some(28));
        assert_eq!(pt.bytes_for_elems(256), Some(56));
        assert_eq!(pt.bytes_for_elems(127), None);
        let f32t = lookup_type(TYPE_F32).unwrap();
        assert_eq!(f32t.bytes_for_elems(3), Some(12));
        let q8 = lookup_type(TYPE_Q8_0).unwrap();
        assert_eq!(q8.bytes_for_elems(32), Some(34));
        assert_eq!(q8.bytes_for_elems(31), None);
        let q4k = lookup_type(TYPE_Q4_K).unwrap();
        assert_eq!(q4k.bytes_for_elems(256), Some(144));
        assert_eq!(q4k.bytes_for_elems(255), None);
        let q6k = lookup_type(TYPE_Q6_K).unwrap();
        assert_eq!(q6k.bytes_for_elems(256), Some(210));
        assert_eq!(q6k.bytes_for_elems(128), None);
    }

    #[test]
    fn legacy_types_block_layout_unchanged() {
        let f32t = lookup_type(TYPE_F32).unwrap();
        assert_eq!(f32t.id, 0);
        assert_eq!((f32t.block_elems, f32t.block_bytes), (1, 4));
        let q8 = lookup_type(TYPE_Q8_0).unwrap();
        assert_eq!(q8.id, 8);
        assert_eq!((q8.block_elems, q8.block_bytes), (32, 34));
        let q4k = lookup_type(TYPE_Q4_K).unwrap();
        assert_eq!(q4k.id, 12);
        assert_eq!((q4k.block_elems, q4k.block_bytes), (256, 144));
        let q6k = lookup_type(TYPE_Q6_K).unwrap();
        assert_eq!(q6k.id, 14);
        assert_eq!((q6k.block_elems, q6k.block_bytes), (256, 210));
    }

    #[test]
    fn dequant_tensor_pq2_and_ptq1() {
        let mut pq = vec![0u8; 34];
        quantize_row_pq2_0(&[0.5f32; 128], &mut pq);
        let mut out = vec![0.0f32; 128];
        dequant_tensor(lookup_type(TYPE_PQ2_0).unwrap(), &pq, &mut out).unwrap();
        assert!(out.iter().all(|v| (v - 0.5).abs() < 0.05));

        let mut pt = vec![0u8; 28];
        pt[26..28].copy_from_slice(&crate::f16::f32_to_f16(1.0).to_le_bytes());
        out.fill(0.0);
        dequant_tensor(lookup_type(TYPE_PTQ1_0).unwrap(), &pt, &mut out).unwrap();
        assert!(out.iter().all(|v| (v + 1.0).abs() < 1e-3));
    }

    #[test]
    fn dequant_tensor_legacy_f32_q8_q4k_q6k() {
        let mut out1 = [0.0f32];
        dequant_tensor(
            lookup_type(TYPE_F32).unwrap(),
            &1.5f32.to_le_bytes(),
            &mut out1,
        )
        .unwrap();
        assert!((out1[0] - 1.5).abs() < 1e-6);

        let mut q8 = [0u8; 34];
        q8[0..2].copy_from_slice(&crate::f16::f32_to_f16(1.0).to_le_bytes());
        q8[2] = 5;
        q8[3] = (-3i8) as u8;
        let mut out8 = [0.0f32; 32];
        dequant_tensor(lookup_type(TYPE_Q8_0).unwrap(), &q8, &mut out8).unwrap();
        assert!((out8[0] - 5.0).abs() < 1e-3);
        assert!((out8[1] + 3.0).abs() < 1e-3);

        let q4k = [0u8; 144];
        let mut out4 = [1.0f32; 256];
        dequant_tensor(lookup_type(TYPE_Q4_K).unwrap(), &q4k, &mut out4).unwrap();
        assert!(out4.iter().all(|&v| v == 0.0));

        let q6k = [0u8; 210];
        let mut out6 = [1.0f32; 256];
        dequant_tensor(lookup_type(TYPE_Q6_K).unwrap(), &q6k, &mut out6).unwrap();
        assert!(out6.iter().all(|&v| v == 0.0));
    }
}
