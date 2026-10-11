//! Host-tensor helpers.
//!
//! Assumed bfi-core layout (ticket 2921 / PRD §5.2):
//!   `Tensor { shape: Vec<usize>, dtype: DType, storage: Storage }`
//!   `DType::F32`
//!   `Storage::F32(Vec<f32>)`
//!   `Tensor::from_f32(shape, data)` and `t.reshape(shape)` if those exist;
//!   otherwise the public fields above.
//!
//! If bfi-core lands with a different layout, this is the only file to patch.

use anyhow::{bail, ensure};
use bfi_core::{DType, Storage, Tensor};
use bfi_gguf::{lookup_type, TYPE_F32};

pub fn f32_tensor(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    debug_assert_eq!(
        shape.iter().copied().product::<usize>(),
        data.len(),
        "f32_tensor shape/len mismatch"
    );
    Tensor {
        shape,
        dtype: DType::F32,
        storage: Storage::F32(data),
    }
}

pub fn zeros_f32(shape: Vec<usize>) -> Tensor {
    let n: usize = shape.iter().copied().product();
    f32_tensor(shape, vec![0.0; n])
}

pub fn shape_of(t: &Tensor) -> &[usize] {
    t.shape.as_slice()
}

pub fn as_f32(t: &Tensor) -> anyhow::Result<&[f32]> {
    match &t.storage {
        Storage::F32(v) => Ok(v.as_slice()),
        _ => bail!("expected F32 storage, dtype={:?}", t.dtype),
    }
}

pub fn as_f32_mut(t: &mut Tensor) -> anyhow::Result<&mut [f32]> {
    match &mut t.storage {
        Storage::F32(v) => Ok(v.as_mut_slice()),
        _ => bail!("expected F32 storage, dtype={:?}", t.dtype),
    }
}

pub fn reshape(mut t: Tensor, shape: Vec<usize>) -> anyhow::Result<Tensor> {
    let old: usize = t.shape.iter().copied().product();
    let new: usize = shape.iter().copied().product();
    ensure!(
        old == new,
        "cannot reshape {:?} ({} elems) -> {:?} ({} elems)",
        t.shape,
        old,
        shape,
        new
    );
    t.shape = shape;
    Ok(t)
}

/// GGUF dims are innermost-first (GGML `ne[0]`). Logical row-major is reversed.
pub fn logical_shape(dims: &[u64]) -> Vec<usize> {
    dims.iter().rev().map(|&d| d as usize).collect()
}

pub fn f32_from_le_bytes(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    ensure!(
        bytes.len() % 4 == 0,
        "F32 tensor byte length {} is not a multiple of 4",
        bytes.len()
    );
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

pub fn is_f32(ggml_type: u32) -> bool {
    ggml_type == TYPE_F32 || lookup_type(ggml_type).map(|t| t.id == TYPE_F32).unwrap_or(false)
}

/// Copy `src` `[t, h, d]` into `dst` `[cap, h, d]` at row `offset`.
pub fn write_rows(dst: &mut Tensor, offset: usize, src: &Tensor) -> anyhow::Result<()> {
    let ds = shape_of(dst);
    let ss = shape_of(src);
    ensure!(ds.len() == 3 && ss.len() == 3, "KV tensors must be rank-3");
    ensure!(
        ds[1] == ss[1] && ds[2] == ss[2],
        "KV head layout mismatch dst={:?} src={:?}",
        ds,
        ss
    );
    let t = ss[0];
    let row = ss[1] * ss[2];
    ensure!(
        offset + t <= ds[0],
        "KV write overflow: offset={offset} t={t} cap={}",
        ds[0]
    );
    let src_s = as_f32(src)?;
    let dst_s = as_f32_mut(dst)?;
    let start = offset * row;
    dst_s[start..start + t * row].copy_from_slice(&src_s[..t * row]);
    Ok(())
}

/// Prefix view of a KV cache: `[cap, h, d]` → `[n, h, d]` (copied).
pub fn prefix_rows(t: &Tensor, n: usize) -> anyhow::Result<Tensor> {
    let s = shape_of(t);
    ensure!(s.len() == 3, "KV tensor must be rank-3, got {s:?}");
    ensure!(n <= s[0], "KV prefix n={n} exceeds cap={}", s[0]);
    let row = s[1] * s[2];
    let data = as_f32(t)?;
    Ok(f32_tensor(
        vec![n, s[1], s[2]],
        data[..n * row].to_vec(),
    ))
}
