//! Minimal GGUF writer so tests can build files without a real model.

use std::collections::BTreeMap;
use std::io::{self, Write};

use crate::header::{align_up, MetaValue, DEFAULT_ALIGNMENT, GGUF_MAGIC};
use crate::types::GgmlType;

pub struct TensorWrite {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: &'static GgmlType,
    pub data: Vec<u8>,
}

pub fn write_gguf(
    metadata: &BTreeMap<String, MetaValue>,
    tensors: &[TensorWrite],
) -> io::Result<Vec<u8>> {
    let alignment = metadata
        .get("general.alignment")
        .and_then(MetaValue::as_u64)
        .filter(|&a| a > 0)
        .unwrap_or(DEFAULT_ALIGNMENT);

    let mut body = Vec::new();
    body.extend_from_slice(&GGUF_MAGIC);
    body.extend_from_slice(&3u32.to_le_bytes());
    body.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    body.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (k, v) in metadata {
        write_string(&mut body, k);
        write_value(&mut body, v);
    }

    let mut offset = 0u64;
    let mut offsets = Vec::with_capacity(tensors.len());
    for t in tensors {
        offset = align_up(offset, alignment);
        offsets.push(offset);
        offset += t.data.len() as u64;
    }
    for (t, off) in tensors.iter().zip(&offsets) {
        write_string(&mut body, &t.name);
        body.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            body.extend_from_slice(&d.to_le_bytes());
        }
        body.extend_from_slice(&t.ty.id.to_le_bytes());
        body.extend_from_slice(&off.to_le_bytes());
    }

    let data_start = align_up(body.len() as u64, alignment);
    body.resize(data_start as usize, 0);
    for (t, off) in tensors.iter().zip(&offsets) {
        let abs = (data_start + off) as usize;
        if body.len() < abs {
            body.resize(abs, 0);
        }
        body.extend_from_slice(&t.data);
    }
    Ok(body)
}

fn write_string(w: &mut Vec<u8>, s: &str) {
    w.extend_from_slice(&(s.len() as u64).to_le_bytes());
    w.extend_from_slice(s.as_bytes());
}

fn value_type(v: &MetaValue) -> u32 {
    match v {
        MetaValue::U8(_) => 0,
        MetaValue::I8(_) => 1,
        MetaValue::U16(_) => 2,
        MetaValue::I16(_) => 3,
        MetaValue::U32(_) => 4,
        MetaValue::I32(_) => 5,
        MetaValue::F32(_) => 6,
        MetaValue::Bool(_) => 7,
        MetaValue::String(_) => 8,
        MetaValue::Array(_) => 9,
        MetaValue::U64(_) => 10,
        MetaValue::I64(_) => 11,
        MetaValue::F64(_) => 12,
    }
}

fn write_payload(w: &mut Vec<u8>, v: &MetaValue) {
    match v {
        MetaValue::U8(x) => w.push(*x),
        MetaValue::I8(x) => w.push(*x as u8),
        MetaValue::U16(x) => w.extend_from_slice(&x.to_le_bytes()),
        MetaValue::I16(x) => w.extend_from_slice(&x.to_le_bytes()),
        MetaValue::U32(x) => w.extend_from_slice(&x.to_le_bytes()),
        MetaValue::I32(x) => w.extend_from_slice(&x.to_le_bytes()),
        MetaValue::F32(x) => w.extend_from_slice(&x.to_le_bytes()),
        MetaValue::Bool(x) => w.push(*x as u8),
        MetaValue::String(s) => write_string(w, s),
        MetaValue::Array(items) => {
            let elem_ty = items.first().map(value_type).unwrap_or(0);
            w.extend_from_slice(&elem_ty.to_le_bytes());
            w.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for it in items {
                write_payload(w, it);
            }
        }
        MetaValue::U64(x) => w.extend_from_slice(&x.to_le_bytes()),
        MetaValue::I64(x) => w.extend_from_slice(&x.to_le_bytes()),
        MetaValue::F64(x) => w.extend_from_slice(&x.to_le_bytes()),
    }
}

fn write_value(w: &mut Vec<u8>, v: &MetaValue) {
    w.extend_from_slice(&value_type(v).to_le_bytes());
    write_payload(w, v);
}

#[allow(dead_code)]
pub fn write_to(path: impl AsRef<std::path::Path>, bytes: &[u8]) -> io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(bytes)
}
