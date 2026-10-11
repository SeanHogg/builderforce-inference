use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use crate::error::{GgufError, Result};
use crate::types::{lookup_type, require_type};

pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";
pub const GGUF_VERSION_MIN: u32 = 2;
pub const GGUF_VERSION_MAX: u32 = 3;
pub const DEFAULT_ALIGNMENT: u64 = 32;
/// Bytes we fetch on an HTTP range request before deciding we have the whole header.
pub const HEADER_HINT_BYTES: usize = 2 * 1024 * 1024;

const VT_UINT8: u32 = 0;
const VT_INT8: u32 = 1;
const VT_UINT16: u32 = 2;
const VT_INT16: u32 = 3;
const VT_UINT32: u32 = 4;
const VT_INT32: u32 = 5;
const VT_FLOAT32: u32 = 6;
const VT_BOOL: u32 = 7;
const VT_STRING: u32 = 8;
const VT_ARRAY: u32 = 9;
const VT_UINT64: u32 = 10;
const VT_INT64: u32 = 11;
const VT_FLOAT64: u32 = 12;

#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetaValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetaValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::String(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            MetaValue::U8(v) => Some(v as u64),
            MetaValue::U16(v) => Some(v as u64),
            MetaValue::U32(v) => Some(v as u64),
            MetaValue::U64(v) => Some(v),
            MetaValue::I8(v) if v >= 0 => Some(v as u64),
            MetaValue::I16(v) if v >= 0 => Some(v as u64),
            MetaValue::I32(v) if v >= 0 => Some(v as u64),
            MetaValue::I64(v) if v >= 0 => Some(v as u64),
            MetaValue::Bool(v) => Some(v as u64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            MetaValue::F32(v) => Some(v as f64),
            MetaValue::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }
    pub fn as_string_array(&self) -> Option<Vec<&str>> {
        match self {
            MetaValue::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for it in items {
                    out.push(it.as_str()?);
                }
                Some(out)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// Offset from the start of the tensor data region (already aligned).
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elems(&self) -> u64 {
        self.dims.iter().copied().product()
    }

    pub fn nbytes(&self) -> Result<u64> {
        let ty = require_type(self.ggml_type, &self.name)?;
        ty.bytes_for_elems(self.n_elems()).ok_or_else(|| GgufError::UnevenBlocks {
            tensor: self.name.clone(),
            ty: ty.name.to_string(),
            block_bytes: ty.block_bytes,
            len: self.n_elems() as usize,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ModelHeader {
    pub version: u32,
    pub alignment: u64,
    pub metadata: BTreeMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    /// Absolute file offset where tensor bytes begin.
    pub tensor_data_offset: u64,
}

impl ModelHeader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        Self::from_reader(file)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut c = Cursor::new(bytes);
        c.parse()
    }

    /// Parse from a reader. Stops after the tensor-info table; does not pull weights.
    ///
    /// Reads in 32 KiB chunks and re-parses until the header is complete, so `bfi fit`
    /// can size a file from the first range request without mapping the tensors.
    pub fn from_reader(mut r: impl Read) -> Result<Self> {
        let mut buf = Vec::new();
        let mut tmp = [0u8; 32 * 1024];
        loop {
            match Self::from_bytes(&buf) {
                Ok(h) => return Ok(h),
                Err(GgufError::Truncated) => {
                    let n = r.read(&mut tmp)?;
                    if n == 0 {
                        return Err(GgufError::Truncated);
                    }
                    buf.extend_from_slice(&tmp[..n]);
                }
                Err(e) => return Err(e),
            }
        }
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture").and_then(MetaValue::as_str)
    }

    pub fn name(&self) -> Option<&str> {
        self.metadata
            .get("general.name")
            .and_then(MetaValue::as_str)
            .or_else(|| self.metadata.get("general.basename").and_then(MetaValue::as_str))
    }

    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(MetaValue::as_u64)
    }

    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.metadata.get(key).and_then(MetaValue::as_f64)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(MetaValue::as_str)
    }

    pub fn arch_u64(&self, field: &str) -> Option<u64> {
        let arch = self.architecture()?;
        self.get_u64(&format!("{arch}.{field}"))
    }

    pub fn arch_f64(&self, field: &str) -> Option<f64> {
        let arch = self.architecture()?;
        self.get_f64(&format!("{arch}.{field}"))
    }

    /// Bytes of each tensor, in file order. Unknown types error with name + id.
    pub fn bytes_by_tensor(&self) -> Result<Vec<(String, u64)>> {
        let mut out = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            let _ = require_type(t.ggml_type, &t.name)?;
            out.push((t.name.clone(), t.nbytes()?));
        }
        Ok(out)
    }

    pub fn weight_bytes(&self) -> Result<u64> {
        Ok(self.bytes_by_tensor()?.into_iter().map(|(_, n)| n).sum())
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| GgufError::MissingTensor(name.to_string()))
    }
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(GgufError::Truncated);
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i8(&mut self) -> Result<i8> {
        Ok(self.u8()? as i8)
    }
    fn i16(&mut self) -> Result<i16> {
        Ok(self.u16()? as i16)
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    fn value(&mut self, ty: u32) -> Result<MetaValue> {
        Ok(match ty {
            VT_UINT8 => MetaValue::U8(self.u8()?),
            VT_INT8 => MetaValue::I8(self.i8()?),
            VT_UINT16 => MetaValue::U16(self.u16()?),
            VT_INT16 => MetaValue::I16(self.i16()?),
            VT_UINT32 => MetaValue::U32(self.u32()?),
            VT_INT32 => MetaValue::I32(self.i32()?),
            VT_FLOAT32 => MetaValue::F32(self.f32()?),
            VT_BOOL => MetaValue::Bool(self.u8()? != 0),
            VT_STRING => MetaValue::String(self.string()?),
            VT_ARRAY => {
                let elem = self.u32()?;
                let n = self.u64()? as usize;
                let mut items = Vec::with_capacity(n);
                for _ in 0..n {
                    items.push(self.value(elem)?);
                }
                MetaValue::Array(items)
            }
            VT_UINT64 => MetaValue::U64(self.u64()?),
            VT_INT64 => MetaValue::I64(self.i64()?),
            VT_FLOAT64 => MetaValue::F64(self.f64()?),
            other => return Err(GgufError::UnknownValueType(other)),
        })
    }

    fn parse(&mut self) -> Result<ModelHeader> {
        let magic = self.take(4)?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic);
        }
        let version = self.u32()?;
        if !(GGUF_VERSION_MIN..=GGUF_VERSION_MAX).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = self.u64()? as usize;
        let kv_count = self.u64()? as usize;

        let mut metadata = BTreeMap::new();
        for _ in 0..kv_count {
            let key = self.string()?;
            let ty = self.u32()?;
            let val = self.value(ty)?;
            metadata.insert(key, val);
        }

        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = self.string()?;
            let n_dims = self.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(self.u64()?);
            }
            let ggml_type = self.u32()?;
            let offset = self.u64()?;
            if lookup_type(ggml_type).is_none() {
                return Err(GgufError::UnknownGgmlType {
                    id: ggml_type,
                    tensor: name,
                });
            }
            tensors.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(MetaValue::as_u64)
            .filter(|&a| a > 0)
            .unwrap_or(DEFAULT_ALIGNMENT);
        let tensor_data_offset = align_up(self.pos as u64, alignment);

        Ok(ModelHeader {
            version,
            alignment,
            metadata,
            tensors,
            tensor_data_offset,
        })
    }
}

pub fn align_up(n: u64, align: u64) -> u64 {
    if align == 0 {
        n
    } else {
        n.div_ceil(align) * align
    }
}
