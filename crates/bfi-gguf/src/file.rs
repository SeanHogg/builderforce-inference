use std::path::{Path, PathBuf};

use crate::error::{GgufError, Result};
use crate::header::{ModelHeader, TensorInfo};
use crate::source::ByteSource;
use crate::types::{dequant_tensor, require_type, GgmlType};

/// A parsed GGUF over a [`ByteSource`]. The header is parsed once; tensor bytes
/// are borrowed from the source (a memory map natively, an owned buffer on wasm32).
pub struct GgufFile {
    pub path: PathBuf,
    header: ModelHeader,
    bytes: ByteSource,
}

impl GgufFile {
    /// Open a file. Native: zero-copy `mmap`. wasm32: there is no filesystem in
    /// the browser, so this only works under WASI-like hosts; browser callers use
    /// [`GgufFile::from_bytes`] or [`crate::TensorStream`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        #[cfg(not(target_arch = "wasm32"))]
        let bytes = ByteSource::map(&std::fs::File::open(&path)?)?;
        #[cfg(target_arch = "wasm32")]
        let bytes = ByteSource::Owned(std::fs::read(&path)?);
        Self::from_source(path, bytes)
    }

    /// Parse GGUF bytes the caller already holds (any target).
    pub fn from_bytes(name: impl Into<PathBuf>, bytes: Vec<u8>) -> Result<Self> {
        Self::from_source(name.into(), ByteSource::Owned(bytes))
    }

    fn from_source(path: PathBuf, bytes: ByteSource) -> Result<Self> {
        let header = ModelHeader::from_bytes(&bytes)?;
        Ok(Self {
            path,
            header,
            bytes,
        })
    }

    pub fn header(&self) -> &ModelHeader {
        &self.header
    }

    pub fn tensor_info(&self, name: &str) -> Result<&TensorInfo> {
        self.header.tensor(name)
    }

    pub fn tensor_type(&self, name: &str) -> Result<&'static GgmlType> {
        let t = self.header.tensor(name)?;
        require_type(t.ggml_type, name)
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let t = self.header.tensor(name)?;
        let nbytes = t.nbytes()? as usize;
        let start = (self.header.tensor_data_offset + t.offset) as usize;
        let end = start.checked_add(nbytes).ok_or(GgufError::Truncated)?;
        if end > self.bytes.len() {
            return Err(GgufError::Truncated);
        }
        Ok(&self.bytes[start..end])
    }

    pub fn dequant_f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.header.tensor(name)?;
        let ty = require_type(t.ggml_type, name)?;
        let bytes = self.tensor_bytes(name)?;
        let n = t.n_elems() as usize;
        let mut out = vec![0.0f32; n];
        dequant_tensor(ty, bytes, &mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::{write_gguf, TensorWrite};
    use crate::{lookup_type, MetaValue, TYPE_F32};
    use std::collections::BTreeMap;

    #[test]
    fn from_bytes_reads_tensors_without_a_file() {
        let mut meta = BTreeMap::new();
        meta.insert(
            "general.architecture".into(),
            MetaValue::String("qwen3".into()),
        );
        let bytes = write_gguf(
            &meta,
            &[TensorWrite {
                name: "w".into(),
                dims: vec![2],
                ty: lookup_type(TYPE_F32).unwrap(),
                data: [1.5f32.to_le_bytes(), (-2.0f32).to_le_bytes()].concat(),
            }],
        )
        .unwrap();
        let f = GgufFile::from_bytes("mem.gguf", bytes).unwrap();
        assert_eq!(f.header().architecture(), Some("qwen3"));
        assert_eq!(f.dequant_f32("w").unwrap(), vec![1.5, -2.0]);
        assert_eq!(f.tensor_bytes("w").unwrap().len(), 8);
    }
}
