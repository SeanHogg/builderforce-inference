use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::error::{GgufError, Result};
use crate::header::{ModelHeader, TensorInfo};
use crate::types::{dequant_tensor, require_type, GgmlType};

/// Memory-mapped GGUF. The header is parsed once; tensor bytes are borrowed from the map.
pub struct GgufFile {
    pub path: PathBuf,
    header: ModelHeader,
    mmap: Mmap,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        // Safety: we never write the file while the map is alive; GGUF files are immutable
        // once downloaded. Windows: FILE_SHARE_READ is the default for File::open.
        let mmap = unsafe { Mmap::map(&file)? };
        let header = ModelHeader::from_bytes(&mmap)?;
        Ok(Self { path, header, mmap })
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
        if end > self.mmap.len() {
            return Err(GgufError::Truncated);
        }
        Ok(&self.mmap[start..end])
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
