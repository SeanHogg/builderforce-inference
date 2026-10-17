//! Where a [`crate::GgufFile`]'s bytes live.
//!
//! Native builds map the file (`memmap2`, zero-copy). wasm32 has no filesystem
//! or mmap, so the same reader works over bytes the host already holds. Tensor
//! data is always a borrowed `&[u8]` slice of the source; callers never care
//! which variant backs it.

use std::ops::Deref;

/// Immutable GGUF bytes: a memory map (native) or an owned buffer (any target).
pub enum ByteSource {
    #[cfg(not(target_arch = "wasm32"))]
    Mapped(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl ByteSource {
    /// Map `file` read-only.
    ///
    /// Safety contract (upheld by callers): GGUF files are immutable once
    /// downloaded; nothing writes the file while the map is alive. Windows:
    /// `FILE_SHARE_READ` is the default for `File::open`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn map(file: &std::fs::File) -> std::io::Result<Self> {
        let mmap = unsafe { memmap2::Mmap::map(file)? };
        Ok(Self::Mapped(mmap))
    }

    pub fn len(&self) -> usize {
        self.as_ref().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl AsRef<[u8]> for ByteSource {
    fn as_ref(&self) -> &[u8] {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            ByteSource::Mapped(m) => m,
            ByteSource::Owned(v) => v,
        }
    }
}

impl Deref for ByteSource {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_ref()
    }
}

impl From<Vec<u8>> for ByteSource {
    fn from(v: Vec<u8>) -> Self {
        Self::Owned(v)
    }
}
