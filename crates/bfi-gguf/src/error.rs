use std::io;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("not a GGUF file (bad magic)")]
    BadMagic,
    #[error("unsupported GGUF version {0} (need 2 or 3)")]
    UnsupportedVersion(u32),
    #[error("unexpected end of GGUF data")]
    Truncated,
    #[error("unknown metadata value type {0}")]
    UnknownValueType(u32),
    #[error("unknown GGML type id {id} on tensor {tensor}")]
    UnknownGgmlType { id: u32, tensor: String },
    #[error("unknown GGML type id {0}")]
    UnknownGgmlTypeId(u32),
    #[error("tensor {0} not found")]
    MissingTensor(String),
    #[error("tensor {tensor}: type {ty} block size {block_bytes} does not divide data length {len}")]
    UnevenBlocks {
        tensor: String,
        ty: String,
        block_bytes: usize,
        len: usize,
    },
    #[error("io: {0}")]
    Io(#[from] io::Error),
}

pub type Result<T> = std::result::Result<T, GgufError>;
