use crate::DType;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("expected F32 tensor, got {0:?}")]
    ExpectedF32(DType),
    #[error("{op}: {msg}")]
    Shape { op: &'static str, msg: String },
    #[error("unknown GGML type id {0}")]
    UnknownType(u32),
    #[error("row id {id} out of range (n={n})")]
    RowId { id: i32, n: usize },
}
