//! A host's GGUF byte chunks → whole tensors, handed to a backend's [`TensorSink`] as
//! [`TensorStream`] completes them. Each chunk is pushed and dropped; peak host memory is
//! the header plus the tensor being assembled — never the whole file.

use bfi_gguf::{ModelHeader, StreamedTensor, TensorStream};

use crate::errors::{error, gguf, EngineError, Kind};

/// Where the GGUF bytes come from: a browser `ReadableStream`, chunks a Node host pushes
/// from a file, … Owned by the glue crate.
#[allow(async_fn_in_trait)]
pub trait ByteSource {
    /// The next chunk, or `None` at the end of the file.
    async fn next_chunk(&mut self) -> Result<Option<Vec<u8>>, EngineError>;
}

/// Where a backend puts streamed tensors.
#[allow(async_fn_in_trait)]
pub trait TensorSink {
    /// Called once the header is known, then for every chunk after; `done` is the
    /// tensors that chunk completed (often none).
    async fn accept(&mut self, header: &ModelHeader, done: Vec<StreamedTensor>) -> Result<(), EngineError>;
}

/// Drain `source` into `sink`, reporting `(loaded, total)` bytes after each chunk.
pub async fn pump(
    source: &mut impl ByteSource,
    total_bytes: f64,
    on_progress: &mut dyn FnMut(f64, f64),
    sink: &mut impl TensorSink,
) -> Result<(), EngineError> {
    let mut stream = TensorStream::new();
    let mut loaded = 0f64;
    while let Some(bytes) = source.next_chunk().await? {
        loaded += bytes.len() as f64;
        let done = stream.push(&bytes).map_err(gguf)?;
        drop(bytes);
        if let Some(header) = stream.header() {
            sink.accept(header, done).await?;
        }
        on_progress(loaded, total_bytes);
    }
    let tail = stream.flush_header().map_err(gguf)?;
    let header = stream
        .header()
        .ok_or_else(|| error(Kind::LoadFailed, "empty GGUF stream"))?;
    sink.accept(header, tail).await?;
    stream.finish().map_err(|_| {
        error(
            Kind::LoadFailed,
            format!("stream ended before every tensor arrived ({loaded} of {total_bytes} bytes)"),
        )
    })
}
