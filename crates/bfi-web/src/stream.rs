//! The host's `ReadableStream<Uint8Array>` → whole GGUF tensors, handed to a backend's
//! [`TensorSink`] as [`TensorStream`] completes them. Each chunk is copied into wasm
//! memory, pushed, and dropped; peak host memory is the header plus the tensor being
//! assembled — never the whole file.

use bfi_gguf::{ModelHeader, StreamedTensor, TensorStream};
use js_sys::{Function, Object, Uint8Array};
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

use crate::errors::{error, gguf, Kind};
use crate::js;

/// Where a backend puts streamed tensors.
pub trait TensorSink {
    /// Called once the header is known, then for every chunk after; `done` is the
    /// tensors that chunk completed (often none).
    async fn accept(&mut self, header: &ModelHeader, done: Vec<StreamedTensor>) -> Result<(), JsValue>;
}

/// Drain `chunks` into `sink`, reporting `(loaded, total)` bytes after each chunk.
pub async fn pump(
    chunks: &Object,
    total_bytes: f64,
    on_progress: Option<&Function>,
    sink: &mut impl TensorSink,
) -> Result<(), JsValue> {
    let reader = js::call(chunks, "getReader", &[])
        .map_err(|e| error(Kind::LoadFailed, format!("chunks is not a ReadableStream: {e:?}")))?;
    let mut stream = TensorStream::new();
    let mut loaded = 0f64;
    let result = async {
        loop {
            let read = js::call(&reader, "read", &[])
                .map_err(|e| error(Kind::LoadFailed, format!("reader.read threw: {e:?}")))?;
            let res = JsFuture::from(js_sys::Promise::from(read))
                .await
                .map_err(|e| error(Kind::LoadFailed, format!("stream errored: {e:?}")))?;
            if js::get(&res, "done").as_bool() == Some(true) {
                break;
            }
            let bytes = Uint8Array::new(&js::get(&res, "value")).to_vec();
            loaded += bytes.len() as f64;
            let done = stream.push(&bytes).map_err(gguf)?;
            drop(bytes);
            if let Some(header) = stream.header() {
                sink.accept(header, done).await?;
            }
            if let Some(cb) = on_progress {
                let _ = cb.call2(&JsValue::NULL, &JsValue::from_f64(loaded), &JsValue::from_f64(total_bytes));
            }
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
    .await;
    let _ = js::call(&reader, "releaseLock", &[]);
    result
}
