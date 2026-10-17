//! Push parser: GGUF bytes arrive in arbitrary chunks (a browser
//! `ReadableStream`, an HTTP body) and come out as whole tensors, in file order.
//!
//! Peak memory is the header plus the one tensor being assembled — never the
//! whole file. The browser engine hands each finished tensor straight to its
//! GPU buffer and drops the host copy.

use crate::error::{GgufError, Result};
use crate::header::{ModelHeader, TensorInfo};

/// Smallest growth between header parse attempts. Re-parsing a multi-MB
/// tokenizer header on every 64 KiB chunk would be quadratic.
const HEADER_RETRY_MIN_GROWTH: usize = if cfg!(test) { 16 } else { 256 * 1024 };

/// One tensor's raw (still packed) bytes, exactly `info.nbytes()` long.
#[derive(Debug)]
pub struct StreamedTensor {
    pub info: TensorInfo,
    pub bytes: Vec<u8>,
}

/// Incremental GGUF reader. Feed chunks with [`TensorStream::push`].
#[derive(Default)]
pub struct TensorStream {
    header: Option<ModelHeader>,
    head_buf: Vec<u8>,
    next_header_try: usize,
    /// Tensor indices sorted by data offset.
    order: Vec<usize>,
    next: usize,
    /// Absolute file position of the next byte `push` will see.
    pos: u64,
    cur: Vec<u8>,
}

impl TensorStream {
    pub fn new() -> Self {
        Self::default()
    }

    /// The header, once enough bytes have arrived to parse it.
    pub fn header(&self) -> Option<&ModelHeader> {
        self.header.as_ref()
    }

    /// Bytes consumed so far (header + tensor data + padding).
    pub fn consumed(&self) -> u64 {
        self.pos + self.head_buf.len() as u64
    }

    /// Every tensor in the header has been emitted.
    pub fn is_complete(&self) -> bool {
        self.header.is_some() && self.next >= self.order.len()
    }

    /// Error unless every tensor arrived (call when the source ends).
    pub fn finish(&self) -> Result<()> {
        if self.is_complete() {
            Ok(())
        } else {
            Err(GgufError::Truncated)
        }
    }

    /// Feed the next chunk. Returns the tensors this chunk completed, in file order.
    /// Header errors (bad magic, unknown GGML type naming the tensor) surface from
    /// the call that first parses the complete header: this one, or
    /// [`TensorStream::flush_header`] at end of source.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<StreamedTensor>> {
        let mut done = Vec::new();
        if self.header.is_none() {
            self.head_buf.extend_from_slice(chunk);
            if self.head_buf.len() < self.next_header_try {
                return Ok(done);
            }
            match ModelHeader::from_bytes(&self.head_buf) {
                Ok(h) => {
                    let buf = std::mem::take(&mut self.head_buf);
                    self.start(h)?;
                    self.consume(&buf, &mut done)?;
                }
                Err(GgufError::Truncated) => {
                    self.next_header_try = self.head_buf.len() + HEADER_RETRY_MIN_GROWTH;
                }
                Err(e) => return Err(e),
            }
            return Ok(done);
        }
        self.consume(chunk, &mut done)?;
        Ok(done)
    }

    /// Parse whatever is buffered even if under the retry threshold (end of source).
    pub fn flush_header(&mut self) -> Result<Vec<StreamedTensor>> {
        self.next_header_try = 0;
        self.push(&[])
    }

    fn start(&mut self, h: ModelHeader) -> Result<()> {
        for t in &h.tensors {
            t.nbytes()?; // every type must be known before any byte is placed
        }
        let mut order: Vec<usize> = (0..h.tensors.len()).collect();
        order.sort_by_key(|&i| h.tensors[i].offset);
        self.order = order;
        self.next = 0;
        self.pos = 0;
        self.header = Some(h);
        Ok(())
    }

    fn consume(&mut self, mut chunk: &[u8], done: &mut Vec<StreamedTensor>) -> Result<()> {
        let h = self.header.as_ref().expect("header parsed");
        while !chunk.is_empty() && self.next < self.order.len() {
            let info = &h.tensors[self.order[self.next]];
            let start = h.tensor_data_offset + info.offset;
            let end = start + info.nbytes()?;
            if self.pos < start {
                // Header tail / alignment padding before this tensor.
                let skip = ((start - self.pos) as usize).min(chunk.len());
                chunk = &chunk[skip..];
                self.pos += skip as u64;
                continue;
            }
            if self.cur.is_empty() {
                self.cur.reserve_exact((end - start) as usize);
            }
            let take = ((end - self.pos) as usize).min(chunk.len());
            self.cur.extend_from_slice(&chunk[..take]);
            chunk = &chunk[take..];
            self.pos += take as u64;
            if self.pos == end {
                done.push(StreamedTensor {
                    info: info.clone(),
                    bytes: std::mem::take(&mut self.cur),
                });
                self.next += 1;
            }
        }
        self.pos += chunk.len() as u64; // trailing bytes after the last tensor
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::{write_gguf, TensorWrite};
    use crate::{lookup_type, quantize_row_q1_0, MetaValue, TYPE_F32, TYPE_Q1_0};
    use std::collections::BTreeMap;

    fn sample() -> Vec<u8> {
        let mut meta = BTreeMap::new();
        meta.insert(
            "general.architecture".into(),
            MetaValue::String("qwen3".into()),
        );
        meta.insert("general.alignment".into(), MetaValue::U32(64));
        let mut q = vec![0u8; 36];
        let row: Vec<f32> = (0..128).map(|i| (i as f32) - 64.0).collect();
        quantize_row_q1_0(&row, &mut q[..18]);
        quantize_row_q1_0(&row, &mut q[18..]);
        write_gguf(
            &meta,
            &[
                TensorWrite {
                    name: "a".into(),
                    dims: vec![3],
                    ty: lookup_type(TYPE_F32).unwrap(),
                    data: [1.0f32, 2.0, 3.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
                },
                TensorWrite {
                    name: "b".into(),
                    dims: vec![128, 2],
                    ty: lookup_type(TYPE_Q1_0).unwrap(),
                    data: q,
                },
            ],
        )
        .unwrap()
    }

    fn run(bytes: &[u8], chunk: usize) -> Vec<StreamedTensor> {
        let mut s = TensorStream::new();
        let mut out = Vec::new();
        for c in bytes.chunks(chunk) {
            out.extend(s.push(c).unwrap());
        }
        out.extend(s.flush_header().unwrap());
        s.finish().unwrap();
        out
    }

    #[test]
    fn any_chunking_yields_the_same_tensors_as_the_file_reader() {
        let bytes = sample();
        let file = crate::GgufFile::from_bytes("m", bytes.clone()).unwrap();
        for chunk in [1usize, 7, 64, 1000, bytes.len()] {
            let got = run(&bytes, chunk);
            assert_eq!(got.len(), 2, "chunk {chunk}");
            assert_eq!(got[0].info.name, "a");
            assert_eq!(got[0].bytes, file.tensor_bytes("a").unwrap());
            assert_eq!(got[1].info.name, "b");
            assert_eq!(got[1].bytes.len(), 36);
            assert_eq!(got[1].bytes, file.tensor_bytes("b").unwrap());
        }
    }

    #[test]
    fn truncated_source_is_an_error() {
        let bytes = sample();
        let mut s = TensorStream::new();
        s.push(&bytes[..bytes.len() - 5]).unwrap();
        s.flush_header().unwrap();
        assert!(s.finish().is_err());
        assert!(!s.is_complete());
    }

    #[test]
    fn unknown_type_surfaces_with_tensor_name() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&crate::GGUF_MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        let name = b"blk.0.ffn_up.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name);
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&128u64.to_le_bytes());
        buf.extend_from_slice(&77u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        // Surfaces from whichever call first parses the full header — here `push`.
        let mut s = TensorStream::new();
        let err = s.push(&buf).err().or_else(|| s.flush_header().err()).unwrap();
        let msg = err.to_string();
        assert!(msg.contains("77") && msg.contains("blk.0.ffn_up.weight"), "{msg}");
    }
}
