//! GGUF v2/v3 reader and GGML type registry.
//!
//! Type layouts (including PrismML `PQ2_0` / `PTQ1_0`) live in [`types`] so
//! `bfi fit` can size a file from the header alone. Adding a GGML type means
//! adding one table row in `types/mod.rs` and one file in `types/`.

mod error;
mod f16;
mod file;
mod header;
mod types;
pub mod writer;

pub use error::{GgufError, Result};
pub use f16::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16};
pub use file::GgufFile;
pub use header::{
    align_up, MetaValue, ModelHeader, TensorInfo, DEFAULT_ALIGNMENT, GGUF_MAGIC, HEADER_HINT_BYTES,
};
pub use types::{
    dequant_bf16, dequant_f16, dequant_f32, dequant_q4_0, dequant_q4_k, dequant_q5_k, dequant_q6_k,
    dequant_q8_0, dequant_tensor, dequantize_row_pq2_0, dequantize_row_ptq1_0, lookup_type,
    quantize_row_pq2_0, require_type, type_by_name, vec_dot_pq2_0, vec_dot_ptq1_0, vec_dot_q4_0,
    vec_dot_q8_0, GgmlType, TYPE_BF16, TYPE_F16, TYPE_F32, TYPE_PQ2_0, TYPE_PTQ1_0, TYPE_Q4_0,
    TYPE_Q4_K, TYPE_Q5_K, TYPE_Q6_K, TYPE_Q8_0, TYPES,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::{write_gguf, TensorWrite};
    use std::collections::BTreeMap;
    use std::io::{self, Read, Write};

    fn meta_str(s: &str) -> MetaValue {
        MetaValue::String(s.to_string())
    }

    #[test]
    fn roundtrip_header_and_pq2_tensor() {
        let mut meta = BTreeMap::new();
        meta.insert("general.architecture".into(), meta_str("qwen3"));
        meta.insert("general.name".into(), meta_str("test"));
        meta.insert("qwen3.block_count".into(), MetaValue::U32(1));
        meta.insert("qwen3.embedding_length".into(), MetaValue::U32(128));

        let ty = lookup_type(TYPE_PQ2_0).unwrap();
        assert_eq!(ty.block_bytes, 34);
        assert_eq!(ty.block_elems, 128);

        let mut data = vec![0u8; 34];
        quantize_row_pq2_0(&[0.5f32; 128], &mut data);

        let bytes = write_gguf(
            &meta,
            &[TensorWrite {
                name: "token_embd.weight".into(),
                dims: vec![128, 1],
                ty,
                data,
            }],
        )
        .unwrap();

        let header = ModelHeader::from_bytes(&bytes).unwrap();
        assert_eq!(header.architecture(), Some("qwen3"));
        assert_eq!(header.version, 3);
        assert_eq!(header.tensors.len(), 1);
        assert_eq!(header.tensors[0].name, "token_embd.weight");
        assert_eq!(header.tensors[0].ggml_type, TYPE_PQ2_0);
        let sized = header.bytes_by_tensor().unwrap();
        assert_eq!(sized, vec![("token_embd.weight".into(), 34)]);
        assert_eq!(header.weight_bytes().unwrap(), 34);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.gguf");
        std::fs::File::create(&path).unwrap().write_all(&bytes).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let deq = file.dequant_f32("token_embd.weight").unwrap();
        assert_eq!(deq.len(), 128);
        assert!(deq.iter().all(|v| (v - 0.5).abs() < 0.05));
        let raw = file.tensor_bytes("token_embd.weight").unwrap();
        assert_eq!(raw.len(), 34);
    }

    #[test]
    fn unknown_type_names_tensor() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        buf.extend_from_slice(&0u64.to_le_bytes()); // 0 kv
        let name = b"blk.0.attn_q.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name);
        buf.extend_from_slice(&1u32.to_le_bytes()); // 1 dim
        buf.extend_from_slice(&128u64.to_le_bytes());
        buf.extend_from_slice(&99u32.to_le_bytes()); // unknown type
        buf.extend_from_slice(&0u64.to_le_bytes());
        let err = ModelHeader::from_bytes(&buf).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("99"), "{msg}");
        assert!(msg.contains("blk.0.attn_q.weight"), "{msg}");
    }

    #[test]
    fn type_registry_pq2_ptq1() {
        let pq = lookup_type(TYPE_PQ2_0).unwrap();
        assert_eq!(pq.name, "PQ2_0");
        assert_eq!(pq.block_elems, 128);
        assert_eq!(pq.block_bytes, 34);
        let pt = lookup_type(TYPE_PTQ1_0).unwrap();
        assert_eq!(pt.name, "PTQ1_0");
        assert_eq!(pt.block_elems, 128);
        assert_eq!(pt.block_bytes, 28);
        assert!(lookup_type(42).is_none());
    }

    #[test]
    fn metadata_arrays_and_alignment() {
        let mut meta = BTreeMap::new();
        meta.insert("general.architecture".into(), meta_str("qwen3"));
        meta.insert(
            "tokenizer.ggml.tokens".into(),
            MetaValue::Array(vec![meta_str("a"), meta_str("b")]),
        );
        meta.insert("general.alignment".into(), MetaValue::U32(32));
        let bytes = write_gguf(&meta, &[]).unwrap();
        let h = ModelHeader::from_bytes(&bytes).unwrap();
        let toks = h.get("tokenizer.ggml.tokens").unwrap().as_string_array().unwrap();
        assert_eq!(toks, vec!["a", "b"]);
        assert_eq!(h.alignment, 32);
        assert_eq!(h.tensor_data_offset % 32, 0);
    }

    #[test]
    fn all_metadata_value_types_roundtrip() {
        let mut meta = BTreeMap::new();
        meta.insert("k.u8".into(), MetaValue::U8(7));
        meta.insert("k.i8".into(), MetaValue::I8(-3));
        meta.insert("k.u16".into(), MetaValue::U16(400));
        meta.insert("k.i16".into(), MetaValue::I16(-400));
        meta.insert("k.u32".into(), MetaValue::U32(1_000_000));
        meta.insert("k.i32".into(), MetaValue::I32(-1_000_000));
        meta.insert("k.f32".into(), MetaValue::F32(1.5));
        meta.insert("k.bool".into(), MetaValue::Bool(true));
        meta.insert("k.str".into(), meta_str("hi"));
        meta.insert("k.u64".into(), MetaValue::U64(1u64 << 40));
        meta.insert("k.i64".into(), MetaValue::I64(-(1i64 << 40)));
        meta.insert("k.f64".into(), MetaValue::F64(2.25));
        meta.insert(
            "k.arr.u32".into(),
            MetaValue::Array(vec![MetaValue::U32(1), MetaValue::U32(2)]),
        );
        meta.insert(
            "k.arr.f32".into(),
            MetaValue::Array(vec![MetaValue::F32(0.5), MetaValue::F32(1.5)]),
        );
        meta.insert(
            "k.arr.bool".into(),
            MetaValue::Array(vec![MetaValue::Bool(true), MetaValue::Bool(false)]),
        );
        meta.insert("k.arr.empty".into(), MetaValue::Array(vec![]));
        let bytes = write_gguf(&meta, &[]).unwrap();
        let h = ModelHeader::from_bytes(&bytes).unwrap();
        assert_eq!(h.get("k.u8"), Some(&MetaValue::U8(7)));
        assert_eq!(h.get("k.i8"), Some(&MetaValue::I8(-3)));
        assert_eq!(h.get("k.u16"), Some(&MetaValue::U16(400)));
        assert_eq!(h.get("k.i16"), Some(&MetaValue::I16(-400)));
        assert_eq!(h.get("k.u32"), Some(&MetaValue::U32(1_000_000)));
        assert_eq!(h.get("k.i32"), Some(&MetaValue::I32(-1_000_000)));
        assert_eq!(h.get("k.f32"), Some(&MetaValue::F32(1.5)));
        assert_eq!(h.get("k.bool"), Some(&MetaValue::Bool(true)));
        assert_eq!(h.get_str("k.str"), Some("hi"));
        assert_eq!(h.get("k.u64"), Some(&MetaValue::U64(1u64 << 40)));
        assert_eq!(h.get("k.i64"), Some(&MetaValue::I64(-(1i64 << 40))));
        assert_eq!(h.get("k.f64"), Some(&MetaValue::F64(2.25)));
        assert_eq!(
            h.get("k.arr.u32"),
            Some(&MetaValue::Array(vec![MetaValue::U32(1), MetaValue::U32(2)]))
        );
        match h.get("k.arr.f32") {
            Some(MetaValue::Array(v)) => {
                assert_eq!(v.len(), 2);
                assert!((v[0].as_f64().unwrap() - 0.5).abs() < 1e-6);
            }
            other => panic!("bad arr.f32 {other:?}"),
        }
        assert_eq!(
            h.get("k.arr.bool"),
            Some(&MetaValue::Array(vec![
                MetaValue::Bool(true),
                MetaValue::Bool(false)
            ]))
        );
        assert_eq!(h.get("k.arr.empty"), Some(&MetaValue::Array(vec![])));
    }

    #[test]
    fn parses_gguf_v2() {
        let mut meta = BTreeMap::new();
        meta.insert("general.architecture".into(), meta_str("qwen3"));
        let mut bytes = write_gguf(&meta, &[]).unwrap();
        bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
        let h = ModelHeader::from_bytes(&bytes).unwrap();
        assert_eq!(h.version, 2);
        assert_eq!(h.architecture(), Some("qwen3"));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        match ModelHeader::from_bytes(&buf) {
            Err(GgufError::UnsupportedVersion(1)) => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn custom_alignment_and_tensor_padding() {
        let mut meta = BTreeMap::new();
        meta.insert("general.architecture".into(), meta_str("qwen3"));
        meta.insert("general.alignment".into(), MetaValue::U32(64));
        let f32t = lookup_type(TYPE_F32).unwrap();
        let bytes = write_gguf(
            &meta,
            &[
                TensorWrite {
                    name: "a".into(),
                    dims: vec![1],
                    ty: f32t,
                    data: 1.0f32.to_le_bytes().to_vec(),
                },
                TensorWrite {
                    name: "b".into(),
                    dims: vec![1],
                    ty: f32t,
                    data: 2.0f32.to_le_bytes().to_vec(),
                },
            ],
        )
        .unwrap();
        let h = ModelHeader::from_bytes(&bytes).unwrap();
        assert_eq!(h.alignment, 64);
        assert_eq!(h.tensor_data_offset % 64, 0);
        assert_eq!(h.tensors[0].offset, 0);
        assert_eq!(h.tensors[1].offset % 64, 0);
        let sized = h.bytes_by_tensor().unwrap();
        assert_eq!(sized, vec![("a".into(), 4), ("b".into(), 4)]);
    }

    /// Reader that errors if more than `allow` bytes are pulled — proves
    /// `from_reader` stops after the tensor-info table instead of slurping weights.
    struct CapRead<'a> {
        data: &'a [u8],
        pos: usize,
        allow: usize,
    }

    impl Read for CapRead<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.pos >= self.allow {
                return Err(io::Error::new(io::ErrorKind::Other, "read past header"));
            }
            let n = buf
                .len()
                .min(self.allow - self.pos)
                .min(self.data.len().saturating_sub(self.pos));
            buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            Ok(n)
        }
    }

    #[test]
    fn from_reader_does_not_need_weights() {
        let mut meta = BTreeMap::new();
        meta.insert("general.architecture".into(), meta_str("qwen3"));
        let ty = lookup_type(TYPE_F32).unwrap();
        let mut bytes = write_gguf(
            &meta,
            &[TensorWrite {
                name: "w".into(),
                dims: vec![1],
                ty,
                data: 3.0f32.to_le_bytes().to_vec(),
            }],
        )
        .unwrap();
        let header = ModelHeader::from_bytes(&bytes).unwrap();
        let allow = (header.tensor_data_offset as usize).max(64) + 32 * 1024;
        bytes.extend(vec![0xAAu8; 1_000_000]);
        let h = ModelHeader::from_reader(CapRead {
            data: &bytes,
            pos: 0,
            allow,
        })
        .unwrap();
        assert_eq!(h.tensors[0].name, "w");
        assert_eq!(h.bytes_by_tensor().unwrap()[0].1, 4);
    }

    #[test]
    fn bytes_by_tensor_mixed_types() {
        let mut meta = BTreeMap::new();
        meta.insert("general.architecture".into(), meta_str("qwen3"));
        let mut pq = vec![0u8; 34];
        quantize_row_pq2_0(&[0.25f32; 128], &mut pq);
        let mut pt = vec![0u8; 28];
        pt[26..28].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        let bytes = write_gguf(
            &meta,
            &[
                TensorWrite {
                    name: "tok".into(),
                    dims: vec![128],
                    ty: lookup_type(TYPE_PQ2_0).unwrap(),
                    data: pq,
                },
                TensorWrite {
                    name: "attn".into(),
                    dims: vec![128],
                    ty: lookup_type(TYPE_PTQ1_0).unwrap(),
                    data: pt,
                },
            ],
        )
        .unwrap();
        let h = ModelHeader::from_bytes(&bytes).unwrap();
        assert_eq!(
            h.bytes_by_tensor().unwrap(),
            vec![("tok".into(), 34), ("attn".into(), 28)]
        );
        assert_eq!(h.weight_bytes().unwrap(), 62);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixed.gguf");
        std::fs::File::create(&path).unwrap().write_all(&bytes).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let tok = file.dequant_f32("tok").unwrap();
        assert_eq!(tok.len(), 128);
        assert!(tok.iter().all(|v| (v - 0.25).abs() < 0.05));
        let attn = file.dequant_f32("attn").unwrap();
        assert_eq!(attn.len(), 128);
        assert!(attn.iter().all(|v| (v + 1.0).abs() < 1e-3));
    }

    #[test]
    fn bad_magic() {
        let err = ModelHeader::from_bytes(b"XXXX").unwrap_err();
        assert!(matches!(err, GgufError::BadMagic));
    }
}
