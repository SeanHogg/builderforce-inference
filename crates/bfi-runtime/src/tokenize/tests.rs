//! Tokenizer unit tests.

use super::*;
use bfi_gguf::{writer::write_gguf, MetaValue, ModelHeader};
use std::collections::BTreeMap;

fn tiny() -> Tokenizer {
    Tokenizer::from_parts(
        ["h", "e", "l", "o", "he", "ll"]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
        ["h e", "l l"].into_iter().map(|s| s.to_string()).collect(),
    )
}

#[test]
fn roundtrip_hello() {
    let tok = tiny();
    let ids = tok.encode("hello", false).unwrap();
    assert!(!ids.is_empty());
    assert_eq!(tok.decode(&ids).unwrap(), "hello");
    assert_eq!(tok.n_vocab(), 6);
    assert!(tok.bos().is_none());
    assert!(tok.eos().is_none());
}

#[test]
fn empty_string() {
    let tok = tiny();
    let ids = tok.encode("", false).unwrap();
    assert!(ids.is_empty());
    assert_eq!(tok.decode(&ids).unwrap(), "");
}

#[test]
fn bpe_applies_ranked_merges() {
    let tok = tiny();
    // "he" and "ll" are merged tokens (ids 4 and 5); "o" stays id 3.
    assert_eq!(tok.encode("he", false).unwrap(), vec![4]);
    assert_eq!(tok.encode("ll", false).unwrap(), vec![5]);
    assert_eq!(tok.encode("hello", false).unwrap(), vec![4, 5, 3]);
}

#[test]
fn from_header_missing_tokens_errors() {
    let bytes = write_gguf(&BTreeMap::new(), &[]).unwrap();
    let h = ModelHeader::from_bytes(&bytes).unwrap();
    assert!(Tokenizer::from_header(&h).is_err());
}

#[test]
fn from_header_add_bos_eos_and_pad() {
    let mut meta = BTreeMap::new();
    meta.insert(
        "tokenizer.ggml.tokens".into(),
        MetaValue::Array(
            ["<bos>", "<eos>", "<pad>", "h", "e", "l", "o", "he", "ll"]
                .into_iter()
                .map(|s| MetaValue::String(s.into()))
                .collect(),
        ),
    );
    meta.insert(
        "tokenizer.ggml.merges".into(),
        MetaValue::Array(vec![
            MetaValue::String("h e".into()),
            MetaValue::String("l l".into()),
        ]),
    );
    meta.insert("tokenizer.ggml.bos_token_id".into(), MetaValue::U32(0));
    meta.insert("tokenizer.ggml.eos_token_id".into(), MetaValue::U32(1));
    meta.insert("tokenizer.ggml.padding_token_id".into(), MetaValue::U32(2));
    meta.insert("tokenizer.ggml.add_bos_token".into(), MetaValue::Bool(true));
    meta.insert("tokenizer.ggml.add_eos_token".into(), MetaValue::Bool(true));
    let bytes = write_gguf(&meta, &[]).unwrap();
    let h = ModelHeader::from_bytes(&bytes).unwrap();
    let tok = Tokenizer::from_header(&h).unwrap();
    assert_eq!(tok.bos(), Some(0));
    assert_eq!(tok.eos(), Some(1));
    assert_eq!(tok.pad(), Some(2));
    // add_special=false: BPE only (he=7, ll=8, o=6).
    assert_eq!(tok.encode("hello", false).unwrap(), vec![7, 8, 6]);
    // add_special=true: bos + BPE + eos.
    assert_eq!(tok.encode("hello", true).unwrap(), vec![0, 7, 8, 6, 1]);
}

#[test]
fn from_header_roundtrip() {
    let mut meta = BTreeMap::new();
    meta.insert(
        "tokenizer.ggml.tokens".into(),
        MetaValue::Array(
            ["h", "e", "l", "o", "he", "ll"]
                .into_iter()
                .map(|s| MetaValue::String(s.into()))
                .collect(),
        ),
    );
    meta.insert(
        "tokenizer.ggml.merges".into(),
        MetaValue::Array(vec![
            MetaValue::String("h e".into()),
            MetaValue::String("l l".into()),
        ]),
    );
    meta.insert(
        "tokenizer.ggml.pre".into(),
        MetaValue::String("qwen2".into()),
    );
    meta.insert("tokenizer.ggml.bos_token_id".into(), MetaValue::U32(0));
    meta.insert("tokenizer.ggml.eos_token_id".into(), MetaValue::U32(1));
    meta.insert("tokenizer.ggml.add_bos_token".into(), MetaValue::Bool(false));
    meta.insert(
        "tokenizer.chat_template".into(),
        MetaValue::String("{{ messages }}".into()),
    );
    let bytes = write_gguf(&meta, &[]).unwrap();
    let h = ModelHeader::from_bytes(&bytes).unwrap();
    let tok = Tokenizer::from_header(&h).unwrap();
    assert_eq!(tok.decode(&tok.encode("hello", false).unwrap()).unwrap(), "hello");
    assert_eq!(tok.bos(), Some(0));
    assert_eq!(tok.eos(), Some(1));
    assert_eq!(tok.chat_template(), Some("{{ messages }}"));
}

/// Real Qwen GGUF shape: `gpt2` model, byte-level alphabet, control tokens.
fn byte_level_tok() -> Tokenizer {
    let tokens = ["<|im_start|>", "<|im_end|>", "h", "i", "Ġ", "Ċ", "hi", "Ġhi", "user"];
    let types = [3i32, 3, 1, 1, 1, 1, 1, 1, 1];
    let mut meta = BTreeMap::new();
    meta.insert(
        "tokenizer.ggml.tokens".into(),
        MetaValue::Array(tokens.iter().map(|s| MetaValue::String((*s).into())).collect()),
    );
    meta.insert(
        "tokenizer.ggml.token_type".into(),
        MetaValue::Array(types.iter().map(|t| MetaValue::I32(*t)).collect()),
    );
    meta.insert(
        "tokenizer.ggml.merges".into(),
        MetaValue::Array(
            ["h i", "Ġ hi"].iter().map(|s| MetaValue::String((*s).into())).collect(),
        ),
    );
    meta.insert("tokenizer.ggml.model".into(), MetaValue::String("gpt2".into()));
    meta.insert("tokenizer.ggml.pre".into(), MetaValue::String("qwen2".into()));
    let bytes = write_gguf(&meta, &[]).unwrap();
    Tokenizer::from_header(&ModelHeader::from_bytes(&bytes).unwrap()).unwrap()
}

#[test]
fn byte_level_space_and_newline() {
    let tok = byte_level_tok();
    assert_eq!(tok.encode("hi hi\n", false).unwrap(), vec![6, 7, 5]);
    assert_eq!(tok.decode(&[6, 7, 5]).unwrap(), "hi hi\n");
}

#[test]
fn control_tokens_match_verbatim() {
    let tok = byte_level_tok();
    let ids = tok.encode("<|im_start|>hi\nhi<|im_end|>", false).unwrap();
    assert_eq!(ids, vec![0, 6, 5, 6, 1]);
    assert_eq!(tok.decode(&ids).unwrap(), "<|im_start|>hi\nhi<|im_end|>");
}
