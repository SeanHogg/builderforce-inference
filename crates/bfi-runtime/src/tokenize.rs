//! GPT-2 byte-level BPE tokenizer loaded from GGUF metadata.

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::{anyhow, bail, Result};
use bfi_gguf::ModelHeader;
use fancy_regex::Regex;

/// Qwen2/Qwen3 pre-tokenizer (llama.cpp `LLAMA_VOCAB_PRE_TYPE_QWEN2`).
const QWEN_PRE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

fn qwen_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(QWEN_PRE).expect("qwen pretok regex"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreTok {
    None,
    Qwen,
}

/// Byte-level BPE tokenizer (GPT-2 / Qwen).
#[derive(Debug, Clone)]
pub struct Tokenizer {
    tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
    /// Merge pair → rank (lower merges first).
    merges: HashMap<(String, String), u32>,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    pad_id: Option<u32>,
    unk_id: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    pre: PreTok,
    chat_template: Option<String>,
    /// Token id for each of the 256 bytes, if the vocab has a fallback.
    byte_id: [Option<u32>; 256],
}

impl Tokenizer {
    pub fn from_header(h: &ModelHeader) -> Result<Self> {
        let raw_tokens = h
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_string_array())
            .ok_or_else(|| anyhow!("missing tokenizer.ggml.tokens"))?;
        let tokens: Vec<String> = raw_tokens.into_iter().map(|s| s.to_string()).collect();

        let raw_merges = h
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_string_array())
            .unwrap_or_default();
        let merges: Vec<String> = raw_merges.into_iter().map(|s| s.to_string()).collect();

        let bos_id = meta_u32(h, "tokenizer.ggml.bos_token_id");
        let eos_id = meta_u32(h, "tokenizer.ggml.eos_token_id");
        let pad_id = meta_u32(h, "tokenizer.ggml.padding_token_id");
        let unk_id = meta_u32(h, "tokenizer.ggml.unknown_token_id");
        let add_bos = meta_bool(h, "tokenizer.ggml.add_bos_token").unwrap_or(false);
        let add_eos = meta_bool(h, "tokenizer.ggml.add_eos_token").unwrap_or(false);
        let pre_s = h.get_str("tokenizer.ggml.pre").unwrap_or("");
        let pre = if pre_s.to_ascii_lowercase().contains("qwen") {
            PreTok::Qwen
        } else {
            PreTok::None
        };
        let chat_template = h.get_str("tokenizer.chat_template").map(|s| s.to_string());

        Ok(Self::build(
            tokens,
            &merges,
            bos_id,
            eos_id,
            pad_id,
            unk_id,
            add_bos,
            add_eos,
            pre,
            chat_template,
        ))
    }

    /// Test / synthetic constructor. `merges` are `"left right"` pairs in rank order.
    pub fn from_parts(tokens: Vec<String>, merges: Vec<String>) -> Self {
        Self::build(
            tokens,
            &merges,
            None,
            None,
            None,
            None,
            false,
            false,
            PreTok::Qwen,
            None,
        )
    }

    fn build(
        tokens: Vec<String>,
        merges: &[String],
        bos_id: Option<u32>,
        eos_id: Option<u32>,
        pad_id: Option<u32>,
        unk_id: Option<u32>,
        add_bos: bool,
        add_eos: bool,
        pre: PreTok,
        chat_template: Option<String>,
    ) -> Self {
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        let mut byte_id = [None; 256];
        for (i, t) in tokens.iter().enumerate() {
            let id = i as u32;
            token_to_id.entry(t.clone()).or_insert(id);
            if let Some(b) = parse_byte_token(t) {
                byte_id[b as usize] = Some(id);
            } else if t.len() == 1 {
                let b = t.as_bytes()[0];
                if byte_id[b as usize].is_none() {
                    byte_id[b as usize] = Some(id);
                }
            }
        }

        let mut merge_map = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            if let Some((a, b)) = m.split_once(' ') {
                merge_map
                    .entry((a.to_string(), b.to_string()))
                    .or_insert(rank as u32);
            }
        }

        Self {
            tokens,
            token_to_id,
            merges: merge_map,
            bos_id,
            eos_id,
            pad_id,
            unk_id,
            add_bos,
            add_eos,
            pre,
            chat_template,
            byte_id,
        }
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        if add_special && self.add_bos {
            if let Some(bos) = self.bos_id {
                ids.push(bos);
            }
        }

        let pieces = self.pre_tokenize(text)?;
        for piece in pieces {
            ids.extend(self.bpe(&piece)?);
        }

        if add_special && self.add_eos {
            if let Some(eos) = self.eos_id {
                ids.push(eos);
            }
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(tok) = self.tokens.get(id as usize) else {
                bail!("token id {id} out of range (n_vocab={})", self.tokens.len());
            };
            if let Some(b) = parse_byte_token(tok) {
                bytes.push(b);
            } else {
                bytes.extend_from_slice(tok.as_bytes());
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn bos(&self) -> Option<u32> {
        self.bos_id
    }

    pub fn eos(&self) -> Option<u32> {
        self.eos_id
    }

    pub fn pad(&self) -> Option<u32> {
        self.pad_id
    }

    pub fn n_vocab(&self) -> usize {
        self.tokens.len()
    }

    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    fn pre_tokenize(&self, text: &str) -> Result<Vec<String>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        match self.pre {
            PreTok::Qwen => {
                let mut out = Vec::new();
                for m in qwen_regex().find_iter(text) {
                    let m = m.map_err(|e| anyhow!("pretok: {e}"))?;
                    out.push(m.as_str().to_string());
                }
                Ok(out)
            }
            PreTok::None => Ok(vec![text.to_string()]),
        }
    }

    fn bpe(&self, piece: &str) -> Result<Vec<u32>> {
        if piece.is_empty() {
            return Ok(Vec::new());
        }

        let mut word: Vec<String> = Vec::new();
        for ch in piece.chars() {
            let s = ch.to_string();
            if self.token_to_id.contains_key(&s) {
                word.push(s);
            } else {
                for b in s.into_bytes() {
                    word.push(self.byte_symbol(b)?);
                }
            }
        }

        loop {
            if word.len() < 2 {
                break;
            }
            let mut best_i = None;
            let mut best_rank = u32::MAX;
            for i in 0..word.len() - 1 {
                if let Some(&rank) = self.merges.get(&(word[i].clone(), word[i + 1].clone())) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_i = Some(i);
                    }
                }
            }
            let Some(i) = best_i else { break };
            let merged = format!("{}{}", word[i], word[i + 1]);
            word[i] = merged;
            word.remove(i + 1);
        }

        word.into_iter()
            .map(|s| self.id_of(&s))
            .collect()
    }

    fn byte_symbol(&self, b: u8) -> Result<String> {
        if let Some(id) = self.byte_id[b as usize] {
            return Ok(self.tokens[id as usize].clone());
        }
        let hex = format!("<0x{b:02X}>");
        if self.token_to_id.contains_key(&hex) {
            return Ok(hex);
        }
        if let Some(unk) = self.unk_id {
            return Ok(self.tokens[unk as usize].clone());
        }
        bail!("no vocab entry for byte 0x{b:02X}");
    }

    fn id_of(&self, s: &str) -> Result<u32> {
        if let Some(&id) = self.token_to_id.get(s) {
            return Ok(id);
        }
        if let Some(unk) = self.unk_id {
            return Ok(unk);
        }
        bail!("unknown token {s:?}");
    }
}

fn meta_u32(h: &ModelHeader, key: &str) -> Option<u32> {
    h.get(key).and_then(|v| v.as_u64()).map(|x| x as u32)
}

fn meta_bool(h: &ModelHeader, key: &str) -> Option<bool> {
    h.get(key).and_then(|v| v.as_u64()).map(|x| x != 0)
}

fn parse_byte_token(s: &str) -> Option<u8> {
    let rest = s.strip_prefix("<0x")?.strip_suffix('>')?;
    if rest.len() != 2 {
        return None;
    }
    u8::from_str_radix(rest, 16).ok()
}

#[cfg(test)]
mod tests {
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
}
