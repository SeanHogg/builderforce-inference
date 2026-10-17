//! GPT-2 byte-level alphabet (`bytes_to_unicode` in OpenAI's `encoder.py`,
//! llama.cpp `unicode_byte_to_utf8`).
//!
//! GGUF `tokenizer.ggml.model = "gpt2"` vocabularies (Qwen2/Qwen3, Bonsai) store
//! tokens in this alphabet: printable bytes map to themselves, every other byte
//! to a code point at 256+. So `" hello"` is the token `"Ġhello"` and `"\n"` is
//! `"Ċ"`. Encoding maps text bytes into the alphabet before BPE; decoding maps
//! token characters back to bytes.

use std::sync::OnceLock;

struct Tables {
    to_char: [char; 256],
    to_byte: std::collections::HashMap<char, u8>,
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let printable = |b: u32| {
            (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b)
        };
        let mut to_char = ['\0'; 256];
        let mut to_byte = std::collections::HashMap::with_capacity(256);
        let mut n = 0u32;
        for b in 0u32..256 {
            let cp = if printable(b) {
                b
            } else {
                n += 1;
                255 + n
            };
            let c = char::from_u32(cp).expect("byte-level code points are valid");
            to_char[b as usize] = c;
            to_byte.insert(c, b as u8);
        }
        Tables { to_char, to_byte }
    })
}

/// Text bytes → byte-level alphabet characters, one char per byte.
pub fn encode_bytes(text: &str) -> impl Iterator<Item = char> + '_ {
    let t = tables();
    text.bytes().map(move |b| t.to_char[b as usize])
}

/// A byte-level token back to raw bytes. Characters outside the alphabet
/// (none in a well-formed vocab) pass through as UTF-8.
pub fn decode_into(token: &str, out: &mut Vec<u8>) {
    let t = tables();
    for c in token.chars() {
        match t.to_byte.get(&c) {
            Some(&b) => out.push(b),
            None => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_newline_and_printables() {
        assert_eq!(encode_bytes(" ").collect::<String>(), "Ġ");
        assert_eq!(encode_bytes("\n").collect::<String>(), "Ċ");
        assert_eq!(encode_bytes("A~!").collect::<String>(), "A~!");
        let mut out = Vec::new();
        decode_into("ĠhelloĊ", &mut out);
        assert_eq!(out, b" hello\n");
    }

    #[test]
    fn every_byte_roundtrips() {
        let all: Vec<u8> = (0u8..=255).collect();
        let t = tables();
        let s: String = all.iter().map(|&b| t.to_char[b as usize]).collect();
        let mut out = Vec::new();
        decode_into(&s, &mut out);
        assert_eq!(out, all);
    }
}
