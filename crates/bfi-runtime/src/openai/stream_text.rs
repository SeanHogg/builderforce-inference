//! [`StreamText`]: the decoded text of a growing token sequence, extended one
//! token at a time instead of re-decoding every generated token per step.
//!
//! Equal at every step to `String::from_utf8_lossy(bytes of all tokens)` —
//! what `Tokenizer::decode(&generated)` returned — including the single
//! U+FFFD that stands for an incomplete trailing UTF-8 sequence until the next
//! token completes it.

use crate::tokenize::Tokenizer;

#[derive(Debug, Default)]
pub(super) struct StreamText {
    text: String,
    /// Trailing bytes of a not-yet-complete UTF-8 character.
    tail: Vec<u8>,
    /// Bytes at the end of `text` that render `tail` (its U+FFFD placeholder).
    tail_text_len: usize,
}

impl StreamText {
    pub(super) fn as_str(&self) -> &str {
        &self.text
    }

    /// Append token `id`. Returns the byte offset in [`Self::as_str`] from
    /// which the text changed (earlier bytes are untouched). An id the
    /// tokenizer cannot decode renders as U+FFFD (it used to blank the text).
    pub(super) fn push(&mut self, tok: &Tokenizer, id: u32) -> usize {
        let start = self.text.len() - self.tail_text_len;
        self.text.truncate(start);
        let mut bytes = std::mem::take(&mut self.tail);
        if tok.decode_bytes_into(id, &mut bytes).is_err() {
            bytes.extend_from_slice("\u{FFFD}".as_bytes());
        }
        let mut rest: &[u8] = &bytes;
        loop {
            match std::str::from_utf8(rest) {
                Ok(s) => {
                    self.text.push_str(s);
                    rest = &[];
                    break;
                }
                Err(e) => {
                    let (valid, after) = rest.split_at(e.valid_up_to());
                    self.text.push_str(&String::from_utf8_lossy(valid));
                    match e.error_len() {
                        // Invalid bytes: one U+FFFD per maximal invalid run, as lossy decoding does.
                        Some(n) => {
                            self.text.push('\u{FFFD}');
                            rest = &after[n..];
                        }
                        // Incomplete character at the end: wait for the next token.
                        None => {
                            rest = after;
                            break;
                        }
                    }
                }
            }
        }
        self.tail = rest.to_vec();
        self.tail_text_len = 0;
        if !self.tail.is_empty() {
            self.text.push('\u{FFFD}');
            self.tail_text_len = '\u{FFFD}'.len_utf8();
        }
        start
    }

    /// Cut the text at byte `at` (a stop string); the incomplete tail goes too.
    pub(super) fn truncate(&mut self, at: usize) {
        self.text.truncate(at);
        self.tail.clear();
        self.tail_text_len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Byte tokens `<0x..>` split multi-byte characters across tokens.
    fn tok() -> Tokenizer {
        let mut v: Vec<String> = ["a", "b", "\u{e9}"].iter().map(|s| s.to_string()).collect();
        v.extend((0..=255u32).map(|b| format!("<0x{b:02X}>")));
        Tokenizer::from_parts(v, vec![])
    }

    fn byte(b: u8) -> u32 {
        3 + b as u32
    }

    #[test]
    fn matches_full_lossy_decode_at_every_step() {
        let t = tok();
        // "a", "é" split in two byte tokens, an invalid byte, "b", a dangling lead byte.
        let ids = [0, byte(0xC3), byte(0xA9), byte(0xFF), 1, 2, byte(0xE2), byte(0x82)];
        let mut s = StreamText::default();
        for i in 0..ids.len() {
            let from = s.push(&t, ids[i]);
            let want = t.decode(&ids[..=i]).unwrap();
            assert_eq!(s.as_str(), want, "after {} tokens", i + 1);
            assert!(from <= s.as_str().len());
        }
        assert_eq!(s.as_str(), "a\u{e9}\u{FFFD}b\u{e9}\u{FFFD}");
    }

    #[test]
    fn change_offset_covers_a_completed_character() {
        let t = tok();
        let mut s = StreamText::default();
        s.push(&t, 0);
        assert_eq!(s.push(&t, byte(0xC3)), 1, "placeholder starts at 1");
        assert_eq!(s.push(&t, byte(0xA9)), 1, "completion rewrites from the placeholder");
        assert_eq!(s.as_str(), "a\u{e9}");
        s.truncate(1);
        assert_eq!(s.as_str(), "a");
        assert_eq!(s.push(&t, 1), 1);
    }

    #[test]
    fn undecodable_id_is_a_replacement_char_not_a_blank() {
        let t = tok();
        let mut s = StreamText::default();
        s.push(&t, 0);
        s.push(&t, 10_000);
        assert_eq!(s.as_str(), "a\u{FFFD}");
    }
}
