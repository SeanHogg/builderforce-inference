//! bf-compress — LLMLingua-2 token-level compression of large context (PRD C8).
//!
//! LLMLingua-2 is a small encoder (multilingual BERT-base) fine-tuned as a token
//! classifier: per token, the probability that it is needed to preserve the text's
//! meaning. Compression keeps the highest-probability words until the requested share of
//! tokens remains, in their original order. Three things never reach the classifier's
//! judgement: fenced code blocks (copied verbatim), protected words (paths, URLs, inline
//! code, numbers and their units — see `protect`) and the caller's `force_keep` words.
//!
//! The selection core is pure and always built; the ONNX model behind it (`Compressor`)
//! is behind the `compress` feature so a default build pulls no runtime and no model.

pub mod model;
pub mod protect;
pub mod segment;
pub mod select;

#[cfg(feature = "compress")]
pub mod onnx;

use anyhow::Result;
use segment::Segment;
pub use select::WordScore;

/// The compressed text and what it cost before and after, in model tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct Compressed {
    pub text: String,
    pub original_tokens: usize,
    pub kept_tokens: usize,
}

/// The token classifier, as compression sees it. The ONNX model implements it; tests use
/// a deterministic stub.
pub trait Scorer {
    /// One score per word, in order. Words are scored together so context informs each.
    fn score(&self, words: &[&str]) -> Result<Vec<WordScore>>;
    /// Model tokens in a span that is carried through unscored (a code fence).
    fn count_tokens(&self, text: &str) -> Result<usize>;
}

/// Compress `text` to about `rate` of its prose tokens (0..=1; 0.5 keeps half).
pub fn compress_with(scorer: &dyn Scorer, text: &str, rate: f32, force_keep: &[&str]) -> Result<Compressed> {
    if !(rate.is_finite() && rate < 1.0) {
        let n = scorer.count_tokens(text)?;
        return Ok(Compressed { text: text.to_string(), original_tokens: n, kept_tokens: n });
    }
    let segs = segment::segments(text);
    let mut fence_tokens = 0usize;
    let mut words: Vec<segment::Word> = Vec::new();
    // Per segment: the range of its words in `words` (prose) or None (fence).
    let mut layout: Vec<(Segment, Option<std::ops::Range<usize>>)> = Vec::with_capacity(segs.len());
    for seg in segs {
        match seg {
            Segment::Prose(p) => {
                let start = words.len();
                words.extend(segment::words(p));
                layout.push((seg, Some(start..words.len())));
            }
            Segment::Fence(f) => {
                fence_tokens += scorer.count_tokens(f)?;
                layout.push((seg, None));
            }
        }
    }
    let texts: Vec<&str> = words.iter().map(|w| w.text).collect();
    let scores = if texts.is_empty() { Vec::new() } else { scorer.score(&texts)? };
    anyhow::ensure!(scores.len() == texts.len(), "scorer returned {} scores for {} words", scores.len(), texts.len());
    let protected: Vec<bool> = texts
        .iter()
        .enumerate()
        .map(|(i, w)| protect::is_protected(w, i.checked_sub(1).map(|j| texts[j]), force_keep))
        .collect();
    let keep = select::keep_mask(&scores, &protected, rate);

    let mut out = String::with_capacity(text.len());
    for (seg, range) in &layout {
        match (seg, range) {
            (Segment::Fence(f), _) => out.push_str(f),
            (Segment::Prose(p), Some(r)) => {
                assemble(&mut out, &words[r.clone()], &keep[r.clone()]);
                // The prose's trailing whitespace (the newline before a fence) carries over.
                out.push_str(&p[p.trim_end().len()..]);
            }
            (Segment::Prose(_), None) => unreachable!("prose always has a word range"),
        }
    }
    let prose_tokens: usize = scores.iter().map(|s| s.tokens).sum();
    let kept_prose: usize = scores.iter().zip(&keep).filter(|(_, k)| **k).map(|(s, _)| s.tokens).sum();
    Ok(Compressed { text: out, original_tokens: prose_tokens + fence_tokens, kept_tokens: kept_prose + fence_tokens })
}

/// Word-level reassembly. A kept word keeps its own whitespace when nothing before it was
/// dropped; across a dropped run, the strongest break in that run survives (paragraph,
/// line, or a single space), so structure outlives the words that carried it.
fn assemble(out: &mut String, words: &[segment::Word], keep: &[bool]) {
    let mut dropped_gap = String::new();
    let mut dropped_any = false;
    for (w, k) in words.iter().zip(keep) {
        if !*k {
            dropped_gap.push_str(w.gap);
            dropped_any = true;
            continue;
        }
        if !dropped_any {
            out.push_str(w.gap);
        } else {
            dropped_gap.push_str(w.gap);
            let sep = if dropped_gap.contains("\n\n") {
                "\n\n"
            } else if dropped_gap.contains('\n') {
                "\n"
            } else if out.is_empty() || out.ends_with(char::is_whitespace) {
                ""
            } else {
                " "
            };
            out.push_str(sep);
        }
        out.push_str(w.text);
        dropped_gap.clear();
        dropped_any = false;
    }
}

#[cfg(feature = "compress")]
pub use loaded::{compress, Compressor};

#[cfg(feature = "compress")]
mod loaded {
    use super::{compress_with, model, onnx::OnnxScorer, Compressed};
    use anyhow::{anyhow, Result};
    use std::path::Path;
    use std::sync::OnceLock;

    /// The LLMLingua-2 model, loaded (and on first use downloaded) once.
    pub struct Compressor {
        scorer: OnnxScorer,
    }

    impl Compressor {
        /// Ensure the model files are in `cache_dir` (downloading and verifying them on
        /// first use) and load them.
        pub fn load(cache_dir: &Path) -> Result<Self> {
            let files = model::ModelSpec::from_env().ensure(cache_dir)?;
            Ok(Self { scorer: OnnxScorer::load(&files)? })
        }

        pub fn compress(&self, text: &str, rate: f32, force_keep: &[&str]) -> Result<Compressed> {
            compress_with(&self.scorer, text, rate, force_keep)
        }
    }

    /// Compress with a process-wide model in the default cache dir (`model::default_cache_dir`).
    pub fn compress(text: &str, rate: f32, force_keep: &[&str]) -> Result<Compressed> {
        static SHARED: OnceLock<Result<Compressor, String>> = OnceLock::new();
        let c = SHARED
            .get_or_init(|| Compressor::load(&model::default_cache_dir()).map_err(|e| format!("{e:#}")))
            .as_ref()
            .map_err(|e| anyhow!("compression model unavailable: {e}"))?;
        c.compress(text, rate, force_keep)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Keeps words by length (longer = more meaningful); one token per 4 chars, min 1.
    struct ByLength;

    impl Scorer for ByLength {
        fn score(&self, words: &[&str]) -> Result<Vec<WordScore>> {
            Ok(words.iter().map(|w| WordScore { keep: (w.len() as f32 / 20.0).min(1.0), tokens: w.len().div_ceil(4).max(1) }).collect())
        }
        fn count_tokens(&self, text: &str) -> Result<usize> {
            Ok(text.split_whitespace().map(|w| w.len().div_ceil(4).max(1)).sum())
        }
    }

    #[test]
    fn drops_low_value_words_and_keeps_order() {
        let c = compress_with(&ByLength, "a an the comprehensive architecture of it", 0.5, &[]).unwrap();
        assert_eq!(c.text, "comprehensive architecture");
        assert!(c.kept_tokens <= c.original_tokens);
        assert_eq!(c.original_tokens, 1 + 1 + 1 + 4 + 3 + 1 + 1);
    }

    #[test]
    fn fences_paths_numbers_and_force_keep_survive() {
        let text = "So we see the build in src/main.rs takes 512 MB and it is slow\n```rust\nfn  keep_me()  {}\n```\nok then a b c Stripe\n";
        let c = compress_with(&ByLength, text, 0.1, &["Stripe"]).unwrap();
        assert!(c.text.contains("src/main.rs"));
        assert!(c.text.contains("512 MB"));
        assert!(c.text.contains("```rust\nfn  keep_me()  {}\n```\n"), "{}", c.text);
        assert!(c.text.contains("Stripe"));
        assert!(!c.text.contains(" the "));
    }

    #[test]
    fn line_and_paragraph_breaks_outlive_dropped_words() {
        let c = compress_with(&ByLength, "headline\nx\n\ny paragraph", 0.6, &[]).unwrap();
        assert_eq!(c.text, "headline\n\nparagraph");
    }

    #[test]
    fn rate_one_and_empty_input_are_identity() {
        let t = "keep  every\n word";
        let c = compress_with(&ByLength, t, 1.0, &[]).unwrap();
        assert_eq!(c.text, t);
        assert_eq!(c.original_tokens, c.kept_tokens);
        assert_eq!(compress_with(&ByLength, "", 0.5, &[]).unwrap().text, "");
        let only_fence = "```\ncode\n```";
        assert_eq!(compress_with(&ByLength, only_fence, 0.2, &[]).unwrap().text, only_fence);
    }

    #[test]
    fn own_whitespace_kept_when_nothing_dropped_before() {
        // Rate 0 keeps only protected words; the dropped bullets collapse to the line break.
        let c = compress_with(&ByLength, "  - important_item\n  - other_thing", 0.0, &[]).unwrap();
        assert_eq!(c.text, "important_item\nother_thing");
        // Nothing dropped: each word keeps exactly the whitespace it had.
        let c = compress_with(&ByLength, "  first_item\n    second_item", 0.0, &[]).unwrap();
        assert_eq!(c.text, "  first_item\n    second_item");
    }
}
