//! The LLMLingua-2 classifier on ONNX Runtime: words → WordPiece tokens → keep
//! probability per token (softmax over the two labels, label 1 = keep) → mean per word,
//! as LLMLingua-2 itself aggregates.

use crate::model::ModelFiles;
use crate::select::WordScore;
use crate::Scorer;
use anyhow::{anyhow, Result};
use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use tokenizers::Tokenizer;

/// BERT's position limit; two slots go to [CLS] and [SEP].
const WINDOW: usize = 512;
const BODY: usize = WINDOW - 2;
/// The label whose probability means "keep this token".
const KEEP_LABEL: usize = 1;

pub struct OnnxScorer {
    session: Session,
    tokenizer: Tokenizer,
    cls: u32,
    sep: u32,
    needs_type_ids: bool,
    output: String,
}

impl OnnxScorer {
    pub fn load(files: &ModelFiles) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(&files.tokenizer).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(&files.onnx)?;
        let needs_type_ids = session.inputs.iter().any(|i| i.name == "token_type_ids");
        let output = session
            .outputs
            .iter()
            .find(|o| o.name == "logits")
            .or_else(|| session.outputs.first())
            .map(|o| o.name.clone())
            .ok_or_else(|| anyhow!("model has no outputs"))?;
        let cls = tokenizer.token_to_id("[CLS]").unwrap_or(101);
        let sep = tokenizer.token_to_id("[SEP]").unwrap_or(102);
        Ok(Self { session, tokenizer, cls, sep, needs_type_ids, output })
    }

    /// Keep probability for each token of one window (without the special tokens).
    fn window_probs(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let n = ids.len() + 2;
        let mut input: Vec<i64> = Vec::with_capacity(n);
        input.push(self.cls as i64);
        input.extend(ids.iter().map(|&i| i as i64));
        input.push(self.sep as i64);
        let input_ids = Array2::from_shape_vec((1, n), input)?;
        let mask = Array2::from_elem((1, n), 1i64);
        let mut inputs = ort::inputs![
            "input_ids" => Value::from_array(input_ids)?,
            "attention_mask" => Value::from_array(mask)?,
        ]?;
        if self.needs_type_ids {
            inputs.push(("token_type_ids".into(), Value::from_array(Array2::<i64>::zeros((1, n)))?.into()));
        }
        let outputs = self.session.run(inputs)?;
        let logits = outputs[self.output.as_str()].try_extract_tensor::<f32>()?;
        let shape = logits.shape().to_vec();
        if shape.len() != 3 || shape[1] != n || shape[2] <= KEEP_LABEL {
            return Err(anyhow!("unexpected logits shape {shape:?} for {n} tokens"));
        }
        // Skip [CLS] (position 0) and [SEP] (last).
        Ok((1..n - 1)
            .map(|t| {
                let row: Vec<f32> = (0..shape[2]).map(|l| logits[[0, t, l]]).collect();
                keep_probability(&row)
            })
            .collect())
    }
}

/// Softmax probability of the keep label, computed stably.
fn keep_probability(logits: &[f32]) -> f32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|l| (l - max).exp()).sum();
    (logits[KEEP_LABEL] - max).exp() / sum
}

impl Scorer for OnnxScorer {
    fn score(&self, words: &[&str]) -> Result<Vec<WordScore>> {
        // Pre-tokenized input: `word_ids` maps every token back to the word it came from.
        let enc = self.tokenizer.encode(words.to_vec(), false).map_err(|e| anyhow!("tokenize: {e}"))?;
        let ids = enc.get_ids();
        let word_ids = enc.get_word_ids();
        let mut sum = vec![0f32; words.len()];
        let mut count = vec![0usize; words.len()];
        for (w, chunk) in ids.chunks(BODY).enumerate() {
            let probs = self.window_probs(chunk)?;
            for (j, p) in probs.into_iter().enumerate() {
                if let Some(Some(word)) = word_ids.get(w * BODY + j) {
                    let word = *word as usize;
                    if word < words.len() {
                        sum[word] += p;
                        count[word] += 1;
                    }
                }
            }
        }
        Ok(sum
            .into_iter()
            .zip(count)
            .map(|(s, c)| WordScore { keep: if c == 0 { 1.0 } else { s / c as f32 }, tokens: c })
            .collect())
    }

    fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(self.tokenizer.encode(text, false).map_err(|e| anyhow!("tokenize: {e}"))?.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keep_probability_is_softmax_of_label_one() {
        assert!((keep_probability(&[0.0, 0.0]) - 0.5).abs() < 1e-6);
        assert!(keep_probability(&[-5.0, 5.0]) > 0.99);
        assert!(keep_probability(&[1000.0, 0.0]) < 1e-6);
    }

    /// Needs the model (≈179 MB download): `BF_COMPRESS_LIVE=1 cargo test -p bf-compress
    /// --features compress -- --ignored`.
    #[test]
    #[ignore]
    fn live_model_compresses() {
        if std::env::var("BF_COMPRESS_LIVE").is_err() {
            return;
        }
        let dir = crate::model::default_cache_dir();
        let c = crate::Compressor::load(&dir).unwrap();
        let text = "So, um, basically what we decided in the meeting was that the billing service in api/src/billing.ts should retry webhooks three times with a 30 s backoff.";
        let out = c.compress(text, 0.5, &[]).unwrap();
        assert!(out.kept_tokens < out.original_tokens);
        assert!(out.text.contains("api/src/billing.ts"));
        assert!(out.text.contains("30 s"));
    }
}
