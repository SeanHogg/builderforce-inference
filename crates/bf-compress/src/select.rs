//! Which words survive: every protected word, then the highest keep-probability words
//! until the kept share of model tokens reaches the requested rate. Order is never
//! changed — selection only drops.

/// What the classifier said about one word.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WordScore {
    /// Probability (0..1) that the word carries meaning and should be kept.
    pub keep: f32,
    /// Model tokens the word costs.
    pub tokens: usize,
}

/// A keep mask over `scores`. `rate` is the target share of tokens to keep (clamped to
/// 0..=1); protected words are always kept even if they alone exceed it.
pub fn keep_mask(scores: &[WordScore], protected: &[bool], rate: f32) -> Vec<bool> {
    debug_assert_eq!(scores.len(), protected.len());
    let rate = if rate.is_finite() { rate.clamp(0.0, 1.0) } else { 1.0 };
    // Words the tokenizer normalized away cost nothing; dropping them saves nothing.
    let mut keep: Vec<bool> = scores.iter().zip(protected).map(|(s, p)| *p || s.tokens == 0).collect();
    if rate >= 1.0 {
        return vec![true; scores.len()];
    }
    let total: usize = scores.iter().map(|s| s.tokens).sum();
    let target = (rate as f64 * total as f64).ceil() as usize;
    let mut kept: usize = scores.iter().zip(&keep).filter(|(_, k)| **k).map(|(s, _)| s.tokens).sum();
    let mut candidates: Vec<usize> = (0..scores.len()).filter(|&i| !keep[i]).collect();
    // Highest keep-probability first; ties go to the earlier word so the result is stable.
    candidates.sort_by(|&a, &b| scores[b].keep.partial_cmp(&scores[a].keep).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    for i in candidates {
        if kept >= target {
            break;
        }
        keep[i] = true;
        kept += scores[i].tokens;
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(keep: f32, tokens: usize) -> WordScore {
        WordScore { keep, tokens }
    }

    #[test]
    fn keeps_highest_probability_share_in_order() {
        let scores = [s(0.9, 1), s(0.1, 1), s(0.8, 1), s(0.2, 1)];
        assert_eq!(keep_mask(&scores, &[false; 4], 0.5), vec![true, false, true, false]);
        assert_eq!(keep_mask(&scores, &[false; 4], 1.0), vec![true; 4]);
        assert_eq!(keep_mask(&scores, &[false; 4], 0.0), vec![false; 4]);
    }

    #[test]
    fn protected_words_count_toward_the_budget_and_always_stay() {
        let scores = [s(0.9, 1), s(0.1, 1), s(0.8, 1), s(0.2, 1)];
        // Protected low-probability word fills the 50% budget with the top word.
        assert_eq!(keep_mask(&scores, &[false, true, false, false], 0.5), vec![true, true, false, false]);
        // Over budget: protected words still kept.
        assert_eq!(keep_mask(&scores, &[false, true, false, true], 0.25), vec![false, true, false, true]);
    }

    #[test]
    fn token_weighted_and_free_words_kept() {
        // A 3-token word with the top score satisfies a 50% target of 6 tokens on its own.
        let scores = [s(0.95, 3), s(0.5, 1), s(0.4, 2), s(0.0, 0)];
        assert_eq!(keep_mask(&scores, &[false; 4], 0.5), vec![true, false, false, true]);
        assert_eq!(keep_mask(&scores, &[false; 4], f32::NAN), vec![true; 4]);
    }
}
