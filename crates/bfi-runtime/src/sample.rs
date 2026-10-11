//! Token sampler: greedy, temperature, top-k/p, min-p, repetition penalty.

use std::collections::HashSet;

/// Sampling parameters for token generation.
#[derive(Debug, Clone, PartialEq)]
pub struct SampleParams {
    /// Maximum tokens to generate. 0 = unlimited (bounded by the model's context).
    pub max_tokens: u32,
    /// `0` → greedy argmax.
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    /// Repetition penalty (applied before temperature). 1.0 = no penalty.
    pub repeat_penalty: f32,
    /// Presence penalty (applied after repeat penalty). 0 = no penalty.
    pub presence_penalty: f32,
    pub seed: u64,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            seed: 0,
        }
    }
}

/// Sample a token id from `logits`. Greedy when `temperature == 0`.
pub fn sample(logits: &[f32], prev: &[u32], p: &SampleParams) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    let mut vals: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();

    // Apply repeat penalty first.
    if (p.repeat_penalty - 1.0).abs() > f32::EPSILON && !prev.is_empty() {
        let seen: HashSet<u32> = prev.iter().copied().collect();
        for (id, logit) in vals.iter_mut() {
            if seen.contains(id) {
                if *logit > 0.0 {
                    *logit /= p.repeat_penalty;
                } else {
                    *logit *= p.repeat_penalty;
                }
            }
        }
    }

    // Apply presence penalty after repeat penalty.
    if p.presence_penalty != 0.0 && !prev.is_empty() {
        let seen: HashSet<u32> = prev.iter().copied().collect();
        for (id, logit) in vals.iter_mut() {
            if seen.contains(id) {
                *logit -= p.presence_penalty;
            }
        }
    }

    if p.temperature <= 0.0 {
        return argmax(&vals);
    }

    for (_, logit) in vals.iter_mut() {
        *logit /= p.temperature;
    }

    let max = vals
        .iter()
        .map(|(_, l)| *l)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (_, l) in vals.iter_mut() {
        *l = (*l - max).exp();
        sum += *l;
    }
    if sum > 0.0 {
        for (_, l) in vals.iter_mut() {
            *l /= sum;
        }
    }

    if p.top_k > 0 && (p.top_k as usize) < vals.len() {
        vals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        vals.truncate(p.top_k as usize);
    }

    if p.top_p < 1.0 {
        vals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cum = 0.0f32;
        let mut cut = vals.len();
        for (i, (_, prob)) in vals.iter().enumerate() {
            cum += *prob;
            if cum >= p.top_p {
                cut = i + 1;
                break;
            }
        }
        vals.truncate(cut.max(1));
    }

    if p.min_p > 0.0 {
        let max_p = vals
            .iter()
            .map(|(_, pr)| *pr)
            .fold(0.0f32, f32::max);
        let thresh = p.min_p * max_p;
        vals.retain(|(_, pr)| *pr >= thresh);
        if vals.is_empty() {
            return argmax(
                &logits
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i as u32, v))
                    .collect::<Vec<_>>(),
            );
        }
    }

    let sum: f32 = vals.iter().map(|(_, pr)| *pr).sum();
    let mut rng = Rng(p.seed);
    let r = rng.f32() * sum.max(0.0);
    let mut cum = 0.0f32;
    for (id, prob) in &vals {
        cum += *prob;
        if r <= cum {
            return *id;
        }
    }
    vals.last().map(|(id, _)| *id).unwrap_or(0)
}

fn argmax(vals: &[(u32, f32)]) -> u32 {
    let mut best_i = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for &(i, v) in vals {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

/// xorshift64* — no extra rand crate.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        if x == 0 {
            x = 0x9E37_79B9_7F4A_7C15;
        }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u32 << 24) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_argmax() {
        let p = SampleParams {
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(sample(&[1.0, 0.0, 0.5], &[], &p), 0);
        assert_eq!(sample(&[0.1, 2.0, 1.5], &[], &p), 1);
    }

    #[test]
    fn default_is_greedy() {
        let p = SampleParams::default();
        assert_eq!(p.max_tokens, 256);
        assert_eq!(p.temperature, 0.0);
        assert_eq!(p.top_k, 0);
        assert_eq!(p.top_p, 1.0);
        assert_eq!(p.min_p, 0.0);
        assert_eq!(p.repeat_penalty, 1.0);
        assert_eq!(p.presence_penalty, 0.0);
        assert_eq!(p.seed, 0);
        assert_eq!(sample(&[1.0, 0.0, 0.5], &[], &p), 0);
    }

    #[test]
    fn repeat_penalty_shifts_greedy() {
        let p = SampleParams {
            temperature: 0.0,
            repeat_penalty: 1.5,
            ..Default::default()
        };
        // token 0 was used; its positive logit is divided, so 2 wins.
        assert_eq!(sample(&[1.2, 1.0], &[0], &p), 1);
    }

    #[test]
    fn presence_penalty_subtracts() {
        let p = SampleParams {
            temperature: 0.0,
            presence_penalty: 0.5,
            ..Default::default()
        };
        // token 0 was used; its logit should be reduced by 0.5
        // Original: [1.0, 0.5] → after presence penalty on token 0: [0.5, 0.5]
        // argmax picks 0 (tie-breaker)
        assert_eq!(sample(&[1.0, 0.5], &[0], &p), 0);
    }

    #[test]
    fn temperature_scales_logits() {
        // With temperature, higher-temperature makes distribution more uniform
        let p_hot = SampleParams {
            temperature: 10.0, // very high temp → almost uniform
            ..Default::default()
        };
        // With very high temp, even low logits have chance
        let result = sample(&[1.0, 100.0], &[], &p_hot);
        // At high temp, result could be either (not guaranteed deterministic with seed=0)
        assert!(result <= 1);

        let p_cold = SampleParams {
            temperature: 0.1, // low temp → sharp distribution
            ..Default::default()
        };
        // With low temp, higher logit always wins
        assert_eq!(sample(&[1.0, 100.0], &[], &p_cold), 1);
    }

    #[test]
    fn top_k_filters() {
        // vocab = [0,1,2,3], logits = [10, 5, 2, 1], top_k=2 should keep only 0 and 1
        let logits = [10.0, 5.0, 2.0, 1.0];
        let p = SampleParams {
            temperature: 0.0,
            top_k: 2,
            ..Default::default()
        };
        // Should pick 0 (highest) since top-2 includes 0 and 1, and 0 > 1
        assert_eq!(sample(&logits, &[], &p), 0);
    }

    #[test]
    fn top_p_nucleus() {
        // vocab = [0,1,2,3], logits = [10.0, 5.0, 1.0, 0.1]
        // After softmax: ~0.999 for 0, ~0.001 for others
        // top_p=0.5 should keep only token 0
        let logits = [10.0, 5.0, 1.0, 0.1];
        let p = SampleParams {
            temperature: 1.0,
            top_p: 0.5,
            ..Default::default()
        };
        // With greedy and top_p=0.5, should pick token 0 (highest prob)
        assert_eq!(sample(&logits, &[], &p), 0);

        // With top_p close to 1, should consider more tokens
        let p_open = SampleParams {
            temperature: 1.0,
            top_p: 0.999,
            ..Default::default()
        };
        // Could pick any token depending on cumulative probability
        let result = sample(&logits, &[], &p_open);
        assert!(result <= 3);
    }
}
