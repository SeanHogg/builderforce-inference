//! Prompt-lookup speculative decoding (PRD E4).
//!
//! Each step samples the next token as usual, then DRAFTS up to k more by finding the
//! context's last n-gram (n = 3, falling back to 2) earlier in prompt + generated text and
//! proposing what followed it. One `Model::forward_each` pass scores the token and the
//! whole draft; the longest prefix of the draft the target agrees with is kept:
//! - greedy: a drafted token is accepted iff it IS the argmax — output is identical to
//!   plain decode;
//! - sampling: standard speculative acceptance for a deterministic draft (q = point mass):
//!   accept with probability p(d); on rejection the next token is drawn from p with d
//!   removed, renormalized, which leaves the output distribution exactly p.
//!
//! A rejected suffix is rolled back: KV-only state just rewinds `n`; recurrent state is
//! restored from a snapshot taken before the verify pass and the accepted tokens are
//! re-forwarded (the cost of not being able to rewind a recurrence).

use anyhow::Result;

use super::options::SpecLookup;
use super::{Engine, GenSlot, Pushed};
use crate::sample::{draw, target, Rng, SampleParams, Target};

/// Cumulative speculative-decode counters (for `bfi bench` and diagnostics).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpecStats {
    /// Multi-token verify passes run.
    pub verify_passes: u64,
    /// Tokens proposed by the n-gram lookup.
    pub drafted: u64,
    /// Drafted tokens the target accepted.
    pub accepted: u64,
}

impl SpecStats {
    pub fn acceptance(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        }
    }
}

/// Up to `k` tokens that followed the most recent earlier occurrence of `ctx`'s last
/// n-gram, trying `n = ngram_max` down to `ngram_min`. Empty when nothing matches.
pub fn draft(ctx: &[u32], k: usize, ngram_max: usize, ngram_min: usize) -> Vec<u32> {
    if k == 0 {
        return Vec::new();
    }
    let len = ctx.len();
    for n in (ngram_min.max(1)..=ngram_max).rev() {
        if len < n + 1 {
            continue;
        }
        let tail = &ctx[len - n..];
        // Most recent occurrence first; it must end before the tail itself starts.
        for start in (0..len - n).rev() {
            if &ctx[start..start + n] == tail {
                let from = start + n;
                let to = (from + k).min(len);
                if from < to {
                    return ctx[from..to].to_vec();
                }
            }
        }
    }
    Vec::new()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Verdict {
    Accept,
    /// Rejected; the token to emit instead (drawn from the residual distribution).
    Reject(u32),
}

/// Judge drafted token `d` against the target distribution. `u` decides acceptance,
/// `u2` draws the replacement on rejection (both uniform in `[0, 1)`).
pub(crate) fn judge(target: &Target, d: u32, u: f32, u2: f32) -> Verdict {
    match target {
        Target::Token(t) if *t == d => Verdict::Accept,
        Target::Token(t) => Verdict::Reject(*t),
        Target::Weights(vals) => {
            let sum: f32 = vals.iter().map(|(_, w)| *w).sum();
            let p_d = vals.iter().find(|(id, _)| *id == d).map(|(_, w)| *w).unwrap_or(0.0);
            if sum > 0.0 && u < p_d / sum {
                return Verdict::Accept;
            }
            let residual: Vec<(u32, f32)> = vals.iter().copied().filter(|(id, _)| *id != d).collect();
            if residual.is_empty() || residual.iter().all(|(_, w)| *w <= 0.0) {
                // p(d) == 1 up to rounding: there is nothing else to emit.
                return Verdict::Accept;
            }
            Verdict::Reject(draw(&residual, u2))
        }
    }
}

impl Engine {
    /// One speculative decode step: emit the next token plus every accepted draft token.
    pub(super) fn take_token_spec(&mut self, slot: &mut GenSlot, params: &SampleParams, spec: &SpecLookup) -> Result<()> {
        if self.budget_spent(slot) {
            return Ok(());
        }
        let t0 = match slot.pending.take() {
            Some(t) => t,
            None => self.pick(&slot.logits.values, slot, params),
        };
        if self.push_token(slot, t0, params) != Pushed::Continue {
            return Ok(());
        }
        let n0 = slot.state.n;
        let remaining = slot.max_tokens.saturating_sub(slot.all_ids.len() - slot.prompt_len);
        let room = self.model.n_ctx().saturating_sub(n0 + 1);
        let k = spec.draft_len.min(remaining).min(room);
        let drafted = draft(&slot.all_ids, k, spec.ngram_max, spec.ngram_min);
        if drafted.is_empty() {
            slot.logits = self.model.forward(&[t0], &[n0 as i32], &mut slot.state)?;
            self.budget_spent(slot);
            return Ok(());
        }

        let mut tokens = Vec::with_capacity(drafted.len() + 1);
        tokens.push(t0);
        tokens.extend_from_slice(&drafted);
        let pos: Vec<i32> = (n0 as i32..(n0 + tokens.len()) as i32).collect();
        let recurrent_snapshot = slot
            .state
            .recurrent
            .iter()
            .any(Option::is_some)
            .then(|| slot.state.recurrent.clone());
        let mut rows = self.model.forward_each(&tokens, &pos, &mut slot.state)?;
        self.spec_stats.verify_passes += 1;
        self.spec_stats.drafted += drafted.len() as u64;

        // rows[i] = logits after tokens[..=i], i.e. the target for drafted[i].
        let mut accepted = 0usize;
        let mut correction = None;
        for (i, &d) in drafted.iter().enumerate() {
            let masked = self.masked(&rows[i].values, slot, params);
            let verdict = judge(&target(&masked, &slot.all_ids, params), d, slot.rng.f32(), slot.rng.f32());
            match verdict {
                Verdict::Reject(t) => {
                    correction = Some(t);
                    break;
                }
                Verdict::Accept => match self.push_token(slot, d, params) {
                    Pushed::Eos => break,
                    Pushed::Stopped => {
                        accepted += 1;
                        break;
                    }
                    Pushed::Continue => {
                        accepted += 1;
                        if self.budget_spent(slot) {
                            break;
                        }
                    }
                },
            }
        }
        self.spec_stats.accepted += accepted as u64;

        let keep = n0 + 1 + accepted;
        if keep < slot.state.n {
            match recurrent_snapshot {
                Some(snapshot) => {
                    slot.state.recurrent = snapshot;
                    slot.state.n = n0;
                    let _ = self.model.forward(&tokens[..1 + accepted], &pos[..1 + accepted], &mut slot.state)?;
                }
                None => slot.state.n = keep,
            }
        }
        slot.logits = rows.swap_remove(accepted);
        if !slot.done {
            slot.pending = correction;
        }
        Ok(())
    }
}

/// A fresh per-sequence RNG for acceptance tests (distinct from the sampler's fixed draw).
pub(crate) fn slot_rng(params: &SampleParams, prompt_len: usize) -> Rng {
    Rng(params.seed ^ 0xa076_1d64_78bd_642f ^ (prompt_len as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_copies_what_followed_the_last_trigram() {
        // … 1 2 3 4 5 6 … 1 2 3 → propose 4 5 6 7.
        let ctx = [9, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3];
        assert_eq!(draft(&ctx, 4, 3, 2), vec![4, 5, 6, 7]);
        assert_eq!(draft(&ctx, 2, 3, 2), vec![4, 5]);
        assert!(draft(&ctx, 0, 3, 2).is_empty());
    }

    #[test]
    fn draft_prefers_most_recent_match_and_falls_back_to_bigram() {
        // Trigram 7 1 2 never repeats; bigram 1 2 does — the later occurrence wins.
        let ctx = [1, 2, 10, 11, 1, 2, 20, 21, 7, 1, 2];
        assert_eq!(draft(&ctx, 2, 3, 2), vec![20, 21]);
        // No repeat at all → no draft.
        assert!(draft(&[1, 2, 3, 4, 5], 4, 3, 2).is_empty());
        // The continuation may run into the tail (periodic text).
        assert_eq!(draft(&[5, 6, 5, 6, 5, 6], 4, 3, 2), vec![5, 6]);
    }

    #[test]
    fn greedy_judging_is_exact() {
        assert_eq!(judge(&Target::Token(7), 7, 0.99, 0.5), Verdict::Accept);
        assert_eq!(judge(&Target::Token(7), 3, 0.0, 0.5), Verdict::Reject(7));
    }

    #[test]
    fn sampled_judging_follows_target_probability() {
        let t = Target::Weights(vec![(1, 0.75), (2, 0.25)]);
        assert_eq!(judge(&t, 1, 0.5, 0.0), Verdict::Accept, "u < p(d)");
        // Rejected: the replacement comes from the residual (token 1 removed).
        assert_eq!(judge(&t, 1, 0.9, 0.3), Verdict::Reject(2));
        // A token outside the support is always rejected.
        assert_eq!(judge(&t, 9, 0.0, 0.0), Verdict::Reject(1));
        // p(d) = 1: nothing else to emit.
        assert_eq!(judge(&Target::Weights(vec![(4, 1.0)]), 4, 0.999_999, 0.0), Verdict::Accept);
    }

    /// Acceptance + residual reproduces the target distribution (Monte Carlo).
    #[test]
    fn speculative_acceptance_preserves_distribution() {
        let t = Target::Weights(vec![(0, 0.5), (1, 0.3), (2, 0.2)]);
        let mut rng = Rng(12345);
        let mut counts = [0usize; 3];
        let n = 60_000;
        for _ in 0..n {
            let emitted = match judge(&t, 1, rng.f32(), rng.f32()) {
                Verdict::Accept => 1,
                Verdict::Reject(x) => x,
            };
            counts[emitted as usize] += 1;
        }
        for (i, want) in [0.5, 0.3, 0.2].iter().enumerate() {
            let got = counts[i] as f64 / n as f64;
            assert!((got - want).abs() < 0.015, "token {i}: {got} vs {want}");
        }
    }

    #[test]
    fn acceptance_rate() {
        let s = SpecStats { verify_passes: 2, drafted: 8, accepted: 6 };
        assert!((s.acceptance() - 0.75).abs() < 1e-12);
        assert_eq!(SpecStats::default().acceptance(), 0.0);
    }
}
