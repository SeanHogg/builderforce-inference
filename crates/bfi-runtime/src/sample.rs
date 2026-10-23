//! Token sampler: greedy, temperature, top-k/p, min-p, repetition penalty,
//! and a small constrained-decode mask (GBNF subset / JSON object).

use std::collections::HashSet;

use crate::tools::ToolChoice;

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
    /// GBNF grammar. Only a documented subset is enforced (see [`apply_constraint`]).
    pub grammar: Option<String>,
    /// OpenAI `response_format: json_object` — decoded text must stay a JSON-object prefix.
    pub json_object: bool,
    /// `required` / named `tool_choice` constrain decode to a JSON tool call.
    pub tool_choice: ToolChoice,
    /// Allowed function names from the request `tools` array.
    pub tool_names: Vec<String>,
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
            grammar: None,
            json_object: false,
            tool_choice: ToolChoice::None,
            tool_names: Vec::new(),
        }
    }
}

/// Constrained decoding (M4).
///
/// Grammar subset understood here (not full GBNF):
/// - `root ::= id(N) | id(M) | ...` — only those token ids may be sampled
///   (no tokenizer; used by unit tests and as an escape hatch).
/// - `root ::= "literal"` — decoded pieces must stay a prefix of `literal`
///   (needs [`apply_constraint`] + a decode fn; Engine passes tokenizer.decode).
/// - `json_object: true` — decoded text must remain a prefix of a JSON object
///   `{ ... }`. Brace/string scan only; not JSON Schema.
///
/// Anything else is treated as unconstrained (full GBNF is out of scope).
pub fn apply_constraint(
    logits: &mut [f32],
    generated: &[u32],
    p: &SampleParams,
    decode: impl Fn(u32) -> String,
) {
    if p.grammar.is_none() && !p.json_object && !p.tool_choice.constrains() {
        return;
    }
    if let Some(g) = p.grammar.as_deref() {
        if let Some(allowed) = parse_id_union(g) {
            for (i, l) in logits.iter_mut().enumerate() {
                if !allowed.contains(&(i as u32)) {
                    *l = f32::NEG_INFINITY;
                }
            }
            return;
        }
    }
    let so_far: String = generated.iter().map(|&id| decode(id)).collect();
    for (i, l) in logits.iter_mut().enumerate() {
        let piece = decode(i as u32);
        if !token_allowed(&so_far, &piece, p) {
            *l = f32::NEG_INFINITY;
        }
    }
}

fn token_allowed(so_far: &str, piece: &str, p: &SampleParams) -> bool {
    let next = format!("{so_far}{piece}");
    if (p.json_object || p.tool_choice.constrains()) && !json_object_prefix_ok(&next) {
        return false;
    }
    if p.tool_choice.constrains() && !tool_call_prefix_ok(&next, &p.tool_choice, &p.tool_names) {
        return false;
    }
    if let Some(g) = p.grammar.as_deref() {
        if parse_id_union(g).is_some() {
            return true;
        }
        if let Some(lit) = parse_literal_root(g) {
            return lit.starts_with(&next);
        }
    }
    true
}

/// `name` field of a JSON tool-call prefix must stay on an allowed / required name.
pub(crate) fn tool_call_prefix_ok(s: &str, choice: &ToolChoice, names: &[String]) -> bool {
    if !json_object_prefix_ok(s) {
        return false;
    }
    let required = match choice {
        ToolChoice::Named(n) => Some(n.as_str()),
        ToolChoice::Required => None,
        _ => return true,
    };
    match json_name_so_far(s) {
        NameSoFar::Missing => true,
        NameSoFar::Partial(p) => {
            if let Some(req) = required {
                req.starts_with(&p)
            } else {
                names.is_empty() || names.iter().any(|n| n.starts_with(&p))
            }
        }
        NameSoFar::Complete(n) => {
            if let Some(req) = required {
                n == req
            } else {
                names.is_empty() || names.iter().any(|t| t == &n)
            }
        }
        NameSoFar::Invalid => false,
    }
}

enum NameSoFar {
    Missing,
    Partial(String),
    Complete(String),
    Invalid,
}

fn json_name_so_far(s: &str) -> NameSoFar {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i + 6 <= bytes.len() {
        if &bytes[i..i + 6] == b"\"name\"" {
            let mut j = i + 6;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j >= bytes.len() {
                return NameSoFar::Missing;
            }
            if bytes[j] != b':' {
                i += 1;
                continue;
            }
            j += 1;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j >= bytes.len() {
                return NameSoFar::Partial(String::new());
            }
            if bytes[j] != b'"' {
                return NameSoFar::Invalid;
            }
            j += 1;
            let val_start = j;
            let mut escaped = false;
            let mut closed = false;
            while j < bytes.len() {
                let c = bytes[j];
                if escaped {
                    escaped = false;
                    j += 1;
                    continue;
                }
                if c == b'\\' {
                    escaped = true;
                    j += 1;
                    continue;
                }
                if c == b'"' {
                    closed = true;
                    break;
                }
                j += 1;
            }
            let partial = String::from_utf8_lossy(&bytes[val_start..j]).into_owned();
            return if closed {
                NameSoFar::Complete(partial)
            } else {
                NameSoFar::Partial(partial)
            };
        }
        i += 1;
    }
    NameSoFar::Missing
}

/// `root ::= id(1) | id(2)` — token-id union, no tokenizer required.
fn parse_id_union(g: &str) -> Option<HashSet<u32>> {
    let rest = strip_root(g)?;
    let mut set = HashSet::new();
    for part in rest.split('|') {
        let p = part.trim();
        let inner = p.strip_prefix("id(")?.strip_suffix(')')?.trim();
        set.insert(inner.parse().ok()?);
    }
    if set.is_empty() {
        None
    } else {
        Some(set)
    }
}

/// `root ::= "hello"` — character-literal (no escapes besides the closing quote).
fn parse_literal_root(g: &str) -> Option<String> {
    let rest = strip_root(g)?;
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn strip_root(g: &str) -> Option<&str> {
    let s = g.trim();
    let rest = s.strip_prefix("root")?.trim();
    Some(rest.strip_prefix("::=")?.trim())
}

/// True if `s` is empty/whitespace, a complete JSON object, or a brace-balanced
/// prefix of one (incomplete strings allowed). First non-ws char must be `{`.
fn json_object_prefix_ok(s: &str) -> bool {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() && b[i].is_ascii_whitespace() {
        i += 1;
    }
    if i == b.len() {
        return true;
    }
    if b[i] != b'{' {
        return false;
    }
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    while i < b.len() {
        let c = b[i];
        if in_string {
            if escape {
                escape = false;
            } else if c == b'\\' {
                escape = true;
            } else if c == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        match c {
            b'"' => in_string = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
                if depth == 0 {
                    i += 1;
                    while i < b.len() {
                        if !b[i].is_ascii_whitespace() {
                            return false;
                        }
                        i += 1;
                    }
                    return true;
                }
            }
            _ => {}
        }
        i += 1;
    }
    in_string || escape || depth > 0
}

/// What [`sample`] draws from after penalties, temperature and top-k/p/min-p: a forced
/// token (greedy, or a filter that left nothing) or `(id, weight)` pairs whose weights
/// need not sum to 1. Speculative verification judges a drafted token against exactly
/// this distribution, so the two can never disagree about what "the target" is.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Target {
    Token(u32),
    Weights(Vec<(u32, f32)>),
}

/// Sample a token id from `logits`. Greedy when `temperature == 0`.
pub fn sample(logits: &[f32], prev: &[u32], p: &SampleParams) -> u32 {
    match target(logits, prev, p) {
        Target::Token(id) => id,
        Target::Weights(vals) => draw(&vals, Rng(p.seed).f32()),
    }
}

/// Inverse-CDF pick from `(id, weight)` pairs at `u ∈ [0, 1)`.
pub(crate) fn draw(vals: &[(u32, f32)], u: f32) -> u32 {
    let sum: f32 = vals.iter().map(|(_, pr)| *pr).sum();
    let r = u * sum.max(0.0);
    let mut cum = 0.0f32;
    for (id, prob) in vals {
        cum += *prob;
        if r <= cum {
            return *id;
        }
    }
    vals.last().map(|(id, _)| *id).unwrap_or(0)
}

/// The distribution [`sample`] draws from (see [`Target`]).
pub(crate) fn target(logits: &[f32], prev: &[u32], p: &SampleParams) -> Target {
    if logits.is_empty() {
        return Target::Token(0);
    }

    let mut masked: Vec<f32> = logits.to_vec();
    if let Some(g) = p.grammar.as_deref() {
        if parse_id_union(g).is_some() {
            apply_constraint(&mut masked, prev, p, |_| String::new());
        }
    }
    let logits = masked.as_slice();

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
        return Target::Token(argmax(&vals));
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
        let max_p = vals.iter().map(|(_, pr)| *pr).fold(0.0f32, f32::max);
        let thresh = p.min_p * max_p;
        vals.retain(|(_, pr)| *pr >= thresh);
        if vals.is_empty() {
            return Target::Token(argmax(
                &logits
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i as u32, v))
                    .collect::<Vec<_>>(),
            ));
        }
    }

    Target::Weights(vals)
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
pub(crate) struct Rng(pub(crate) u64);

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

    pub(crate) fn f32(&mut self) -> f32 {
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
        assert!(p.grammar.is_none());
        assert!(!p.json_object);
        assert_eq!(p.tool_choice, ToolChoice::None);
        assert!(p.tool_names.is_empty());
        assert_eq!(sample(&[1.0, 0.0, 0.5], &[], &p), 0);
    }

    #[test]
    fn sample_params_carries_grammar() {
        let p = SampleParams {
            grammar: Some("root ::= id(1)".into()),
            json_object: true,
            ..Default::default()
        };
        assert_eq!(p.grammar.as_deref(), Some("root ::= id(1)"));
        assert!(p.json_object);
    }

    #[test]
    fn grammar_id_mask_rejects_forbidden_token() {
        let p = SampleParams {
            temperature: 0.0,
            grammar: Some("root ::= id(1)".into()),
            ..Default::default()
        };
        // token 0 has the highest logit but is forbidden
        assert_eq!(sample(&[9.0, 1.0, 8.0], &[], &p), 1);
    }

    #[test]
    fn json_object_constraint_invoked() {
        let p = SampleParams {
            json_object: true,
            temperature: 0.0,
            ..Default::default()
        };
        // token 0 = "x" (illegal first char), token 1 = "{"
        let decode = |id: u32| if id == 1 { "{".into() } else { "x".into() };
        let mut logits = [9.0, 1.0];
        apply_constraint(&mut logits, &[], &p, decode);
        assert_eq!(sample(&logits, &[], &p), 1);
    }

    #[test]
    fn required_tool_choice_forces_json_object() {
        let p = SampleParams {
            tool_choice: ToolChoice::Required,
            tool_names: vec!["lookup".into()],
            temperature: 0.0,
            ..Default::default()
        };
        let decode = |id: u32| if id == 1 { "{".into() } else { "x".into() };
        let mut logits = [9.0, 1.0];
        apply_constraint(&mut logits, &[], &p, decode);
        assert_eq!(sample(&logits, &[], &p), 1);
    }

    #[test]
    fn named_tool_choice_rejects_other_name() {
        let p = SampleParams {
            tool_choice: ToolChoice::Named("lookup".into()),
            tool_names: vec!["lookup".into(), "other".into()],
            temperature: 0.0,
            ..Default::default()
        };
        let decode = |id: u32| match id {
            0 => r#"{"name":"other"}"#.into(),
            1 => r#"{"name":"lookup"}"#.into(),
            _ => String::new(),
        };
        let mut logits = [9.0, 1.0];
        apply_constraint(&mut logits, &[], &p, decode);
        assert_eq!(sample(&logits, &[], &p), 1);
    }

    #[test]
    fn literal_grammar_stays_on_prefix() {
        let p = SampleParams {
            grammar: Some(r#"root ::= "ab""#.into()),
            temperature: 0.0,
            ..Default::default()
        };
        let decode = |id: u32| match id {
            0 => "a".into(),
            1 => "z".into(),
            2 => "b".into(),
            _ => String::new(),
        };
        let mut logits = [1.0, 9.0, 2.0];
        apply_constraint(&mut logits, &[], &p, decode);
        assert_eq!(sample(&logits, &[], &p), 0);
        let mut logits2 = [1.0, 9.0, 2.0];
        apply_constraint(&mut logits2, &[0], &p, decode);
        assert_eq!(sample(&logits2, &[0], &p), 2);
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
    fn top_k_with_temperature_never_picks_outside_k() {
        // Greedy (temp=0) returns before top-k, so this is the lock that
        // top-k actually truncates the distribution.
        let logits = [10.0, 9.0, 8.0, 7.0];
        for seed in 0..32 {
            let p = SampleParams {
                temperature: 5.0,
                top_k: 1,
                seed,
                ..Default::default()
            };
            assert_eq!(sample(&logits, &[], &p), 0, "top_k=1 seed {seed}");
        }
        for seed in 0..32 {
            let p = SampleParams {
                temperature: 5.0,
                top_k: 2,
                seed,
                ..Default::default()
            };
            let id = sample(&logits, &[], &p);
            assert!(id <= 1, "top_k=2 seed {seed} picked {id}");
        }
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
