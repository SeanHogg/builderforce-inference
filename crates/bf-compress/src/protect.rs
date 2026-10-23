//! Words compression must never drop: the ones whose loss changes meaning rather than
//! style — file paths, URLs, inline code, numbers (with or without a unit), and anything
//! the caller names. A classifier trained on meeting transcripts cannot be trusted with
//! `api/src/billing.ts` or `512 MB`, so these bypass it entirely.

/// Unit words that may follow a bare number as their own word (`512 MB`, `30 ms`).
const UNITS: &[&str] = &[
    "%", "b", "kb", "mb", "gb", "tb", "kib", "mib", "gib", "tib", "bit", "bits", "byte", "bytes", "ns", "us", "µs", "ms",
    "s", "sec", "secs", "second", "seconds", "min", "mins", "minute", "minutes", "h", "hr", "hrs", "hour", "hours", "day",
    "days", "week", "weeks", "month", "months", "year", "years", "px", "em", "rem", "pt", "dpi", "hz", "khz", "mhz",
    "ghz", "k", "m", "x", "token", "tokens", "line", "lines", "file", "files", "usd", "eur", "gbp", "$", "€", "£", "°c",
    "°f", "kg", "g", "mg", "km", "cm", "mm", "rps", "qps", "rpm", "fps", "cpu", "cpus", "core", "cores", "gpu", "gpus",
];

/// Leading/trailing punctuation that does not belong to the word itself.
fn core(word: &str) -> &str {
    word.trim_matches(|c: char| matches!(c, ',' | ';' | ':' | '!' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '"' | '\'' | '<' | '>'))
        .trim_end_matches('.')
}

fn has_digit(word: &str) -> bool {
    word.chars().any(|c| c.is_ascii_digit())
}

/// A bare number: `512`, `3.5`, `1,000`, `-2`.
fn is_number(word: &str) -> bool {
    let w = core(word);
    !w.is_empty() && has_digit(w) && w.chars().all(|c| c.is_ascii_digit() || matches!(c, '.' | ',' | '-' | '+'))
}

fn is_unit(word: &str) -> bool {
    let w = core(word).to_lowercase();
    UNITS.contains(&w.as_str())
}

/// `src/a.ts`, `C:\repo`, `./x`, `~/.config`, `Cargo.toml`, `https://…`.
fn is_path_like(word: &str) -> bool {
    let w = core(word);
    if w.contains("://") || w.starts_with("./") || w.starts_with("../") || w.starts_with("~/") {
        return true;
    }
    let has_sep = w.contains('/') || w.contains('\\');
    // A separator plus real name characters. "and/or" is caught too, which costs one word.
    if has_sep && w.chars().filter(|c| c.is_alphanumeric()).count() >= 2 {
        return true;
    }
    // `name.ext` — a dotted word whose last part is a short extension: Cargo.toml, a.rs.
    if let Some((stem, ext)) = w.rsplit_once('.') {
        let ext_ok = (1..=5).contains(&ext.len()) && ext.chars().all(|c| c.is_ascii_alphanumeric());
        let stem_ok = !stem.is_empty() && stem.chars().all(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | '.'));
        if ext_ok && stem_ok && ext.chars().any(|c| c.is_ascii_alphabetic()) {
            return true;
        }
    }
    false
}

/// Inline code (`` `x` ``), identifiers in snake_case / with `::`, `()` calls.
fn is_code_like(word: &str) -> bool {
    word.contains('`') || word.contains("::") || word.contains("()") || core(word).contains('_')
}

/// Whether `word` must be kept. `prev` is the word before it (a unit after a number).
pub fn is_protected(word: &str, prev: Option<&str>, force_keep: &[&str]) -> bool {
    if force_keep.iter().any(|f| !f.is_empty() && word.contains(f)) {
        return true;
    }
    if has_digit(word) || is_path_like(word) || is_code_like(word) {
        return true;
    }
    matches!(prev, Some(p) if is_number(p)) && is_unit(word)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(w: &str) -> bool {
        is_protected(w, None, &[])
    }

    #[test]
    fn paths_urls_and_code_are_protected() {
        for w in ["api/src/billing.ts", "C:\\repo\\x", "./run.sh", "~/.cargo", "Cargo.toml", "(main.rs)", "https://x.dev", "`foo`", "a::b", "call()", "snake_case"] {
            assert!(p(w), "{w} should be protected");
        }
        for w in ["the", "however,", "end.", "and"] {
            assert!(!p(w), "{w} should not be protected");
        }
    }

    #[test]
    fn numbers_and_their_units_are_protected() {
        for w in ["512MB", "3.5", "42%", "$20", "v2", "10ms"] {
            assert!(p(w), "{w}");
        }
        assert!(is_protected("MB", Some("512"), &[]));
        assert!(is_protected("seconds.", Some("30"), &[]));
        assert!(!is_protected("MB", Some("many"), &[]));
        assert!(!is_protected("apples", Some("3"), &[]));
    }

    #[test]
    fn force_keep_matches_inside_words() {
        assert!(is_protected("Stripe,", None, &["Stripe"]));
        assert!(!is_protected("stripe", None, &["Stripe"]));
        assert!(!is_protected("x", None, &[""]));
    }
}
