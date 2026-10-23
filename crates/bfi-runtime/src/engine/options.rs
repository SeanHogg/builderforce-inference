//! Engine knobs chosen at load time (and toggleable on a loaded engine).

/// Default prefix-cache budget: 1 GiB of cached K/V + recurrent state + logits.
pub const DEFAULT_PREFIX_CACHE_BYTES: usize = 1 << 30;

/// Prompt-lookup speculative decoding (PRD E4). Drafts come from the context itself, so
/// there is no draft model: the win is on outputs that copy their input (code edits,
/// quoting, JSON echo, RAG answers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecLookup {
    /// Tokens drafted per verify pass (k).
    pub draft_len: usize,
    /// Longest context n-gram matched first; shorter ones are the fallback.
    pub ngram_max: usize,
    pub ngram_min: usize,
}

impl Default for SpecLookup {
    fn default() -> Self {
        Self { draft_len: 4, ngram_max: 3, ngram_min: 2 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineOptions {
    /// `None` (default) = plain one-token-per-forward decode.
    pub spec_lookup: Option<SpecLookup>,
    /// Byte budget of the in-process prefix cache (0 disables it).
    pub prefix_cache_bytes: usize,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self { spec_lookup: None, prefix_cache_bytes: DEFAULT_PREFIX_CACHE_BYTES }
    }
}
