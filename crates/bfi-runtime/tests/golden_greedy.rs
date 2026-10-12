//! M2 greedy 20-prompt corpus vs llama.cpp (Qwen3-0.6B, 256 tokens).
//!
//! CI stays green without weights. Live:
//!
//! ```sh
//! BFI_GOLDEN_GGUF=/path/to/qwen3-0.6b.gguf cargo test -p bfi-runtime --test golden_greedy -- --ignored --nocapture
//! ```
//!
//! Optional dumps (little-endian f32, one file per prompt):
//! `BFI_LLAMA_LOGITS=/dir/00.f32` … `19.f32` — last-token logits, cosine ≥ 0.99
//! (`1 - 1e-2`). Optional `BFI_LLAMA_IDS=/dir/00.ids` newline/space token ids
//! must match top-1 every step.

use std::path::{Path, PathBuf};

use bfi_runtime::{Engine, GreedyTrace};

fn golden_gguf_path() -> Option<PathBuf> {
    let p = std::env::var("BFI_GOLDEN_GGUF").ok()?;
    let path = PathBuf::from(p);
    path.is_file().then_some(path)
}

/// Twenty prompts spanning latin, code, CJK, emoji, tool-call XML.
fn prompts() -> [&'static str; 20] {
    [
        "Hello, world.",
        "Explain what a mutex is in one sentence.",
        "fn fib(n: u32) -> u32 {",
        "def greet(name):\n    return f\"hi {name}\"",
        "SELECT count(*) FROM users WHERE active = 1;",
        "你好，请用一句话介绍量子计算。",
        "日本語で自己紹介してください。",
        "한글 테스트: 오늘 날씨는?",
        "Ship it 🚀 then write a haiku.",
        "Emoji soup 😀🎉🔥💯 — continue.",
        "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.rs\"}}</tool_call>",
        "Call search_code for Tokenizer::encode then stop.",
        "The capital of France is",
        "2 + 2 =",
        "Once upon a time in a small village",
        "JSON: {\"ok\": true, \"n\":",
        "Qwen3 system: you are a careful coder.\nUser: fix the off-by-one.",
        "List three prime numbers greater than 50.",
        "Translate to German: the cat sat on the mat.",
        "Write a Rust match on Option<u32>.",
    ]
}

#[test]
fn greedy_prompts_are_twenty_and_mixed() {
    let p = prompts();
    assert_eq!(p.len(), 20);
    assert!(p.iter().any(|s| s.contains("fn fib")));
    assert!(p.iter().any(|s| s.contains('你')));
    assert!(p.iter().any(|s| s.contains('🚀')));
    assert!(p.iter().any(|s| s.contains("<tool_call>")));
}

#[test]
fn golden_greedy_skips_without_env() {
    if golden_gguf_path().is_none() {
        eprintln!("skip: set BFI_GOLDEN_GGUF to a Qwen3-0.6B GGUF to run --ignored");
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let d = na.sqrt() * nb.sqrt();
    if d == 0.0 {
        0.0
    } else {
        dot / d
    }
}

fn read_f32_le(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "{}", path.display());
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_ids(path: &Path) -> Vec<u32> {
    let s = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    s.split(|c: char| !c.is_ascii_digit())
        .filter(|t| !t.is_empty())
        .filter_map(|t| t.parse().ok())
        .collect()
}

#[test]
#[ignore]
fn golden_greedy_twenty_prompts_vs_llama() {
    let Some(path) = golden_gguf_path() else {
        eprintln!("skip: BFI_GOLDEN_GGUF not set");
        return;
    };
    let mut engine = Engine::load(&path).expect("Engine::load");
    let logits_dir = std::env::var("BFI_LLAMA_LOGITS").ok().map(PathBuf::from);
    let ids_dir = std::env::var("BFI_LLAMA_IDS").ok().map(PathBuf::from);
    for (i, prompt) in prompts().iter().enumerate() {
        let GreedyTrace {
            token_ids,
            text,
            last_logits,
        } = engine
            .generate_greedy_trace(prompt, 256)
            .unwrap_or_else(|e| panic!("prompt {i}: {e}"));
        eprintln!(
            "prompt {i}: {} gen tokens, {} chars",
            token_ids.len(),
            text.len()
        );
        if let Some(dir) = ids_dir.as_ref() {
            let theirs = read_ids(&dir.join(format!("{i:02}.ids")));
            assert_eq!(
                token_ids, theirs,
                "prompt {i} top-1 mismatch vs llama.cpp ids"
            );
        }
        if let Some(dir) = logits_dir.as_ref() {
            let theirs = read_f32_le(&dir.join(format!("{i:02}.f32")));
            let c = cosine(&last_logits, &theirs);
            assert!(
                c >= 0.99,
                "prompt {i} last-token logits cosine {c} < 0.99 (≤1e-2 drift)"
            );
        }
    }
    if logits_dir.is_none() && ids_dir.is_none() {
        eprintln!("ran 20 greedy traces; set BFI_LLAMA_IDS / BFI_LLAMA_LOGITS to compare");
    }
}
