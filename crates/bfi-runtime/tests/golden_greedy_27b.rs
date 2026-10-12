//! M3 greedy 10-prompt corpus vs PrismML llama.cpp fork (Bonsai 2 27B, 128 tokens).
//!
//! CI stays green without weights. Live:
//!
//! ```sh
//! BFI_GOLDEN_GGUF=/path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
//!   cargo test -p bfi-runtime --test golden_greedy_27b -- --ignored --nocapture
//! ```
//!
//! `BFI_GOLDEN_GGUF_27B` overrides `BFI_GOLDEN_GGUF` so a 0.6B M2 file can stay
//! set. The ignored test also skips if the GGUF architecture is not `qwen35`.
//!
//! Optional dumps (little-endian f32, one file per prompt):
//! `BFI_LLAMA_LOGITS=/dir/00.f32` … `09.f32` — last-token logits, cosine ≥ 0.99.
//! Optional `BFI_LLAMA_IDS=/dir/00.ids` newline/space token ids must match top-1
//! every step. `BFI_LLAMA_LOGITS_27B` / `BFI_LLAMA_IDS_27B` override those dirs.
//!
//! This harness does **not** claim live 27B vs-fork numbers unless those env
//! vars point at a real Bonsai GGUF and (optionally) fork dumps.

use std::path::{Path, PathBuf};

use bfi_gguf::GgufFile;
use bfi_runtime::{Engine, GreedyTrace};

fn golden_gguf_27b_path() -> Option<PathBuf> {
    for key in ["BFI_GOLDEN_GGUF_27B", "BFI_GOLDEN_GGUF"] {
        if let Ok(p) = std::env::var(key) {
            let path = PathBuf::from(p);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    None
}

/// Ten prompts spanning latin, code, CJK, emoji, tool-call XML.
fn prompts() -> [&'static str; 10] {
    [
        "Hello, world.",
        "Explain what a mutex is in one sentence.",
        "fn fib(n: u32) -> u32 {",
        "SELECT count(*) FROM users WHERE active = 1;",
        "你好，请用一句话介绍量子计算。",
        "Ship it 🚀 then write a haiku.",
        "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.rs\"}}</tool_call>",
        "The capital of France is",
        "2 + 2 =",
        "Once upon a time in a small village",
    ]
}

fn env_dir(keys: &[&str]) -> Option<PathBuf> {
    for key in keys {
        if let Ok(p) = std::env::var(key) {
            let path = PathBuf::from(p);
            if path.is_dir() {
                return Some(path);
            }
        }
    }
    None
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

fn is_qwen35(path: &Path) -> bool {
    match GgufFile::open(path) {
        Ok(gguf) => gguf
            .header()
            .architecture()
            .map(|a| a == "qwen35")
            .unwrap_or(false),
        Err(_) => false,
    }
}

#[test]
fn greedy_27b_prompts_are_ten_and_mixed() {
    let p = prompts();
    assert_eq!(p.len(), 10);
    assert!(p.iter().any(|s| s.contains("fn fib")));
    assert!(p.iter().any(|s| s.contains('你')));
    assert!(p.iter().any(|s| s.contains('🚀')));
    assert!(p.iter().any(|s| s.contains("<tool_call>")));
}

#[test]
fn golden_greedy_27b_skips_without_env() {
    if golden_gguf_27b_path().is_none() {
        eprintln!(
            "skip: set BFI_GOLDEN_GGUF or BFI_GOLDEN_GGUF_27B to a Bonsai-27B GGUF to run --ignored"
        );
    }
}

#[test]
#[ignore]
fn golden_greedy_27b_ten_prompts_vs_fork() {
    let Some(path) = golden_gguf_27b_path() else {
        eprintln!("skip: BFI_GOLDEN_GGUF / BFI_GOLDEN_GGUF_27B not set");
        return;
    };
    if !is_qwen35(&path) {
        eprintln!(
            "skip: {} is not qwen35 (point BFI_GOLDEN_GGUF_27B at Bonsai 2 27B)",
            path.display()
        );
        return;
    }
    let mut engine = Engine::load(&path).expect("Engine::load");
    let logits_dir = env_dir(&["BFI_LLAMA_LOGITS_27B", "BFI_LLAMA_LOGITS"]);
    let ids_dir = env_dir(&["BFI_LLAMA_IDS_27B", "BFI_LLAMA_IDS"]);
    for (i, prompt) in prompts().iter().enumerate() {
        let GreedyTrace {
            token_ids,
            text,
            last_logits,
        } = engine
            .generate_greedy_trace(prompt, 128)
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
                "prompt {i} top-1 mismatch vs PrismML fork ids"
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
        eprintln!(
            "ran 10 greedy traces (128 tok); set BFI_LLAMA_IDS_27B / BFI_LLAMA_LOGITS_27B to compare"
        );
    }
}
