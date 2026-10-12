//! M2 tokenizer golden: 2,000-line mixed corpus vs llama.cpp `llama_tokenize`.
//!
//! CI `cargo test -p bfi-runtime` does not need weights. Live compare:
//!
//! ```sh
//! BFI_GOLDEN_GGUF=/path/to/qwen3.gguf cargo test -p bfi-runtime --test golden_tokenize -- --ignored --nocapture
//! ```
//!
//! Optional `BFI_LLAMA_TOKENIZE=/path/to/llama-tokenize` (or a `llama-tokenize`
//! on PATH) dumps ids per line; mismatch fails the ignored test.

use std::path::{Path, PathBuf};
use std::process::Command;

use bfi_gguf::GgufFile;
use bfi_runtime::Tokenizer;

fn golden_gguf_path() -> Option<PathBuf> {
    let p = std::env::var("BFI_GOLDEN_GGUF").ok()?;
    let path = PathBuf::from(p);
    path.is_file().then_some(path)
}

/// 2,000 lines: code, CJK, emoji, tool-call XML, latin — rotated.
fn corpus() -> Vec<String> {
    const CODE: &[&str] = &[
        "fn main() { println!(\"hi N\"); }",
        "def f(x): return x + N",
        "const x: u32 = N; let y = x * 2;",
        "SELECT id FROM t WHERE n = N;",
        "export function f(a: number) { return a + N; }",
    ];
    const CJK: &[&str] = &[
        "你好世界第N次",
        "日本語テストN番",
        "한글 테스트 N",
        "中文代码注释：变量N",
    ];
    const EMOJI: &[&str] = &[
        "ship it 🚀 #N",
        "hello 😀🎉 line N",
        "🔥 fire N 💯",
        "ok ✅ done N",
    ];
    const TOOLS: &[&str] = &[
        "<tool_call>{\"name\":\"read_file\",\"id\":N}</tool_call>",
        "<tool_call name=\"search_code\" i=\"N\"><query>foo</query></tool_call>",
        "assistant: I'll call tools\n<tool_call>list_files N</tool_call>",
        "<tool_call><function=write_file><parameter=path>a.rs</parameter></function></tool_call> N",
    ];
    const LATIN: &[&str] = &[
        "The quick brown fox jumps N times.",
        "Lorem ipsum N dolor sit amet.",
        "Mixed CASE and digits 42-N.",
        "She said \"hello N\" — then left.",
    ];
    let families: [&[&str]; 5] = [CODE, CJK, EMOJI, TOOLS, LATIN];
    let mut out = Vec::with_capacity(2000);
    let mut n = 0u32;
    while out.len() < 2000 {
        let fam = families[out.len() % families.len()];
        let tmpl = fam[(out.len() / families.len()) % fam.len()];
        out.push(tmpl.replace('N', &n.to_string()));
        n += 1;
    }
    out
}

#[test]
fn corpus_covers_categories_and_length() {
    let lines = corpus();
    assert_eq!(lines.len(), 2000);
    assert!(
        lines.iter().any(|l| l.contains("fn main") || l.contains("def f")),
        "code"
    );
    assert!(
        lines
            .iter()
            .any(|l| l.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))),
        "CJK"
    );
    assert!(
        lines.iter().any(|l| l.contains('🚀') || l.contains('😀')),
        "emoji"
    );
    assert!(
        lines.iter().any(|l| l.contains("<tool_call>")),
        "tool-call XML"
    );
}

#[test]
fn golden_tokenize_skips_without_env() {
    if golden_gguf_path().is_none() {
        eprintln!("skip: set BFI_GOLDEN_GGUF to a qwen3 GGUF to run --ignored");
    }
}

fn llama_tokenize_bin() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BFI_LLAMA_TOKENIZE") {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Some(path);
        }
    }
    which("llama-tokenize")
}

fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
        let exe = dir.join(format!("{name}.exe"));
        if exe.is_file() {
            return Some(exe);
        }
    }
    None
}

fn llama_ids(bin: &Path, model: &Path, text: &str) -> Result<Vec<u32>, String> {
    let out = Command::new(bin)
        .arg("-m")
        .arg(model)
        .arg("--ids")
        .arg("-p")
        .arg(text)
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "llama-tokenize {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    Ok(stdout
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok())
        .collect())
}

#[test]
#[ignore]
fn golden_tokenize_matches_llama_tokenize() {
    let Some(path) = golden_gguf_path() else {
        eprintln!("skip: BFI_GOLDEN_GGUF not set");
        return;
    };
    let gguf = GgufFile::open(&path).expect("open gguf");
    let tok = Tokenizer::from_header(gguf.header()).expect("tokenizer from header");
    let lines = corpus();
    let llama = llama_tokenize_bin();
    let mut compared = 0usize;
    for (i, line) in lines.iter().enumerate() {
        let ours = tok.encode(line, true).unwrap_or_else(|e| panic!("encode {i}: {e}"));
        if line.is_empty() {
            continue;
        }
        assert!(
            !ours.is_empty(),
            "line {i} encoded to no tokens: {line:?}"
        );
        if let Some(bin) = llama.as_ref() {
            let theirs = llama_ids(bin, &path, line)
                .unwrap_or_else(|e| panic!("llama-tokenize line {i}: {e}"));
            assert_eq!(
                ours, theirs,
                "line {i} id mismatch\ntext={line:?}\nours={ours:?}\ntheirs={theirs:?}"
            );
            compared += 1;
        }
    }
    if llama.is_none() {
        eprintln!(
            "encoded {} lines; set BFI_LLAMA_TOKENIZE to compare against llama.cpp",
            lines.len()
        );
    } else {
        eprintln!("matched llama-tokenize on {compared} lines");
    }
}
