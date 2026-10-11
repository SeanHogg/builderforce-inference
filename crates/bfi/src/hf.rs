//! Hugging Face resolve + resumable ranged download + sha256.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use bfi_runtime::AliasMap;

const UA: &str = "bfi/2026.10.4";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    /// `org/name`, a filesystem path, or a leftover alias token.
    pub repo: String,
    pub quant: Option<String>,
}

/// Parse `repo[:quant]` after alias expansion. Windows `C:\...` is a path, not a quant.
pub fn parse_spec(raw: &str, aliases: &AliasMap) -> ModelSpec {
    let resolved = aliases
        .get(raw)
        .cloned()
        .unwrap_or_else(|| raw.to_string());
    if looks_like_path(&resolved) {
        return ModelSpec {
            repo: resolved,
            quant: None,
        };
    }
    if let Some((repo, quant)) = resolved.rsplit_once(':') {
        if !repo.is_empty() && !quant.is_empty() && !quant.contains('/') && !quant.contains('\\') {
            return ModelSpec {
                repo: repo.to_string(),
                quant: Some(quant.to_string()),
            };
        }
    }
    ModelSpec {
        repo: resolved,
        quant: None,
    }
}

pub fn looks_like_path(s: &str) -> bool {
    let p = Path::new(s);
    if p.exists() {
        return true;
    }
    s.ends_with(".gguf")
        || s.starts_with('.')
        || s.starts_with('/')
        || s.starts_with('\\')
        || (s.len() >= 2 && s.as_bytes().get(1) == Some(&b':') && !s.contains('/'))
}

pub fn hf_endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".into())
}

pub fn hf_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok())
        .filter(|s| !s.is_empty())
}

fn apply_auth(mut req: ureq::Request) -> ureq::Request {
    req = req.set("User-Agent", UA);
    if let Some(token) = hf_token() {
        req = req.set("Authorization", &format!("Bearer {token}"));
    }
    req
}

#[derive(Debug, Deserialize)]
struct HfEntry {
    #[serde(rename = "type")]
    kind: Option<String>,
    path: String,
    size: Option<u64>,
    lfs: Option<HfLfs>,
}

#[derive(Debug, Deserialize)]
struct HfLfs {
    oid: Option<String>,
    sha256: Option<String>,
    size: Option<u64>,
}

impl HfLfs {
    fn sha256(&self) -> Option<&str> {
        self.sha256
            .as_deref()
            .or(self.oid.as_deref())
            .map(str::trim)
            .filter(|s| !s.is_empty())
    }
}

#[derive(Debug, Clone)]
pub struct RemoteFile {
    pub path: String,
    pub url: String,
    pub size: Option<u64>,
    pub sha256: Option<String>,
}

/// Pick a GGUF in the repo matching `quant` (case-insensitive substring).
pub fn match_quant_file<'a>(paths: &'a [String], quant: Option<&str>) -> Option<&'a str> {
    let ggufs: Vec<&str> = paths
        .iter()
        .map(String::as_str)
        .filter(|p| p.to_ascii_lowercase().ends_with(".gguf"))
        .collect();
    if ggufs.is_empty() {
        return None;
    }
    let Some(q) = quant.filter(|s| !s.is_empty()) else {
        return ggufs.into_iter().min_by_key(|p| p.len());
    };
    let q = q.to_ascii_lowercase();
    let mut hits: Vec<&str> = ggufs
        .iter()
        .copied()
        .filter(|p| p.to_ascii_lowercase().contains(&q))
        .collect();
    if hits.is_empty() {
        return None;
    }
    hits.sort_by_key(|p| p.len());
    hits.first().copied()
}

pub fn dest_path(models_dir: &Path, spec: &ModelSpec, filename: &str) -> PathBuf {
    let model_dir = spec.repo.replace('/', "--");
    models_dir
        .join(model_dir)
        .join("resolve")
        .join("main")
        .join(filename)
}

pub fn list_repo_files(repo: &str) -> Result<Vec<RemoteFile>> {
    let endpoint = hf_endpoint();
    let url = format!("{endpoint}/api/models/{repo}/tree/main?recursive=1");
    let resp = apply_auth(ureq::get(&url))
        .call()
        .map_err(|e| hf_err(repo, e))?;
    let entries: Vec<HfEntry> = resp.into_json().context("HF tree JSON")?;
    let mut out = Vec::new();
    for e in entries {
        if e.kind.as_deref() == Some("directory") {
            continue;
        }
        let size = e.lfs.as_ref().and_then(|l| l.size).or(e.size);
        let sha = e.lfs.as_ref().and_then(|l| l.sha256().map(|s| s.to_string()));
        let file_url = format!("{endpoint}/{repo}/resolve/main/{}", e.path);
        out.push(RemoteFile {
            path: e.path,
            url: file_url,
            size,
            sha256: sha,
        });
    }
    Ok(out)
}

fn hf_err(repo: &str, e: ureq::Error) -> anyhow::Error {
    match &e {
        ureq::Error::Status(401, _) | ureq::Error::Status(403, _) => anyhow!(
            "Hugging Face denied {repo} (HTTP {status}). Set HF_TOKEN for gated repos.",
            status = match &e {
                ureq::Error::Status(s, _) => *s,
                _ => 0,
            }
        ),
        ureq::Error::Status(status, _) => anyhow!("Hugging Face HTTP {status} for {repo}"),
        _ => anyhow!("Hugging Face request for {repo}: {e}"),
    }
}

pub fn resolve_remote(spec: &ModelSpec) -> Result<RemoteFile> {
    let files = list_repo_files(&spec.repo)?;
    let paths: Vec<String> = files.iter().map(|f| f.path.clone()).collect();
    let picked = match_quant_file(&paths, spec.quant.as_deref())
        .ok_or_else(|| {
            anyhow!(
                "no GGUF matching {:?} in {}",
                spec.quant.as_deref().unwrap_or("(any)"),
                spec.repo
            )
        })?;
    files
        .into_iter()
        .find(|f| f.path == picked)
        .ok_or_else(|| anyhow!("internal: lost picked file"))
}

/// GET `Range: bytes=start-end` (inclusive). Returns body bytes.
pub fn http_range(url: &str, start: u64, end: u64) -> Result<Vec<u8>> {
    let resp = apply_auth(ureq::get(url))
        .set("Range", &format!("bytes={start}-{end}"))
        .call()
        .map_err(|e| anyhow!("range GET {url}: {e}"))?;
    let mut buf = Vec::new();
    resp.into_reader()
        .read_to_end(&mut buf)
        .context("read range body")?;
    Ok(buf)
}

/// Resumable download into `dest`. Uses `.partial` + `Range` and verifies sha256 when known.
pub fn download_resumable(remote: &RemoteFile, dest: &Path) -> Result<PathBuf> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    if dest.exists() {
        if let Some(expected) = remote.size {
            if dest.metadata()?.len() == expected {
                if let Some(want) = remote.sha256.as_deref() {
                    let got = sha256_file(dest)?;
                    if hashes_equal(&got, want) {
                        eprintln!("already present {}", dest.display());
                        return Ok(dest.to_path_buf());
                    }
                } else {
                    eprintln!("already present {}", dest.display());
                    return Ok(dest.to_path_buf());
                }
            }
        }
    }

    let partial = dest.with_extension("gguf.partial");
    let mut start = if partial.exists() {
        partial.metadata()?.len()
    } else {
        0
    };
    if let Some(total) = remote.size {
        if start > total {
            let _ = fs::remove_file(&partial);
            start = 0;
        }
        if start == total {
            verify_and_promote(&partial, dest, remote.sha256.as_deref())?;
            return Ok(dest.to_path_buf());
        }
    }

    let mut req = apply_auth(ureq::get(&remote.url));
    if start > 0 {
        req = req.set("Range", &format!("bytes={start}-"));
        eprintln!("resuming {} at byte {start}", remote.path);
    } else {
        eprintln!("downloading {}", remote.path);
    }

    let resp = req.call().map_err(|e| anyhow!("download {}: {e}", remote.url))?;
    let status = resp.status();
    if start > 0 && status == 200 {
        // Server ignored Range — restart.
        let _ = fs::remove_file(&partial);
        start = 0;
    }

    let mut file = if start > 0 {
        OpenOptions::new().append(true).open(&partial)?
    } else {
        if let Some(parent) = partial.parent() {
            fs::create_dir_all(parent)?;
        }
        File::create(&partial)?
    };

    let mut reader = resp.into_reader();
    let mut buf = [0u8; 64 * 1024];
    let mut written = start;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        written += n as u64;
        if let Some(total) = remote.size {
            if total > 0 && written % (8 * 1024 * 1024) < 64 * 1024 {
                let pct = (written as f64 / total as f64) * 100.0;
                eprint!("\r{:.1}%  {} / {}", pct, written, total);
            }
        }
    }
    file.flush()?;
    eprintln!();

    verify_and_promote(&partial, dest, remote.sha256.as_deref())?;
    Ok(dest.to_path_buf())
}

fn verify_and_promote(partial: &Path, dest: &Path, want: Option<&str>) -> Result<()> {
    if let Some(want) = want {
        let got = sha256_file(partial)?;
        if !hashes_equal(&got, want) {
            bail!(
                "sha256 mismatch for {}: got {got}, expected {want}",
                dest.display()
            );
        }
    }
    if dest.exists() {
        fs::remove_file(dest)?;
    }
    fs::rename(partial, dest).with_context(|| format!("rename {}", dest.display()))?;
    Ok(())
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn hashes_equal(got: &str, want: &str) -> bool {
    let want = want
        .strip_prefix("sha256:")
        .unwrap_or(want)
        .trim()
        .to_ascii_lowercase();
    got.eq_ignore_ascii_case(&want)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_repo_quant() {
        let empty = AliasMap::new();
        let s = parse_spec("Qwen/Qwen3-0.6B-GGUF:Q4_K_M", &empty);
        assert_eq!(s.repo, "Qwen/Qwen3-0.6B-GGUF");
        assert_eq!(s.quant.as_deref(), Some("Q4_K_M"));
    }

    #[test]
    fn parse_alias() {
        let mut aliases = AliasMap::new();
        aliases.insert(
            "qwen3:0.6b".into(),
            "Qwen/Qwen3-0.6B-GGUF:Q4_K_M".into(),
        );
        let s = parse_spec("qwen3:0.6b", &aliases);
        assert_eq!(s.repo, "Qwen/Qwen3-0.6B-GGUF");
        assert_eq!(s.quant.as_deref(), Some("Q4_K_M"));
    }

    #[test]
    fn parse_path_not_quant() {
        let empty = AliasMap::new();
        let s = parse_spec("/tmp/model.gguf", &empty);
        assert_eq!(s.repo, "/tmp/model.gguf");
        assert!(s.quant.is_none());
    }

    #[test]
    fn match_quant_prefers_named_gguf() {
        let paths = vec![
            "README.md".into(),
            "Qwen3-0.6B-Q8_0.gguf".into(),
            "Qwen3-0.6B-Q4_K_M.gguf".into(),
            "gguf/extra-Q4_K_M-imat.gguf".into(),
        ];
        assert_eq!(
            match_quant_file(&paths, Some("Q4_K_M")),
            Some("Qwen3-0.6B-Q4_K_M.gguf")
        );
        assert!(match_quant_file(&paths, Some("PTQ1_0")).is_none());
    }

    #[test]
    fn dest_follows_runtime_layout() {
        let spec = ModelSpec {
            repo: "Qwen/Qwen3-0.6B-GGUF".into(),
            quant: Some("Q4_K_M".into()),
        };
        let root = PathBuf::from("models");
        let p = dest_path(&root, &spec, "Qwen3-0.6B-Q4_K_M.gguf");
        assert_eq!(
            p,
            root.join("Qwen--Qwen3-0.6B-GGUF")
                .join("resolve")
                .join("main")
                .join("Qwen3-0.6B-Q4_K_M.gguf")
        );
    }
}
