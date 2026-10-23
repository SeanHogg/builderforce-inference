//! The LLMLingua-2 model files: where they come from, where they are cached, and the
//! checks a download must pass before it is used.
//!
//! Default: `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` as a published
//! ONNX export (`ldenoue/llmlingua-2-bert-base-multilingual-cased-meetingbank`, dynamic
//! int8, 178,566,832 bytes), pinned to one revision with known SHA-256s. Override with
//! `BF_COMPRESS_MODEL_URL` / `BF_COMPRESS_TOKENIZER_URL` (and optionally
//! `BF_COMPRESS_MODEL_SHA256` / `BF_COMPRESS_TOKENIZER_SHA256`) to serve your own export:
//!
//! ```sh
//! optimum-cli export onnx --task token-classification \
//!   --model microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank llmlingua2-onnx/
//! optimum-cli onnxruntime quantize --avx2 --onnx_model llmlingua2-onnx/ -o llmlingua2-int8/
//! ```
//!
//! Files land under the same cache dir fastembed uses for the embedding model, in the
//! hf-hub layout (`models--<org>--<name>/snapshots/<revision>/<file>`).

use anyhow::{anyhow, Result};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

pub const DEFAULT_REPO: &str = "ldenoue/llmlingua-2-bert-base-multilingual-cased-meetingbank";
pub const DEFAULT_REVISION: &str = "d7070a9b381ca5297dd7bd5cf7704dde81470518";
/// The source model the export was made from (what the scores mean).
pub const BASE_MODEL: &str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank";

const MODEL_FILE: &str = "onnx/model_quantized.onnx";
const MODEL_SHA256: &str = "6745c7aff5cb5c2c3c96b8e72ab8a65ce42157058f9221f6c340f0c37e760f5d";
const MODEL_BYTES: u64 = 178_566_832;
const TOKENIZER_FILE: &str = "tokenizer.json";
const TOKENIZER_SHA256: &str = "bf1b59b7b11c95f194f51708d918eea378e09d05f84c0e1656dc5180e8117088";
const TOKENIZER_BYTES: u64 = 2_919_362;

/// One file of the model and how to trust it.
#[derive(Debug, Clone, PartialEq)]
pub struct RemoteFile {
    /// Path inside the snapshot dir.
    pub rel_path: String,
    pub url: String,
    pub sha256: Option<String>,
    pub bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    pub repo: String,
    pub revision: String,
    pub model: RemoteFile,
    pub tokenizer: RemoteFile,
}

/// Local paths of a ready model.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelFiles {
    pub onnx: PathBuf,
    pub tokenizer: PathBuf,
}

fn hf_url(repo: &str, revision: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/{revision}/{file}")
}

/// fastembed's convention: `FASTEMBED_CACHE_DIR`, else `.fastembed_cache`. Our own
/// `BF_COMPRESS_CACHE_DIR` wins over both. bf-context passes its model cache explicitly.
pub fn default_cache_dir() -> PathBuf {
    std::env::var_os("BF_COMPRESS_CACHE_DIR")
        .or_else(|| std::env::var_os("FASTEMBED_CACHE_DIR"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".fastembed_cache"))
}

impl Default for ModelSpec {
    fn default() -> Self {
        let file = |rel: &str, sha: &str, bytes: u64| RemoteFile {
            rel_path: rel.into(),
            url: hf_url(DEFAULT_REPO, DEFAULT_REVISION, rel),
            sha256: Some(sha.into()),
            bytes: Some(bytes),
        };
        Self {
            repo: DEFAULT_REPO.into(),
            revision: DEFAULT_REVISION.into(),
            model: file(MODEL_FILE, MODEL_SHA256, MODEL_BYTES),
            tokenizer: file(TOKENIZER_FILE, TOKENIZER_SHA256, TOKENIZER_BYTES),
        }
    }
}

impl ModelSpec {
    /// The default spec with any `BF_COMPRESS_*` overrides applied.
    pub fn from_env() -> Self {
        Self::with_overrides(|k| std::env::var(k).ok().filter(|v| !v.trim().is_empty()))
    }

    /// `get` reads one override. An overridden URL drops the pinned checksum and size
    /// (they describe the default file) unless its own `*_SHA256` is given.
    pub fn with_overrides(get: impl Fn(&str) -> Option<String>) -> Self {
        let mut spec = Self::default();
        for (file, url_key, sha_key) in [
            (&mut spec.model, "BF_COMPRESS_MODEL_URL", "BF_COMPRESS_MODEL_SHA256"),
            (&mut spec.tokenizer, "BF_COMPRESS_TOKENIZER_URL", "BF_COMPRESS_TOKENIZER_SHA256"),
        ] {
            if let Some(url) = get(url_key) {
                file.url = url;
                file.sha256 = None;
                file.bytes = None;
            }
            if let Some(sha) = get(sha_key) {
                file.sha256 = Some(sha.trim().to_lowercase());
            }
        }
        let custom = get("BF_COMPRESS_MODEL_URL").is_some() || get("BF_COMPRESS_TOKENIZER_URL").is_some();
        if custom {
            spec.repo = "custom/llmlingua-2".into();
            spec.revision = override_key(&spec.model.url, &spec.tokenizer.url);
        }
        spec
    }

    pub fn snapshot_dir(&self, cache_dir: &Path) -> PathBuf {
        cache_dir.join(format!("models--{}", self.repo.replace('/', "--"))).join("snapshots").join(&self.revision)
    }

    pub fn files(&self, cache_dir: &Path) -> ModelFiles {
        let dir = self.snapshot_dir(cache_dir);
        ModelFiles { onnx: dir.join(&self.model.rel_path), tokenizer: dir.join(&self.tokenizer.rel_path) }
    }

    /// The model files, downloading and verifying any that are missing.
    #[cfg(feature = "compress")]
    pub fn ensure(&self, cache_dir: &Path) -> Result<ModelFiles> {
        use anyhow::Context as _;
        let files = self.files(cache_dir);
        for (remote, local) in [(&self.tokenizer, &files.tokenizer), (&self.model, &files.onnx)] {
            if !local.is_file() {
                download(remote, local).with_context(|| format!("downloading {}", remote.url))?;
            }
        }
        Ok(files)
    }
}

/// A stable short dir name for a custom URL pair, so two overrides never share a cache.
fn override_key(a: &str, b: &str) -> String {
    let mut h = Sha256::new();
    h.update(a.as_bytes());
    h.update([0]);
    h.update(b.as_bytes());
    hex::encode(&h.finalize()[..8])
}

/// Stream `reader` into `dest` via a `.part` file, checking size and SHA-256 before the
/// final rename — a truncated or tampered download is never left where `ensure` trusts it.
pub fn write_verified(mut reader: impl Read, dest: &Path, expect: &RemoteFile) -> Result<()> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let part = dest.with_extension("part");
    let result = (|| -> Result<()> {
        let mut file = std::fs::File::create(&part)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 16];
        let mut total = 0u64;
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            std::io::Write::write_all(&mut file, &buf[..n])?;
            total += n as u64;
        }
        std::io::Write::flush(&mut file)?;
        if let Some(bytes) = expect.bytes {
            if total != bytes {
                return Err(anyhow!("{}: got {total} bytes, expected {bytes}", expect.rel_path));
            }
        }
        if let Some(sha) = &expect.sha256 {
            let got = hex::encode(hasher.finalize());
            if &got != sha {
                return Err(anyhow!("{}: sha256 {got} does not match {sha}", expect.rel_path));
            }
        }
        Ok(())
    })();
    match result {
        Ok(()) => Ok(std::fs::rename(&part, dest)?),
        Err(e) => {
            let _ = std::fs::remove_file(&part);
            Err(e)
        }
    }
}

#[cfg(feature = "compress")]
fn download(remote: &RemoteFile, dest: &Path) -> Result<()> {
    let res = ureq::get(&remote.url).call().map_err(|e| anyhow!("{e}"))?;
    write_verified(res.into_reader(), dest, remote)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_spec_is_pinned_and_cached_in_hf_layout() {
        let spec = ModelSpec::default();
        assert!(spec.model.url.contains(DEFAULT_REVISION));
        assert!(spec.model.url.ends_with("/onnx/model_quantized.onnx"));
        let files = spec.files(Path::new("/cache"));
        assert!(files.onnx.ends_with(
            Path::new("models--ldenoue--llmlingua-2-bert-base-multilingual-cased-meetingbank/snapshots")
                .join(DEFAULT_REVISION)
                .join("onnx/model_quantized.onnx")
        ));
    }

    #[test]
    fn url_override_drops_pinned_checksum_unless_given() {
        let spec = ModelSpec::with_overrides(|k| match k {
            "BF_COMPRESS_MODEL_URL" => Some("https://mirror.example/m.onnx".into()),
            "BF_COMPRESS_MODEL_SHA256" => Some("ABCD".into()),
            _ => None,
        });
        assert_eq!(spec.model.url, "https://mirror.example/m.onnx");
        assert_eq!(spec.model.sha256.as_deref(), Some("abcd"));
        assert_eq!(spec.model.bytes, None);
        assert_eq!(spec.tokenizer.sha256.as_deref(), Some(TOKENIZER_SHA256));
        assert_ne!(spec.revision, DEFAULT_REVISION);
    }

    #[test]
    fn verified_write_accepts_match_and_rejects_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("a/b.bin");
        let body = b"hello";
        let good = RemoteFile {
            rel_path: "b.bin".into(),
            url: String::new(),
            sha256: Some(hex::encode(Sha256::digest(body))),
            bytes: Some(5),
        };
        write_verified(&body[..], &dest, &good).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), body);

        let dest2 = dir.path().join("c.bin");
        let bad = RemoteFile { sha256: Some("00".into()), ..good.clone() };
        assert!(write_verified(&body[..], &dest2, &bad).is_err());
        assert!(!dest2.exists() && !dest2.with_extension("part").exists());
        let short = RemoteFile { bytes: Some(6), sha256: None, ..good };
        assert!(write_verified(&body[..], &dest2, &short).is_err());
    }
}
