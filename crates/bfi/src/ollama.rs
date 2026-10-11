//! `bfi import-ollama` — link Ollama blob-store GGUFs into our store.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

const MODEL_LAYER: &str = "application/vnd.ollama.image.model";

#[derive(Debug, Deserialize)]
struct Manifest {
    #[serde(default)]
    layers: Vec<Layer>,
}

#[derive(Debug, Deserialize)]
struct Layer {
    #[serde(default, rename = "mediaType")]
    media_type: String,
    #[serde(default)]
    digest: String,
}

/// Default Ollama models dir: `$OLLAMA_MODELS` or `~/.ollama/models`.
pub fn ollama_models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("OLLAMA_MODELS") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    home_dir().join(".ollama").join("models")
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Link every GGUF blob referenced by Ollama manifests into `models_dir/ollama/`.
/// Returns the number of models linked (existing links count as success, not an extra).
pub fn import_ollama(ollama_dir: &Path, models_dir: &Path) -> Result<usize> {
    let manifests = ollama_dir.join("manifests");
    let blobs = ollama_dir.join("blobs");
    if !manifests.exists() {
        eprintln!("no Ollama manifests at {}", manifests.display());
        return Ok(0);
    }
    let dest_root = models_dir.join("ollama");
    fs::create_dir_all(&dest_root)?;

    let mut linked = 0usize;
    walk_manifests(&manifests, &manifests, &blobs, &dest_root, &mut linked)?;
    Ok(linked)
}

fn walk_manifests(
    root: &Path,
    dir: &Path,
    blobs: &Path,
    dest_root: &Path,
    linked: &mut usize,
) -> Result<()> {
    for ent in fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let ent = ent?;
        let path = ent.path();
        if path.is_dir() {
            walk_manifests(root, &path, blobs, dest_root, linked)?;
            continue;
        }
        if let Ok(did) = import_one(root, &path, blobs, dest_root) {
            if did {
                *linked += 1;
            }
        }
    }
    Ok(())
}

fn import_one(root: &Path, manifest_path: &Path, blobs: &Path, dest_root: &Path) -> Result<bool> {
    let text = fs::read_to_string(manifest_path)?;
    let man: Manifest = serde_json::from_str(&text)
        .with_context(|| format!("manifest {}", manifest_path.display()))?;
    let layer = man
        .layers
        .iter()
        .find(|l| l.media_type == MODEL_LAYER)
        .or_else(|| man.layers.last());
    let Some(layer) = layer else {
        return Ok(false);
    };
    if layer.digest.is_empty() {
        return Ok(false);
    }
    let blob = blob_path(blobs, &layer.digest);
    if !blob.exists() {
        eprintln!("missing blob {} for {}", blob.display(), manifest_path.display());
        return Ok(false);
    }
    let name = model_name_from_manifest(root, manifest_path);
    let dest = dest_root.join(format!("{name}.gguf"));
    if dest.exists() {
        return Ok(true);
    }
    link_file(&blob, &dest)?;
    eprintln!("linked {name} -> {}", dest.display());
    Ok(true)
}

fn blob_path(blobs: &Path, digest: &str) -> PathBuf {
    let d = digest.trim();
    let file = d.replace(':', "-");
    blobs.join(file)
}

/// `manifests/registry.ollama.ai/library/llama3.2/latest` → `llama3.2-latest`.
fn model_name_from_manifest(root: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    let parts: Vec<String> = rel
        .iter()
        .filter_map(|s| s.to_str().map(|s| s.to_string()))
        .collect();
    if parts.len() >= 2 {
        let tag = parts.last().unwrap();
        let name = parts.get(parts.len() - 2).unwrap();
        format!("{name}-{tag}")
    } else if let Some(stem) = path.file_name().and_then(|s| s.to_str()) {
        stem.to_string()
    } else {
        "model".into()
    }
}

fn link_file(from: &Path, to: &Path) -> Result<()> {
    if let Some(parent) = to.parent() {
        fs::create_dir_all(parent)?;
    }
    if to.exists() {
        return Ok(());
    }
    if fs::hard_link(from, to).is_ok() {
        return Ok(());
    }
    #[cfg(unix)]
    std::os::unix::fs::symlink(from, to)
        .with_context(|| format!("symlink {} -> {}", from.display(), to.display()))?;
    #[cfg(windows)]
    std::os::windows::fs::symlink_file(from, to)
        .with_context(|| format!("symlink {} -> {}", from.display(), to.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_empty_ollama_dir() {
        let ollama = tempfile::tempdir().unwrap();
        let store = tempfile::tempdir().unwrap();
        let n = import_ollama(ollama.path(), store.path()).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn import_links_model_layer() {
        let ollama = tempfile::tempdir().unwrap();
        let store = tempfile::tempdir().unwrap();
        let man_dir = ollama
            .path()
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library")
            .join("llama3.2");
        fs::create_dir_all(&man_dir).unwrap();
        let blobs = ollama.path().join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        let digest = "sha256:abc123def";
        let blob = blobs.join("sha256-abc123def");
        fs::write(&blob, b"GGUF-FAKE").unwrap();
        let manifest = serde_json::json!({
            "schemaVersion": 2,
            "layers": [
                { "mediaType": "application/vnd.ollama.image.params", "digest": "sha256:skip" },
                { "mediaType": "application/vnd.ollama.image.model", "digest": digest }
            ]
        });
        fs::write(man_dir.join("latest"), serde_json::to_vec(&manifest).unwrap()).unwrap();

        let n = import_ollama(ollama.path(), store.path()).unwrap();
        assert_eq!(n, 1);
        let dest = store.path().join("ollama").join("llama3.2-latest.gguf");
        assert!(dest.exists());
        assert_eq!(fs::read(&dest).unwrap(), b"GGUF-FAKE");

        let n2 = import_ollama(ollama.path(), store.path()).unwrap();
        assert_eq!(n2, 1);
    }

    #[test]
    fn blob_path_colon_to_dash() {
        let root = PathBuf::from("blobs");
        let p = blob_path(&root, "sha256:deadbeef");
        assert_eq!(p, root.join("sha256-deadbeef"));
    }
}
