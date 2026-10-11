//! Model store: list installed models, resolve paths, model aliases.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use serde::Deserialize;

/// The model store directory: `~/.builderforce/models/`.
pub fn models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("BFI_MODELS_DIR") {
        return PathBuf::from(dir);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".builderforce")
        .join("models")
}

/// Path to a model by name or alias.
/// First checks if it's an existing path, then looks in models_dir.
pub fn model_path(name: &str) -> PathBuf {
    let p = PathBuf::from(name);
    if p.exists() {
        return p;
    }

    // Try exact name in models_dir
    let in_store = models_dir().join(name);
    if in_store.exists() {
        return in_store;
    }

    // Try resolving alias
    if let Ok(aliases) = aliases() {
        if let Some(resolved) = aliases.get(name) {
            let resolved_path = models_dir().join(resolved);
            if resolved_path.exists() {
                return resolved_path;
            }
        }
    }

    // Fall back: assume it's a HuggingFace repo
    // Format: "org/model:quant" -> "org--model/resolve/main/gguf/quant.gguf"
    if name.contains('/') && !name.contains("--") {
        let parts: Vec<&str> = name.split(':').collect();
        let repo = parts[0];
        let quant = parts.get(1).unwrap_or(&"gguf");
        let model_dir = repo.replace('/', "--");
        return models_dir()
            .join(&model_dir)
            .join("resolve")
            .join("main")
            .join(format!("{}.gguf", quant));
    }

    in_store
}

/// An installed model entry.
#[derive(Debug, Clone, Deserialize)]
pub struct InstalledModel {
    /// Model name (file stem or alias).
    pub name: String,
    /// Full path to the GGUF file.
    pub path: PathBuf,
    /// File size in bytes.
    pub size: u64,
}

/// List all installed models in the store.
/// Scans for .gguf files in models_dir.
pub fn list_models() -> Result<Vec<InstalledModel>> {
    let dir = models_dir();
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut models = Vec::new();

    // Walk the models directory for .gguf files
    for entry in walkdir(&dir)? {
        if let Some(ext) = entry.path().extension() {
            if ext == "gguf" {
                let name = entry
                    .path()
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);

                models.push(InstalledModel {
                    name,
                    path: entry.path().to_path_buf(),
                    size,
                });
            }
        }
    }

    // Sort by name
    models.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(models)
}

fn walkdir(dir: &Path) -> Result<Vec<DirEntry>> {
    let mut entries = Vec::new();
    walkdir_rec(dir, &mut entries)?;
    Ok(entries)
}

fn walkdir_rec(dir: &Path, entries: &mut Vec<DirEntry>) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walkdir_rec(&path, entries)?;
        } else {
            entries.push(DirEntry { path, metadata: entry.metadata()? });
        }
    }
    Ok(())
}

struct DirEntry {
    path: PathBuf,
    metadata: std::fs::Metadata,
}

/// Alias mapping: user alias -> HF path.
pub type AliasMap = std::collections::HashMap<String, String>;

/// Load aliases.toml from the model store.
/// Returns an empty map if the file doesn't exist.
pub fn aliases() -> Result<AliasMap> {
    let path = models_dir().join("aliases.toml");
    if !path.exists() {
        return Ok(AliasMap::new());
    }

    let content = std::fs::read_to_string(&path)?;
    let mut map = AliasMap::new();

    // Simple TOML parser for aliases section
    // Format:
    // [aliases]
    // bonsai2:27b = "prism-ml/Ternary-Bonsai-2-27B-gguf:PTQ1_0"
    // qwen3:1.7b = "Qwen/Qwen3-1.7B-GGUF:Q4_K_M"

    let mut in_aliases = false;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[aliases]" {
            in_aliases = true;
            continue;
        }
        if line.starts_with('[') && in_aliases {
            // Entered a new section
            break;
        }
        if in_aliases {
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim().trim_matches('"').to_string();
                let value = value.trim().trim_matches('"').to_string();
                if !key.is_empty() && !value.is_empty() {
                    map.insert(key, value);
                }
            }
        }
    }

    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_dir_default() {
        let dir = models_dir();
        assert!(dir.to_string_lossy().ends_with(".builderforce/models"));
    }

    #[test]
    fn model_path_absolute() {
        // An absolute path should return itself if it exists
        let path = model_path("/tmp/test.gguf");
        // Won't exist in test, so returns fallback
        assert!(path.to_string_lossy().contains("test.gguf"));
    }

    #[test]
    fn aliases_empty_when_missing() {
        let aliases = aliases().unwrap();
        assert!(aliases.is_empty());
    }
}
