//! Local store under `~/.builderforce/models/` (or `BFI_MODELS_DIR`).

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreEntry {
    pub name: String,
    pub path: PathBuf,
    pub size: u64,
}

/// List `.gguf` files under `dir`. Missing dir → empty list (not an error).
pub fn list_store(dir: &Path) -> Result<Vec<StoreEntry>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    walk_gguf(dir, &mut out)?;
    out.sort_by(|a, b| a.name.cmp(&b.name).then(a.path.cmp(&b.path)));
    Ok(out)
}

fn walk_gguf(dir: &Path, out: &mut Vec<StoreEntry>) -> Result<()> {
    let entries = fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))?;
    for ent in entries {
        let ent = ent?;
        let path = ent.path();
        if path.is_dir() {
            walk_gguf(&path, out)?;
            continue;
        }
        if path.extension().and_then(|s| s.to_str()) != Some("gguf") {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let size = ent.metadata().map(|m| m.len()).unwrap_or(0);
        out.push(StoreEntry { name, path, size });
    }
    Ok(())
}

/// Delete a model by file stem, file name, or path fragment.
pub fn remove_from_store(dir: &Path, name: &str) -> Result<PathBuf> {
    let entries = list_store(dir)?;
    let hit = entries.iter().find(|e| {
        e.name == name
            || e.path.file_name().and_then(|s| s.to_str()) == Some(name)
            || e.path.to_string_lossy() == name
            || e.path
                .file_stem()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case(name))
    });
    let Some(hit) = hit else {
        bail!("model not found in store: {name}");
    };
    let path = hit.path.clone();
    fs::remove_file(&path).with_context(|| format!("remove {}", path.display()))?;
    Ok(path)
}

pub fn format_bytes(n: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let x = n as f64;
    if x >= GIB {
        format!("{:.2} GiB", x / GIB)
    } else if x >= MIB {
        format!("{:.2} MiB", x / MIB)
    } else if x >= KIB {
        format!("{:.2} KiB", x / KIB)
    } else {
        format!("{n} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ls_on_empty_store() {
        let dir = tempfile::tempdir().unwrap();
        let models = list_store(dir.path()).unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn ls_on_missing_dir_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nope");
        let models = list_store(&missing).unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn ls_finds_gguf_and_rm_deletes() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("org--model").join("resolve").join("main");
        fs::create_dir_all(&nested).unwrap();
        let gguf = nested.join("Q4_K_M.gguf");
        fs::write(&gguf, b"GGUF").unwrap();
        fs::write(nested.join("notes.txt"), b"x").unwrap();

        let models = list_store(dir.path()).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "Q4_K_M");
        assert_eq!(models[0].size, 4);

        let removed = remove_from_store(dir.path(), "Q4_K_M").unwrap();
        assert_eq!(removed, gguf);
        assert!(!gguf.exists());
        assert!(list_store(dir.path()).unwrap().is_empty());
    }

    #[test]
    fn rm_missing_errors() {
        let dir = tempfile::tempdir().unwrap();
        let err = remove_from_store(dir.path(), "ghost").unwrap_err();
        assert!(err.to_string().contains("not found"));
    }
}
