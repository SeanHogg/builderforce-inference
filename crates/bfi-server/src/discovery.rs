//! `~/.builderforce/bfi.json` — `{port, pid, version, token}` with user-only perms.
//! Same scheme as `bf-context/src/discovery.rs` (`desktop.json`).

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Chosen listen address, written after bind so clients never assume 11434.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Discovery {
    pub port: u16,
    pub pid: u32,
    pub version: String,
    pub token: String,
}

/// Override with `BFI_DISCOVERY`; otherwise `~/.builderforce/bfi.json`.
pub fn discovery_path() -> PathBuf {
    if let Ok(p) = std::env::var("BFI_DISCOVERY") {
        let p = p.trim();
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".builderforce")
        .join("bfi.json")
}

pub fn publish_discovery(port: u16, token: &str) -> Result<Discovery> {
    publish_discovery_to(&discovery_path(), port, token)
}

pub fn publish_discovery_to(path: &Path, port: u16, token: &str) -> Result<Discovery> {
    let d = Discovery {
        port,
        pid: std::process::id(),
        version: env!("CARGO_PKG_VERSION").into(),
        token: token.to_string(),
    };
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    write_private(path, &serde_json::to_vec_pretty(&d)?)?;
    Ok(d)
}

/// Remove the file only if it is still ours (a newer instance may have replaced it).
pub fn withdraw_discovery(d: &Discovery) {
    withdraw_at(&discovery_path(), d);
}

pub(crate) fn withdraw_at(path: &Path, d: &Discovery) {
    if read_at(path)
        .map(|cur| cur.pid == d.pid && cur.port == d.port)
        .unwrap_or(false)
    {
        let _ = std::fs::remove_file(path);
    }
}

pub fn read() -> Option<Discovery> {
    read_at(&discovery_path())
}

pub(crate) fn read_at(path: &Path) -> Option<Discovery> {
    let raw = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

/// Drop guard: withdraws the file when `serve` returns or is cancelled.
pub struct DiscoveryGuard {
    pub discovery: Discovery,
    path: PathBuf,
}

impl DiscoveryGuard {
    pub fn new(discovery: Discovery) -> Self {
        Self {
            discovery,
            path: discovery_path(),
        }
    }

    pub fn at(path: PathBuf, discovery: Discovery) -> Self {
        Self { discovery, path }
    }
}

impl Drop for DiscoveryGuard {
    fn drop(&mut self) {
        withdraw_at(&self.path, &self.discovery);
    }
}

#[cfg(unix)]
fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)?;
    f.write_all(bytes)?;
    Ok(())
}

#[cfg(not(unix))]
fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    // The file lives under the user's profile, whose ACL already excludes other users.
    std::fs::write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publish_roundtrip_and_withdraw() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bfi.json");
        let d = publish_discovery_to(&path, 11435, "tok").unwrap();
        assert_eq!(d.port, 11435);
        assert_eq!(d.token, "tok");
        assert_eq!(d.pid, std::process::id());
        assert!(!d.version.is_empty());
        let got = read_at(&path).unwrap();
        assert_eq!(got, d);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
        withdraw_at(&path, &d);
        assert!(!path.exists());
    }

    #[test]
    fn withdraw_leaves_a_newer_instance() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bfi.json");
        let old = Discovery {
            port: 1,
            pid: 1,
            version: "x".into(),
            token: "a".into(),
        };
        let _ = publish_discovery_to(&path, 2, "b").unwrap();
        withdraw_at(&path, &old);
        assert!(path.exists());
    }
}
