//! `~/.builderforce/desktop.json` `models` field — how the VSIX (M6) finds Synapse's local engine.
//!
//! Schema (additive; old readers ignore unknown fields). The context service already writes
//! `{port, token, pid, version}` for the loopback index. This module **merges** `models`
//! without wiping those fields:
//!
//! ```json
//! {
//!   "port": 17890,
//!   "token": "…",
//!   "pid": 123,
//!   "version": "…",
//!   "models": {
//!     "provider": "synapse",
//!     "endpoint": "http://127.0.0.1:<bfi-port>/v1",
//!     "port": 11434,
//!     "token": "…",
//!     "version": "…"
//!   }
//! }
//! ```
//!
//! `models.provider` is always `"synapse"`. Written when the runtime is healthy; the `models`
//! key is withdrawn when the supervisor stops.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

use crate::runtime::BfiJson;

/// The only provider string the desktop writes. M6 VSIX matches this exactly.
pub const PROVIDER: &str = "synapse";

/// OpenAI-compatible base the VSIX (and any other client) should call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelsDiscovery {
    pub provider: String,
    pub endpoint: String,
    pub port: u16,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub token: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub version: String,
}

/// `http://127.0.0.1:<port>/v1`
pub fn models_endpoint(port: u16) -> String {
    format!("http://127.0.0.1:{port}/v1")
}

/// OpenAI `/v1` URL for a runtime base such as `http://127.0.0.1:11434`.
pub fn endpoint_from_base(base: &str) -> String {
    let url = base.trim().trim_end_matches('/');
    if url.ends_with("/v1") {
        url.to_string()
    } else {
        format!("{url}/v1")
    }
}

pub fn from_bfi(d: &BfiJson) -> ModelsDiscovery {
    ModelsDiscovery {
        provider: PROVIDER.to_string(),
        endpoint: models_endpoint(d.port),
        port: d.port,
        token: d.token.clone(),
        version: d.version.clone(),
    }
}

/// Parse `http://host:port` (the runtime base) into a models discovery record.
pub fn from_base(base: &str, token: &str, version: &str) -> Option<ModelsDiscovery> {
    let url = base.trim().trim_end_matches('/');
    let after = url.split("://").nth(1)?;
    let port: u16 = after.rsplit(':').next()?.parse().ok()?;
    Some(ModelsDiscovery {
        provider: PROVIDER.to_string(),
        endpoint: endpoint_from_base(url),
        port,
        token: token.to_string(),
        version: version.to_string(),
    })
}

/// `BUILDERFORCE_DESKTOP_DISCOVERY` when set, else `~/.builderforce/desktop.json`.
pub fn desktop_json_path() -> PathBuf {
    if let Ok(p) = std::env::var("BUILDERFORCE_DESKTOP_DISCOVERY") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".builderforce")
        .join("desktop.json")
}

pub fn publish_default(models: &ModelsDiscovery) -> Result<()> {
    merge_write(&desktop_json_path(), models)
}

pub fn withdraw_default() -> Result<()> {
    withdraw_at(&desktop_json_path())
}

/// Merge `models` into an existing desktop.json without clobbering context fields.
pub fn merge_write(path: &Path, models: &ModelsDiscovery) -> Result<()> {
    let mut root = match std::fs::read_to_string(path) {
        Ok(t) => serde_json::from_str::<Value>(&t).unwrap_or_else(|_| json!({})),
        Err(_) => json!({}),
    };
    if !root.is_object() {
        root = json!({});
    }
    root["models"] = serde_json::to_value(models)?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(&root)?)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

/// Drop the `models` key; leave every other field alone. Missing file is a no-op.
pub fn withdraw_at(path: &Path) -> Result<()> {
    let Ok(t) = std::fs::read_to_string(path) else {
        return Ok(());
    };
    let mut root: Value = serde_json::from_str(&t)?;
    if let Some(obj) = root.as_object_mut() {
        if obj.remove("models").is_none() {
            return Ok(());
        }
    }
    std::fs::write(path, serde_json::to_string_pretty(&root)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_schema_has_synapse_provider_and_v1_endpoint() {
        let d = ModelsDiscovery {
            provider: PROVIDER.into(),
            endpoint: models_endpoint(11434),
            port: 11434,
            token: "tok".into(),
            version: "1".into(),
        };
        let v = serde_json::to_value(&d).unwrap();
        assert_eq!(v["provider"], "synapse");
        assert_eq!(v["endpoint"], "http://127.0.0.1:11434/v1");
        assert_eq!(v["port"], 11434);
        assert_eq!(PROVIDER, "synapse");
    }

    #[test]
    fn from_bfi_uses_synapse_and_loopback_v1() {
        let m = from_bfi(&BfiJson {
            port: 11434,
            pid: 7,
            version: "dev".into(),
            token: "m".into(),
        });
        assert_eq!(m.provider, "synapse");
        assert_eq!(m.endpoint, "http://127.0.0.1:11434/v1");
        assert_eq!(m.port, 11434);
        assert_eq!(m.token, "m");
    }

    #[test]
    fn from_base_parses_ollama_style_host() {
        let m = from_base("http://127.0.0.1:11434", "", "").unwrap();
        assert_eq!(m.provider, "synapse");
        assert_eq!(m.endpoint, "http://127.0.0.1:11434/v1");
        assert_eq!(m.port, 11434);
    }

    #[test]
    fn merge_write_keeps_context_discovery_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("desktop.json");
        std::fs::write(&path, r#"{"port":17890,"token":"ctx","pid":9,"version":"0.1"}"#).unwrap();
        let m = from_bfi(&BfiJson {
            port: 11434,
            pid: 1,
            version: "dev".into(),
            token: "m".into(),
        });
        merge_write(&path, &m).unwrap();
        let root: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(root["port"], 17890);
        assert_eq!(root["token"], "ctx");
        assert_eq!(root["pid"], 9);
        assert_eq!(root["models"]["provider"], "synapse");
        assert_eq!(root["models"]["endpoint"], "http://127.0.0.1:11434/v1");
        withdraw_at(&path).unwrap();
        let root: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert!(root.get("models").is_none());
        assert_eq!(root["port"], 17890);
    }
}
