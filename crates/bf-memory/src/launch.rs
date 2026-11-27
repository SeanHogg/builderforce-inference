//! How Synapse starts the memory server — the same server every other MCP host on the
//! machine runs, so they all share one store.
//!
//! Resolution, first match wins:
//! 1. `SYNAPSE_MEMORY_MCP` — a path to the server's `stdio.js` (run with `node`) or to an
//!    executable. For development against any checkout.
//! 2. A sibling checkout: `builderforce-memory/packages/memory-mcp/dist/bin/stdio.js` beside
//!    any folder above the Synapse executable (the monorepo layout).
//! 3. The published package through `npx`, with the same arguments the memory installer
//!    writes into every host (`buildServerSpec` in memory-mcp's `install/server-spec.ts`).

use crate::LaunchSpec;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub const LAUNCH_ENV: &str = "SYNAPSE_MEMORY_MCP";

const CHECKOUT_BIN: [&str; 6] = ["builderforce-memory", "packages", "memory-mcp", "dist", "bin", "stdio.js"];

#[derive(Deserialize)]
struct PublishedLaunch {
    posix: LaunchSpec,
    windows: LaunchSpec,
}

fn npx(windows: bool) -> LaunchSpec {
    // The Memory installer owns these commands. Regenerate the snapshot with
    // `node crates/bf-memory/sync-launch.mjs` from a built Memory checkout.
    static SPEC: OnceLock<PublishedLaunch> = OnceLock::new();
    let spec = SPEC.get_or_init(|| {
        serde_json::from_str(include_str!("generated-memory-launch.json"))
            .expect("valid generated Memory launch spec")
    });
    if windows { spec.windows.clone() } else { spec.posix.clone() }
}

/// The launch spec in effect for this process.
pub fn resolve() -> LaunchSpec {
    let exe = std::env::current_exe().ok();
    resolve_from(std::env::var(LAUNCH_ENV).ok().as_deref(), exe.as_deref(), cfg!(windows))
}

pub fn resolve_from(env_override: Option<&str>, exe: Option<&Path>, windows: bool) -> LaunchSpec {
    if let Some(v) = env_override.map(str::trim).filter(|v| !v.is_empty()) {
        let p = Path::new(v);
        return if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("js")) {
            LaunchSpec::node(p)
        } else {
            LaunchSpec { command: v.to_string(), args: vec![] }
        };
    }
    if let Some(script) = exe.and_then(checkout_bin) {
        return LaunchSpec::node(&script);
    }
    npx(windows)
}

fn checkout_bin(exe: &Path) -> Option<PathBuf> {
    exe.ancestors().skip(1).map(|dir| CHECKOUT_BIN.iter().fold(dir.to_path_buf(), |p, s| p.join(s))).find(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_runs_a_script_with_node_or_an_executable_as_is() {
        assert_eq!(resolve_from(Some("C:/m/stdio.js"), None, true), LaunchSpec { command: "node".into(), args: vec!["C:/m/stdio.js".into()] });
        assert_eq!(resolve_from(Some("/opt/bfmem"), None, false), LaunchSpec { command: "/opt/bfmem".into(), args: vec![] });
    }

    #[test]
    fn falls_back_to_the_installers_npx_spec() {
        let spec = resolve_from(None, None, true);
        assert_eq!(spec.command, "cmd");
        assert_eq!(&spec.args[..3], ["/c", "npx", "-y"]);
        assert_eq!(spec.args.last().unwrap(), "builderforce-memory-mcp");
        assert!(spec.args.iter().any(|a| a == "fake-indexeddb"));
        let posix = resolve_from(None, None, false);
        assert_eq!(posix.command, "npx");
        assert_eq!(&spec.args[2..], posix.args);
    }

    #[test]
    fn finds_a_sibling_checkout_above_the_executable() {
        let root = std::env::temp_dir().join(format!("bfmem-launch-{}", std::process::id()));
        let script = CHECKOUT_BIN.iter().fold(root.clone(), |p, s| p.join(s));
        std::fs::create_dir_all(script.parent().unwrap()).unwrap();
        std::fs::write(&script, "").unwrap();
        let exe = root.join("Builderforce.ai").join("desktop").join("target").join("release").join("synapse.exe");
        assert_eq!(resolve_from(None, Some(&exe), true), LaunchSpec::node(&script));
        let _ = std::fs::remove_dir_all(&root);
    }
}
