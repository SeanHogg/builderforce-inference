//! Command and arguments for a local Evermind process.

use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct LaunchSpec {
    pub command: String,
    pub args: Vec<String>,
}

impl LaunchSpec {
    /// A script run with `node`.
    pub fn node(script: &Path) -> Self {
        Self { command: "node".into(), args: vec![script.to_string_lossy().into_owned()] }
    }
}
