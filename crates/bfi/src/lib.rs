//! Builderforce native inference CLI (`bfi`).
//!
//! Subcommands: `pull`, `run`, `serve`, `ps`, `fit`, `rm`, `ls`, `bench`,
//! `import-ollama`. See PRD §5.6–5.7.

pub mod cli;
pub mod commands;
pub mod fit;
pub mod hf;
pub mod ollama;
pub mod store;

pub use cli::{Cli, Commands};

use anyhow::Result;
use clap::Parser;

/// Parse argv and run the selected subcommand.
pub fn run() -> Result<()> {
    let cli = Cli::parse();
    commands::dispatch(cli.cmd)
}
