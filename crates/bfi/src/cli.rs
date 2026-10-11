//! clap subcommands for `bfi`. Every command has `--help`.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// Builderforce native inference CLI.
#[derive(Parser, Debug)]
#[command(name = "bfi", version, about = "Builderforce native inference CLI")]
pub struct Cli {
    #[command(subcommand)]
    pub cmd: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Resumable ranged download from Hugging Face with sha256 verification.
    Pull {
        /// `repo[:quant]` or a store alias (see aliases.toml).
        spec: String,
    },
    /// Interactive chat in the terminal. One-shot when `[prompt]` is given.
    Run {
        /// Installed model name, path, or alias.
        model: String,
        /// Optional one-shot prompt (omit for a REPL).
        prompt: Option<String>,
    },
    /// Start the HTTP server (OpenAI / Anthropic / Ollama dialects).
    Serve {
        /// Bind port. Default 11434 (Ollama-compatible).
        #[arg(long, default_value_t = 11434)]
        port: u16,
        /// Require a bearer key on `/v1/*`. Uses `BFI_API_KEY`, or generates one.
        #[arg(long)]
        auth: bool,
        /// Model store directory (also set as `BFI_MODELS_DIR`).
        #[arg(long)]
        models_dir: Option<PathBuf>,
    },
    /// Loaded models, device, VRAM/RAM, tok/s of the last request.
    Ps {
        /// Server port to query. Default 11434.
        #[arg(long, default_value_t = 11434)]
        port: u16,
    },
    /// Memory need vs this machine. Header-only HTTP range before download.
    ///
    /// Full KV/state/activation formula is ticket 2927; this command emits the
    /// PRD JSON shape with a stub breakdown (weights exact, other terms 0/overhead).
    Fit {
        /// Local path, `repo[:quant]`, or alias.
        model: String,
        /// Context length used for the report.
        #[arg(long, default_value_t = 4096)]
        ctx: usize,
        /// Print the PRD JSON object instead of a table.
        #[arg(long)]
        json: bool,
    },
    /// Remove a model from `~/.builderforce/models/`.
    Rm {
        /// Model name, file stem, or path under the store.
        model: String,
    },
    /// List models in the store.
    Ls,
    /// llama-bench metric names (`pp512`, `tg128`).
    Bench {
        /// Installed model name, path, or alias.
        model: String,
        /// Prompt-processing token count (metric `pp{N}`).
        #[arg(long, default_value_t = 512)]
        pp: usize,
        /// Token-generation count (metric `tg{N}`).
        #[arg(long, default_value_t = 128)]
        tg: usize,
    },
    /// Link Ollama blob-store GGUFs into our store (no re-download).
    ImportOllama {
        /// Ollama models dir. Default `$OLLAMA_MODELS` or `~/.ollama/models`.
        #[arg(long)]
        ollama_dir: Option<PathBuf>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(args).unwrap_or_else(|e| panic!("{e}"))
    }

    fn help_text(args: &[&str]) -> String {
        let err = Cli::try_parse_from(args).expect_err("help should not parse as Ok");
        err.to_string()
    }

    #[test]
    fn parses_every_subcommand() {
        assert!(matches!(parse(&["bfi", "ls"]).cmd, Commands::Ls));
        assert!(matches!(
            parse(&["bfi", "pull", "Qwen/Qwen3-0.6B-GGUF:Q4_K_M"]).cmd,
            Commands::Pull { .. }
        ));
        assert!(matches!(
            parse(&["bfi", "run", "qwen3", "hello"]).cmd,
            Commands::Run { .. }
        ));
        assert!(matches!(parse(&["bfi", "ps"]).cmd, Commands::Ps { .. }));
        assert!(matches!(
            parse(&["bfi", "fit", "m", "--ctx", "2048", "--json"]).cmd,
            Commands::Fit { json: true, ctx: 2048, .. }
        ));
        assert!(matches!(parse(&["bfi", "rm", "m"]).cmd, Commands::Rm { .. }));
        assert!(matches!(
            parse(&["bfi", "bench", "m", "--pp", "256", "--tg", "64"]).cmd,
            Commands::Bench { pp: 256, tg: 64, .. }
        ));
        assert!(matches!(
            parse(&["bfi", "import-ollama"]).cmd,
            Commands::ImportOllama { .. }
        ));
        match parse(&["bfi", "serve", "--port", "9", "--auth", "--models-dir", "/tmp/m"]).cmd {
            Commands::Serve {
                port,
                auth,
                models_dir,
            } => {
                assert_eq!(port, 9);
                assert!(auth);
                assert_eq!(models_dir.unwrap(), PathBuf::from("/tmp/m"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn help_for_each_command() {
        for cmd in [
            "pull",
            "run",
            "serve",
            "ps",
            "fit",
            "rm",
            "ls",
            "bench",
            "import-ollama",
        ] {
            let text = help_text(&["bfi", cmd, "--help"]);
            assert!(
                text.contains("Usage") || text.contains("Usage:"),
                "{cmd} help:\n{text}"
            );
        }
        let root = help_text(&["bfi", "--help"]);
        assert!(root.contains("pull"));
        assert!(root.contains("import-ollama"));
    }
}
