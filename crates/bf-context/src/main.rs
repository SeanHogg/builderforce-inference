//! `bf-context` — the headless context service.
//!
//!   bf-context serve                  run the service (what the desktop app does, minus the UI)
//!   bf-context mcp [--root <dir>]     MCP stdio server for Claude Code / Cursor (default root: cwd)
//!   bf-context index <dir>            index a workspace once and print its status
//!   bf-context search <dir> <query>   search a workspace and print the hits as JSON
//!   bf-context repo-map --root <dir> [--max-tokens N] [--focus <path>…] [--query <task>…]
//!                                     print the reference-ranked repo map (the query's
//!                                     named files/symbols personalize the ranking)
//!   bf-context compress [--rate R] < file
//!                                     LLMLingua-2 compress stdin to stdout (`compress` feature)

use anyhow::{anyhow, Result};
use bf_context::{api, api::ops, client::Client, mcp, paths, server, Registry};
use bf_index::Index;
use serde_json::json;
use std::io::Read;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("serve") => serve(),
        Some("mcp") => mcp::run_stdio(root_arg(&args[1..])?),
        Some("index") => {
            let root = PathBuf::from(args.get(1).ok_or_else(|| anyhow!("usage: bf-context index <dir>"))?);
            index_once(root)
        }
        Some("search") => {
            let root = args.get(1).ok_or_else(|| anyhow!("usage: bf-context search <dir> <query>"))?;
            let query = args[2..].join(" ");
            let out = Client::connect("cli").call(ops::SEARCH, json!({ "root": root, "query": query }))?;
            println!("{}", serde_json::to_string_pretty(&out)?);
            Ok(())
        }
        Some("repo-map") => repo_map(&args[1..]),
        Some("compress") => compress(&args[1..]),
        _ => {
            eprintln!(
                "usage: bf-context <serve | mcp [--root <dir>] | index <dir> | search <dir> <query> | repo-map --root <dir> [--max-tokens N] [--focus <path>…] | compress [--rate R] < file>"
            );
            std::process::exit(2);
        }
    }
}

/// The value after `flag`, if the flag is present.
fn flag_value<'a>(rest: &'a [String], flag: &str) -> Result<Option<&'a str>> {
    match rest.iter().position(|a| a == flag) {
        Some(i) => rest.get(i + 1).map(|v| Some(v.as_str())).ok_or_else(|| anyhow!("{flag} needs a value")),
        None => Ok(None),
    }
}

/// Every value after `flag` up to the next `--option`.
fn flag_values(rest: &[String], flag: &str) -> Vec<String> {
    match rest.iter().position(|a| a == flag) {
        Some(i) => rest[i + 1..].iter().take_while(|a| !a.starts_with("--")).cloned().collect(),
        None => Vec::new(),
    }
}

/// The repo map, from the running service when there is one (warm), else by bringing this
/// workspace's index up to date in-process first so the map is never empty mid-scan.
fn repo_map(rest: &[String]) -> Result<()> {
    let root = PathBuf::from(flag_value(rest, "--root")?.ok_or_else(|| anyhow!("usage: bf-context repo-map --root <dir>"))?);
    let max_tokens = match flag_value(rest, "--max-tokens")? {
        Some(v) => v.parse::<usize>().map_err(|_| anyhow!("--max-tokens must be a number"))?,
        None => api::DEFAULT_MAP_TOKENS,
    }
    .min(api::MAX_MAP_TOKENS);
    let focus = flag_values(rest, "--focus");
    // The task in hand, as free words up to the next option: `--query fix the login redirect`.
    let query = flag_values(rest, "--query").join(" ");
    let client = Client::connect("cli");
    let map = if client.is_remote() {
        let out = client.call(ops::REPO_MAP, json!({ "root": root, "maxTokens": max_tokens, "focus": focus, "query": query }))?;
        out.get("map").and_then(|m| m.as_str()).unwrap_or_default().to_string()
    } else {
        let root = paths::canonical_root(&root).map_err(|e| anyhow!("workspace {} not found: {e}", root.display()))?;
        let index = Index::open(&root, &paths::index_db(&paths::data_dir(), &root), None)?;
        index.scan()?;
        api::repo_map_for(&index, max_tokens, &focus, Some(&query))?
    };
    print!("{map}");
    Ok(())
}

/// stdin → compressed stdout; the token counts go to stderr so the output pipes cleanly.
fn compress(rest: &[String]) -> Result<()> {
    let rate = match flag_value(rest, "--rate")? {
        Some(v) => v.parse::<f32>().map_err(|_| anyhow!("--rate must be a number between 0 and 1"))?,
        None => api::DEFAULT_COMPRESS_RATE,
    };
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text)?;
    let out = Client::connect("cli").call(ops::COMPRESS, json!({ "text": text, "rate": rate }))?;
    print!("{}", out.get("text").and_then(|t| t.as_str()).unwrap_or_default());
    eprintln!(
        "bf-context: kept {} of {} tokens",
        out.get("keptTokens").and_then(|v| v.as_u64()).unwrap_or(0),
        out.get("originalTokens").and_then(|v| v.as_u64()).unwrap_or(0)
    );
    Ok(())
}

fn root_arg(rest: &[String]) -> Result<PathBuf> {
    match rest.iter().position(|a| a == "--root") {
        Some(i) => rest.get(i + 1).map(PathBuf::from).ok_or_else(|| anyhow!("--root needs a directory")),
        None => Ok(std::env::current_dir()?),
    }
}

fn serve() -> Result<()> {
    let reg = Registry::new(paths::data_dir());
    reg.start_background();
    let running = server::start(reg)?;
    eprintln!("bf-context {} listening on 127.0.0.1:{}", env!("CARGO_PKG_VERSION"), running.discovery.port);
    // Park until killed; the discovery file is withdrawn by the next start or goes stale.
    loop {
        std::thread::park();
    }
}

fn index_once(root: PathBuf) -> Result<()> {
    let reg = Registry::new(paths::data_dir());
    let index = reg.ensure(&root)?;
    index.scan()?;
    if index.embed_pending().is_err() {
        index.embedding_unavailable();
    }
    println!("{}", serde_json::to_string_pretty(&index.status())?);
    Ok(())
}
