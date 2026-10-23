//! The service's operations, transport-free. The loopback HTTP server and the MCP server
//! are both thin adapters over `dispatch`, so a new operation is added once and every
//! client (VS Code, Claude Code, Cursor, the desktop UI) gets it.

use crate::activity::{now_ms, Entry};
use crate::registry::Registry;
use anyhow::{anyhow, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;

const DEFAULT_LIMIT: usize = 8;
pub const DEFAULT_MAP_TOKENS: usize = 1500;
pub const MAX_MAP_TOKENS: usize = 8000;
/// Share of prose tokens `compress` keeps when the caller names no rate.
pub const DEFAULT_COMPRESS_RATE: f32 = 0.5;
const MAX_REFERENCE_TEXTS: usize = 50;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RootReq {
    root: PathBuf,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchReq {
    root: PathBuf,
    query: String,
    limit: Option<usize>,
    path_prefix: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RepoMapReq {
    root: PathBuf,
    max_tokens: Option<usize>,
    #[serde(default)]
    focus: Vec<String>,
    /// The task in hand: files and symbols it names personalize the ranking.
    query: Option<String>,
}

/// The repo map for `index` with every option — the ONE place a request's fields become
/// [`bf_index::RepoMapOptions`], shared by the service op and the in-process CLI. A blank
/// query is no query, so it does not split the map cache.
pub fn repo_map_for(index: &bf_index::Index, max_tokens: usize, focus: &[String], query: Option<&str>) -> Result<String> {
    let query = query.map(str::trim).filter(|q| !q.is_empty());
    index.repo_map_with(&bf_index::RepoMapOptions { max_tokens: max_tokens.min(MAX_MAP_TOKENS), focus, query })
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CompressReq {
    text: String,
    rate: Option<f32>,
    #[serde(default)]
    force_keep: Vec<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RefsReq {
    root: PathBuf,
    texts: Vec<String>,
}

/// Operation names — the HTTP path segment (`POST /v1/<op>`) and the MCP tool mapping.
pub mod ops {
    pub const HEALTH: &str = "health";
    pub const WORKSPACES: &str = "workspaces";
    pub const ENSURE: &str = "ensure";
    pub const REMOVE: &str = "remove";
    pub const RESCAN: &str = "rescan";
    pub const STATUS: &str = "status";
    pub const SEARCH: &str = "search";
    pub const REPO_MAP: &str = "repo-map";
    pub const CHECK_REFERENCES: &str = "check-references";
    pub const ACTIVITY: &str = "activity";
    pub const COMPRESS: &str = "compress";
}

const ACTIVITY_LIMIT: usize = 100;

/// The operations worth showing a person in "Recent activity" — work a tool asked for.
/// Health probes, listings and the window's own polling are left out.
fn is_tracked(op: &str) -> bool {
    matches!(op, ops::SEARCH | ops::REPO_MAP | ops::CHECK_REFERENCES | ops::ENSURE | ops::COMPRESS)
}

/// `dispatch` on behalf of an outside client (`vscode`, `mcp`, …), recorded in the
/// registry's activity log. The desktop window calls `dispatch` directly: its own
/// polling is not activity.
pub fn dispatch_from(reg: &Arc<Registry>, client: &str, op: &str, body: Value) -> Result<Value> {
    if !is_tracked(op) {
        return dispatch(reg, op, body);
    }
    let started = std::time::Instant::now();
    let root = body.get("root").and_then(Value::as_str).map(str::to_owned);
    let detail = describe(op, &body);
    let result = dispatch(reg, op, body);
    reg.activity.record(Entry {
        at: now_ms(),
        client: client.to_string(),
        op: op.to_string(),
        root,
        detail,
        count: result.as_ref().ok().and_then(|v| result_count(op, v)),
        ms: started.elapsed().as_millis() as u64,
        ok: result.is_ok(),
    });
    result
}

fn describe(op: &str, body: &Value) -> Option<String> {
    match op {
        ops::SEARCH => body.get("query").and_then(Value::as_str).map(str::to_owned),
        ops::REPO_MAP => body
            .get("query")
            .and_then(Value::as_str)
            .filter(|q| !q.trim().is_empty())
            .map(str::to_owned)
            .or_else(|| {
                body.get("focus")
                    .and_then(Value::as_array)
                    .filter(|f| !f.is_empty())
                    .map(|f| f.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(", "))
            }),
        ops::CHECK_REFERENCES => body.get("texts").and_then(Value::as_array).map(|t| t.len().to_string()),
        ops::COMPRESS => body.get("text").and_then(Value::as_str).map(|t| format!("{} chars", t.chars().count())),
        _ => None,
    }
}

/// Search → hits returned; memory check → texts found to name missing code.
fn result_count(op: &str, v: &Value) -> Option<usize> {
    let results = v.get("results")?.as_array()?;
    match op {
        ops::SEARCH => Some(results.len()),
        ops::CHECK_REFERENCES => Some(
            results
                .iter()
                .filter(|r| r.get("missing").and_then(Value::as_array).is_some_and(|m| !m.is_empty()))
                .count(),
        ),
        _ => None,
    }
}

pub fn dispatch(reg: &Arc<Registry>, op: &str, body: Value) -> Result<Value> {
    match op {
        ops::HEALTH => Ok(serde_json::to_value(reg.health())?),
        ops::WORKSPACES => Ok(json!({ "workspaces": reg.statuses() })),
        ops::ACTIVITY => Ok(json!({ "entries": reg.activity.recent(ACTIVITY_LIMIT) })),
        ops::ENSURE | ops::STATUS => {
            let r: RootReq = serde_json::from_value(body)?;
            Ok(serde_json::to_value(reg.ensure(&r.root)?.status())?)
        }
        ops::REMOVE => {
            let r: RootReq = serde_json::from_value(body)?;
            Ok(json!({ "removed": reg.remove(&r.root)? }))
        }
        ops::RESCAN => {
            let r: RootReq = serde_json::from_value(body)?;
            reg.rescan(&r.root)?;
            Ok(json!({ "ok": true }))
        }
        ops::SEARCH => {
            let r: SearchReq = serde_json::from_value(body)?;
            if r.query.trim().is_empty() {
                return Err(anyhow!("query is empty"));
            }
            let index = reg.ensure(&r.root)?;
            let results = index.search(&r.query, r.limit.unwrap_or(DEFAULT_LIMIT), r.path_prefix.as_deref())?;
            Ok(json!({ "results": results, "status": index.status() }))
        }
        ops::REPO_MAP => {
            let r: RepoMapReq = serde_json::from_value(body)?;
            let index = reg.ensure(&r.root)?;
            let tokens = r.max_tokens.unwrap_or(DEFAULT_MAP_TOKENS);
            let map = repo_map_for(&index, tokens, &r.focus, r.query.as_deref())?;
            Ok(json!({ "map": map, "status": index.status() }))
        }
        ops::CHECK_REFERENCES => {
            let r: RefsReq = serde_json::from_value(body)?;
            if r.texts.len() > MAX_REFERENCE_TEXTS {
                return Err(anyhow!("at most {MAX_REFERENCE_TEXTS} texts per call"));
            }
            let index = reg.ensure(&r.root)?;
            Ok(json!({ "results": index.check_references(&r.texts)?, "status": index.status() }))
        }
        ops::COMPRESS => {
            let r: CompressReq = serde_json::from_value(body)?;
            let rate = r.rate.unwrap_or(DEFAULT_COMPRESS_RATE);
            if !(rate.is_finite() && (0.0..=1.0).contains(&rate)) {
                return Err(anyhow!("rate must be between 0 and 1"));
            }
            let force: Vec<&str> = r.force_keep.iter().map(String::as_str).collect();
            Ok(serde_json::to_value(reg.compressor.compress(&r.text, rate, &force)?)?)
        }
        other => Err(anyhow!("unknown operation '{other}'")),
    }
}
