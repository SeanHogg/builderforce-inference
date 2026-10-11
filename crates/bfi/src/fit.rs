//! `bfi fit` — header-only sizing. Full formula is ticket 2927.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};
use bfi_gguf::{ModelHeader, HEADER_HINT_BYTES};
use serde::Serialize;

use crate::hf::{self, looks_like_path, parse_spec, RemoteFile};
use crate::store::format_bytes;

const STUB_OVERHEAD: u64 = 100 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FitJson {
    pub fits: bool,
    pub need_bytes: u64,
    pub budget_bytes: u64,
    pub device: String,
    pub max_ctx_that_fits: usize,
    pub breakdown: FitBreakdownJson,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FitBreakdownJson {
    pub weights: u64,
    pub kv: u64,
    pub state: u64,
    pub activations: u64,
    pub overhead: u64,
}

/// Stub breakdown: exact `weights` from the GGUF header; KV/state/activations
/// left at 0 until ticket 2927 lands the PRD formula.
pub fn fit_from_header(header: &ModelHeader, ctx: usize, budget_bytes: u64, device: &str) -> Result<FitJson> {
    let weights = header.weight_bytes().context("tensor sizes from header")?;
    let breakdown = FitBreakdownJson {
        weights,
        kv: 0,
        state: 0,
        activations: 0,
        overhead: STUB_OVERHEAD,
    };
    let need_bytes = breakdown.weights
        + breakdown.kv
        + breakdown.state
        + breakdown.activations
        + breakdown.overhead;
    let fits = need_bytes <= budget_bytes;
    Ok(FitJson {
        fits,
        need_bytes,
        budget_bytes,
        device: device.to_string(),
        max_ctx_that_fits: if fits { ctx } else { 0 },
        breakdown,
    })
}

pub fn cpu_budget_bytes() -> u64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    (sys.total_memory() as f64 * 0.75) as u64
}

pub fn device_label() -> String {
    std::env::var("BFI_BACKEND")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "cpu".into())
}

/// Read only the GGUF header prefix of a local file (not the weights).
pub fn header_from_local(path: &Path) -> Result<ModelHeader> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut buf = vec![0u8; HEADER_HINT_BYTES];
    let n = f.read(&mut buf)?;
    buf.truncate(n);
    ModelHeader::from_bytes(&buf).map_err(|e| anyhow::anyhow!("GGUF header {}: {e}", path.display()))
}

/// Header-only HTTP range (`HEADER_HINT_BYTES`) so fit works before `pull`.
pub fn header_from_remote(remote: &RemoteFile) -> Result<ModelHeader> {
    let end = HEADER_HINT_BYTES.saturating_sub(1) as u64;
    let bytes = hf::http_range(&remote.url, 0, end)?;
    ModelHeader::from_bytes(&bytes).map_err(|e| anyhow::anyhow!("GGUF header from {}: {e}", remote.url))
}

pub fn resolve_header(model: &str) -> Result<ModelHeader> {
    if looks_like_path(model) {
        return header_from_local(Path::new(model));
    }
    let local = bfi_runtime::model_path(model);
    if local.exists() {
        return header_from_local(&local);
    }
    let aliases = bfi_runtime::aliases().unwrap_or_default();
    let spec = parse_spec(model, &aliases);
    if looks_like_path(&spec.repo) {
        return header_from_local(Path::new(&spec.repo));
    }
    let remote = hf::resolve_remote(&spec)?;
    header_from_remote(&remote)
}

pub fn print_fit(report: &FitJson, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(report)?);
        return Ok(());
    }
    println!(
        "fits:           {}",
        if report.fits { "yes" } else { "no" }
    );
    println!("need:           {}", format_bytes(report.need_bytes));
    println!("budget:         {}", format_bytes(report.budget_bytes));
    println!("device:         {}", report.device);
    println!("max ctx:        {}", report.max_ctx_that_fits);
    println!("  weights:      {}", format_bytes(report.breakdown.weights));
    println!("  kv:           {}", format_bytes(report.breakdown.kv));
    println!("  state:        {}", format_bytes(report.breakdown.state));
    println!(
        "  activations:  {}",
        format_bytes(report.breakdown.activations)
    );
    println!("  overhead:     {}", format_bytes(report.breakdown.overhead));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bfi_gguf::writer::{write_gguf, TensorWrite};
    use bfi_gguf::{lookup_type, MetaValue, TYPE_F32};
    use std::collections::BTreeMap;

    fn tiny_header() -> ModelHeader {
        let mut meta = BTreeMap::new();
        meta.insert(
            "general.architecture".into(),
            MetaValue::String("qwen3".into()),
        );
        let n = 8u64;
        let mut data = Vec::new();
        for _ in 0..n {
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }
        let bytes = write_gguf(
            &meta,
            &[TensorWrite {
                name: "token_embd.weight".into(),
                dims: vec![n],
                ty: lookup_type(TYPE_F32).unwrap(),
                data,
            }],
        )
        .unwrap();
        ModelHeader::from_bytes(&bytes).unwrap()
    }

    #[test]
    fn stub_json_shape() {
        let h = tiny_header();
        let report = fit_from_header(&h, 4096, 10 * 1024 * 1024 * 1024, "cpu").unwrap();
        assert!(report.fits);
        assert_eq!(report.breakdown.weights, 32);
        assert_eq!(report.breakdown.kv, 0);
        assert_eq!(report.breakdown.state, 0);
        assert_eq!(report.breakdown.activations, 0);
        assert_eq!(report.breakdown.overhead, STUB_OVERHEAD);
        assert_eq!(report.need_bytes, 32 + STUB_OVERHEAD);
        assert_eq!(report.max_ctx_that_fits, 4096);
        assert_eq!(report.device, "cpu");
        let v = serde_json::to_value(&report).unwrap();
        assert!(v.get("needBytes").is_some());
        assert!(v.get("budgetBytes").is_some());
        assert!(v.get("maxCtxThatFits").is_some());
        assert!(v["breakdown"].get("weights").is_some());
        assert!(v["breakdown"].get("kv").is_some());
        assert!(v["breakdown"].get("state").is_some());
        assert!(v["breakdown"].get("activations").is_some());
        assert!(v["breakdown"].get("overhead").is_some());
    }

    #[test]
    fn header_from_local_prefix() {
        let h = tiny_header();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.gguf");
        // Reconstruct via writer again so we have bytes on disk.
        let mut meta = BTreeMap::new();
        meta.insert(
            "general.architecture".into(),
            MetaValue::String("qwen3".into()),
        );
        let mut data = Vec::new();
        for _ in 0..8u64 {
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }
        let bytes = write_gguf(
            &meta,
            &[TensorWrite {
                name: "token_embd.weight".into(),
                dims: vec![8],
                ty: lookup_type(TYPE_F32).unwrap(),
                data,
            }],
        )
        .unwrap();
        std::fs::write(&path, &bytes).unwrap();
        let got = header_from_local(&path).unwrap();
        assert_eq!(got.weight_bytes().unwrap(), h.weight_bytes().unwrap());
    }
}
