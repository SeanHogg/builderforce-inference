//! `bfi fit` — PRD §5.7 memory estimate from a GGUF header.
//!
//! Math lives in `bfi_runtime::FitReport::from_header`. This module keeps the
//! CLI JSON shape, local/remote header fetch (HTTP Range, no weights), and
//! human/JSON printers.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};
use bfi_gguf::{ModelHeader, HEADER_HINT_BYTES};
use bfi_runtime::FitReport;
use serde::Serialize;

use crate::hf::{self, looks_like_path, parse_spec, RemoteFile};
use crate::store::format_bytes;

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

/// Map the runtime §5.7 report onto the CLI JSON object.
pub fn fit_from_header(
    header: &ModelHeader,
    ctx: usize,
    budget_bytes: u64,
    device: &str,
) -> Result<FitJson> {
    let report = FitReport::from_header(header, ctx, budget_bytes, device)?;
    Ok(FitJson {
        fits: report.fits,
        need_bytes: report.need_bytes,
        budget_bytes: report.budget_bytes,
        device: report.device,
        max_ctx_that_fits: report.max_ctx_that_fits,
        breakdown: FitBreakdownJson {
            weights: report.breakdown.weights,
            kv: report.breakdown.kv_cache,
            state: report.breakdown.recurrent_state,
            activations: report.breakdown.activations,
            overhead: report.breakdown.overhead,
        },
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

    fn f32_tensor(name: &str, n: u64) -> TensorWrite {
        let mut data = Vec::new();
        for _ in 0..n {
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }
        TensorWrite {
            name: name.into(),
            dims: vec![n],
            ty: lookup_type(TYPE_F32).unwrap(),
            data,
        }
    }

    fn tiny_header() -> ModelHeader {
        let mut meta = BTreeMap::new();
        meta.insert(
            "general.architecture".into(),
            MetaValue::String("qwen3".into()),
        );
        let bytes = write_gguf(&meta, &[f32_tensor("token_embd.weight", 8)]).unwrap();
        ModelHeader::from_bytes(&bytes).unwrap()
    }

    fn qwen3_header() -> ModelHeader {
        let mut meta = BTreeMap::new();
        meta.insert(
            "general.architecture".into(),
            MetaValue::String("qwen3".into()),
        );
        meta.insert("qwen3.block_count".into(), MetaValue::U32(4));
        meta.insert("qwen3.embedding_length".into(), MetaValue::U32(16));
        meta.insert("qwen3.attention.head_count".into(), MetaValue::U32(4));
        meta.insert("qwen3.attention.head_count_kv".into(), MetaValue::U32(2));
        meta.insert("qwen3.attention.key_length".into(), MetaValue::U32(8));
        let bytes = write_gguf(&meta, &[f32_tensor("token_embd.weight", 8)]).unwrap();
        ModelHeader::from_bytes(&bytes).unwrap()
    }

    fn qwen35_header() -> ModelHeader {
        let mut meta = BTreeMap::new();
        meta.insert(
            "general.architecture".into(),
            MetaValue::String("qwen35".into()),
        );
        meta.insert("qwen35.block_count".into(), MetaValue::U32(2));
        meta.insert("qwen35.embedding_length".into(), MetaValue::U32(8));
        meta.insert("qwen35.attention.head_count".into(), MetaValue::U32(2));
        meta.insert("qwen35.attention.head_count_kv".into(), MetaValue::U32(1));
        meta.insert("qwen35.attention.key_length".into(), MetaValue::U32(4));
        meta.insert(
            "qwen35.attention.layer_types".into(),
            MetaValue::Array(vec![
                MetaValue::String("linear_attention".into()),
                MetaValue::String("full_attention".into()),
            ]),
        );
        let bytes = write_gguf(&meta, &[f32_tensor("token_embd.weight", 8)]).unwrap();
        ModelHeader::from_bytes(&bytes).unwrap()
    }

    #[test]
    fn json_shape_camel_case() {
        let h = tiny_header();
        let report = fit_from_header(&h, 4096, 10 * 1024 * 1024 * 1024, "cpu").unwrap();
        assert!(report.fits);
        assert_eq!(report.breakdown.weights, 32);
        // no block_count → no KV/state/activations; overhead is the 100 MiB runtime term
        assert_eq!(report.breakdown.kv, 0);
        assert_eq!(report.breakdown.state, 0);
        assert_eq!(report.breakdown.activations, 0);
        assert_eq!(report.breakdown.overhead, 100 * 1024 * 1024);
        assert_eq!(report.need_bytes, 32 + 100 * 1024 * 1024);
        assert_eq!(report.device, "cpu");
        // no KV growth: any ctx fits, so max is the search cap (262K)
        assert_eq!(report.max_ctx_that_fits, 262_144);
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
    fn qwen3_kv_not_stub_zero() {
        let h = qwen3_header();
        let report = fit_from_header(&h, 1024, 10 * 1024 * 1024 * 1024, "cpu").unwrap();
        assert_eq!(report.breakdown.kv, 524_288);
        assert_eq!(report.breakdown.state, 0);
        assert_eq!(report.breakdown.activations, 1_024);
        assert!(report.fits);
    }

    #[test]
    fn qwen35_hybrid_kv_and_state() {
        let h = qwen35_header();
        let report = fit_from_header(&h, 8, 10 * 1024 * 1024 * 1024, "cpu").unwrap();
        assert_eq!(report.breakdown.kv, 256);
        assert_eq!(report.breakdown.state, 64);
        assert_eq!(report.device, "cpu");
    }

    #[test]
    fn header_from_local_prefix() {
        let h = tiny_header();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.gguf");
        let mut meta = BTreeMap::new();
        meta.insert(
            "general.architecture".into(),
            MetaValue::String("qwen3".into()),
        );
        let bytes = write_gguf(&meta, &[f32_tensor("token_embd.weight", 8)]).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        let got = header_from_local(&path).unwrap();
        assert_eq!(got.weight_bytes().unwrap(), h.weight_bytes().unwrap());
    }
}
