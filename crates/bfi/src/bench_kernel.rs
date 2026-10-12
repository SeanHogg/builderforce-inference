//! `bfi bench-kernel` — fused PQ2_0 / PTQ1_0 GEMV timing (PRD M1).
//!
//! Compiles without a GGUF on disk: omit `--gguf` and a synthetic Bonsai-27B
//! hidden=5120 matrix is generated.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use bfi_core::{
    fused_gemv_pq2_0, fused_gemv_ptq1_0, select_backend, DType, Tensor,
};
use bfi_gguf::{quantize_row_pq2_0, GgufFile, TYPE_PQ2_0, TYPE_PTQ1_0};

const SYNTH_LAYERS: usize = 48;
const SYNTH_GEMV_PER_LAYER: usize = 7;

pub fn run(
    gguf: Option<PathBuf>,
    n_out: usize,
    n_in: usize,
    iters: usize,
    dtype: &str,
) -> Result<()> {
    let backend = select_backend();
    println!("format=bfi-bench-kernel/1");
    println!("backend={}", backend.name());
    println!("iters={iters}");

    match gguf {
        Some(path) => run_gguf(&path, iters)?,
        None => run_synthetic(n_out, n_in, iters, dtype)?,
    }
    Ok(())
}

fn run_synthetic(n_out: usize, n_in: usize, iters: usize, dtype: &str) -> Result<()> {
    if n_in == 0 || n_out == 0 || n_in % 128 != 0 {
        bail!("synthetic GEMV needs n_out>0 and n_in multiple of 128, got {n_out}x{n_in}");
    }
    let iters = iters.max(1);
    let kind = parse_dtype(dtype)?;
    let x = vec![0.01f32; n_in];
    let bytes = match kind {
        DType::PQ2_0 => synthetic_pq2(n_out, n_in),
        DType::PTQ1_0 => synthetic_ptq1(n_out, n_in),
        _ => unreachable!(),
    };
    let gemv = |b: &[u8]| -> Result<Vec<f32>> {
        match kind {
            DType::PQ2_0 => fused_gemv_pq2_0(b, &x, n_out, n_in),
            DType::PTQ1_0 => fused_gemv_ptq1_0(b, &x, n_out, n_in),
            _ => unreachable!(),
        }
    };
    let _ = gemv(&bytes)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = gemv(&bytes)?;
    }
    let secs = t0.elapsed().as_secs_f64() / iters as f64;
    let gemv_s = if secs > 0.0 { 1.0 / secs } else { 0.0 };
    let tok_s = gemv_s / (SYNTH_LAYERS * SYNTH_GEMV_PER_LAYER) as f64;
    println!("source=synthetic");
    println!("dtype={kind:?}");
    println!("n_out={n_out}");
    println!("n_in={n_in}");
    println!("hidden={n_in}");
    println!("gemv_s={gemv_s:.4}");
    println!("gemv_us={:.1}", secs * 1e6);
    println!("layers={SYNTH_LAYERS}");
    println!("gemv_per_layer={SYNTH_GEMV_PER_LAYER}");
    println!("tok_s={tok_s:.4}");
    time_backend_matmul(kind, n_out, n_in, &bytes, &x, iters)?;
    Ok(())
}

fn run_gguf(path: &PathBuf, iters: usize) -> Result<()> {
    let file = GgufFile::open(path).with_context(|| format!("open {}", path.display()))?;
    let header = file.header();
    let layers = header.arch_u64("block_count").unwrap_or(0) as usize;
    println!("source={}", path.display());
    println!("arch={}", header.architecture().unwrap_or("-"));
    println!("model={}", header.name().unwrap_or("-"));
    println!("layers={layers}");
    println!("hidden={}", header.arch_u64("embedding_length").unwrap_or(0));

    let iters = iters.max(1);
    let mut total_s = 0.0f64;
    let mut n_timed = 0usize;
    println!("tensor\tdtype\tn_out\tn_in\tgemv_us\tgemv_s");
    for t in &header.tensors {
        if t.dims.len() != 2 {
            continue;
        }
        let dtype = match t.ggml_type {
            TYPE_PQ2_0 => DType::PQ2_0,
            TYPE_PTQ1_0 => DType::PTQ1_0,
            _ => continue,
        };
        let n_in = t.dims[0] as usize;
        let n_out = t.dims[1] as usize;
        if n_in == 0 || n_in % 128 != 0 {
            continue;
        }
        let bytes = file.tensor_bytes(&t.name)?;
        let x = vec![0.01f32; n_in];
        let gemv = |b: &[u8]| -> Result<Vec<f32>> {
            match dtype {
                DType::PQ2_0 => fused_gemv_pq2_0(b, &x, n_out, n_in),
                DType::PTQ1_0 => fused_gemv_ptq1_0(b, &x, n_out, n_in),
                _ => unreachable!(),
            }
        };
        let _ = gemv(bytes)?;
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = gemv(bytes)?;
        }
        let secs = t0.elapsed().as_secs_f64() / iters as f64;
        let gemv_s = if secs > 0.0 { 1.0 / secs } else { 0.0 };
        println!(
            "{}\t{:?}\t{n_out}\t{n_in}\t{:.1}\t{gemv_s:.4}",
            t.name,
            dtype,
            secs * 1e6
        );
        total_s += secs;
        n_timed += 1;
    }
    if n_timed == 0 {
        println!("note=no PQ2_0/PTQ1_0 2-D tensors; ran nothing");
        return Ok(());
    }
    let attn_s = total_s * 0.10;
    let tok_s = if total_s + attn_s > 0.0 {
        1.0 / (total_s + attn_s)
    } else {
        0.0
    };
    println!("tensors_timed={n_timed}");
    println!("sum_gemv_s={total_s:.6}");
    println!("attn_norm_est_s={attn_s:.6}");
    println!("tok_s={tok_s:.4}");
    Ok(())
}

fn time_backend_matmul(
    dtype: DType,
    n_out: usize,
    n_in: usize,
    bytes: &[u8],
    x: &[f32],
    iters: usize,
) -> Result<()> {
    let backend = select_backend();
    let w = backend.upload_quant(&[n_out, n_in], dtype, bytes)?;
    let xt = Tensor::f32(vec![1, n_in], x.to_vec());
    let _ = backend.matmul_q(&w, &xt)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = backend.matmul_q(&w, &xt)?;
    }
    let secs = t0.elapsed().as_secs_f64() / iters as f64;
    let gemv_s = if secs > 0.0 { 1.0 / secs } else { 0.0 };
    println!("backend_matmul_us={:.1}", secs * 1e6);
    println!("backend_gemv_s={gemv_s:.4}");
    Ok(())
}

fn parse_dtype(s: &str) -> Result<DType> {
    match s.trim().to_ascii_lowercase().as_str() {
        "pq2" | "pq2_0" | "pq2-0" => Ok(DType::PQ2_0),
        "ptq1" | "ptq1_0" | "ptq1-0" => Ok(DType::PTQ1_0),
        other => bail!("dtype must be pq2 or ptq1, got {other}"),
    }
}

fn synthetic_pq2(n_out: usize, n_in: usize) -> Vec<u8> {
    let n_blocks = n_in / 128;
    let mut block = vec![0u8; 34];
    quantize_row_pq2_0(&[0.5f32; 128], &mut block);
    let mut bytes = Vec::with_capacity(n_out * n_blocks * 34);
    for _ in 0..n_out {
        for _ in 0..n_blocks {
            bytes.extend_from_slice(&block);
        }
    }
    bytes
}

fn synthetic_ptq1(n_out: usize, n_in: usize) -> Vec<u8> {
    let n_blocks = n_in / 128;
    let mut block = vec![0u8; 28];
    block[26..28].copy_from_slice(&0x4000u16.to_le_bytes());
    let mut bytes = Vec::with_capacity(n_out * n_blocks * 28);
    for _ in 0..n_out {
        for _ in 0..n_blocks {
            bytes.extend_from_slice(&block);
        }
    }
    bytes
}
