//! Golden dequant vs a real GGUF (PRD §8 M2).
//!
//! Default `cargo test -p bfi-gguf` stays green without weights: this file's
//! live-GGUF case is `#[ignore]` and also no-ops when `BFI_GOLDEN_GGUF` is unset.
//!
//! Run the full sample against a Bonsai / qwen3 GGUF:
//!
//! ```sh
//! BFI_GOLDEN_GGUF=/path/to/model.gguf cargo test -p bfi-gguf --test golden_dequant -- --ignored --nocapture
//! ```
//!
//! Optional `BFI_GOLDEN_REF=/path/to/dir` holds llama.cpp C-reference dumps named
//! `<TYPE>-<i>.f32` (little-endian f32, one block). When present, we require
//! bit-exact match, or ≤1 ULP if `BFI_GOLDEN_ULP=1`.

use std::env;
use std::fs;
use std::path::Path;

use bfi_gguf::{
    lookup_type, GgufFile, TYPE_F16, TYPE_PQ2_0, TYPE_PTQ1_0, TYPE_Q1_0, TYPE_Q4_K, TYPE_Q5_K,
    TYPE_Q6_K, TYPE_Q8_0,
};

const GOLDEN_TYPES: &[(u32, &str)] = &[
    (TYPE_Q4_K, "Q4_K"),
    (TYPE_Q5_K, "Q5_K"),
    (TYPE_Q6_K, "Q6_K"),
    (TYPE_Q8_0, "Q8_0"),
    (TYPE_F16, "F16"),
    (TYPE_PQ2_0, "PQ2_0"),
    (TYPE_PTQ1_0, "PTQ1_0"),
    (TYPE_Q1_0, "Q1_0"),
];

const BLOCKS_PER_TYPE: usize = 8;

fn golden_gguf_path() -> Option<String> {
    env::var("BFI_GOLDEN_GGUF")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Deterministic mix of LCG indices so two runs on the same file sample the same 8 blocks.
fn sample_indices(n_blocks: usize, want: usize) -> Vec<usize> {
    if n_blocks == 0 {
        return Vec::new();
    }
    let take = want.min(n_blocks);
    let mut out = Vec::with_capacity(take);
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    while out.len() < take {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        let i = (x as usize) % n_blocks;
        if !out.contains(&i) {
            out.push(i);
        }
        if out.len() < take && out.len() == n_blocks {
            break;
        }
    }
    out
}

fn ulp_distance(a: f32, b: f32) -> u32 {
    if a == b || (a.is_nan() && b.is_nan()) {
        return 0;
    }
    let mut ai = a.to_bits() as i32;
    let mut bi = b.to_bits() as i32;
    if ai < 0 {
        ai = i32::MIN - ai;
    }
    if bi < 0 {
        bi = i32::MIN - bi;
    }
    (ai as i64 - bi as i64).unsigned_abs() as u32
}

fn read_f32_le(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_default();
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Always-on: the skip helper must treat an empty env as "not a failure".
#[test]
fn golden_dequant_skips_without_env() {
    // CI path: no weights. The ignored test is the live GGUF sample.
    let _ = golden_gguf_path();
}

/// Live GGUF sample. Ignored in CI; see the module docs for the command.
#[test]
#[ignore]
fn golden_dequant_eight_blocks_per_type() {
    let Some(path) = golden_gguf_path() else {
        eprintln!("BFI_GOLDEN_GGUF unset — skip (CI without weights)");
        return;
    };
    let file = GgufFile::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let allow_ulp = env::var("BFI_GOLDEN_ULP").ok().as_deref() == Some("1");
    let ref_dir = env::var("BFI_GOLDEN_REF").ok();

    let mut found_any = false;
    for &(id, name) in GOLDEN_TYPES {
        let ty = lookup_type(id).unwrap_or_else(|| panic!("missing type {name}"));
        let tensor = file
            .header()
            .tensors
            .iter()
            .find(|t| t.ggml_type == id);
        let Some(tensor) = tensor else {
            eprintln!("no {name} tensor in {path} — skip type");
            continue;
        };
        found_any = true;
        let bytes = file
            .tensor_bytes(&tensor.name)
            .unwrap_or_else(|e| panic!("{}: {e}", tensor.name));
        assert_eq!(
            bytes.len() % ty.block_bytes,
            0,
            "{}: tensor length {} is not a multiple of block {}",
            tensor.name,
            bytes.len(),
            ty.block_bytes
        );
        let n_blocks = bytes.len() / ty.block_bytes;
        let idxs = sample_indices(n_blocks, BLOCKS_PER_TYPE);
        assert!(
            !idxs.is_empty(),
            "{} ({name}) has 0 blocks",
            tensor.name
        );
        for (k, &bi) in idxs.iter().enumerate() {
            let src = &bytes[bi * ty.block_bytes..(bi + 1) * ty.block_bytes];
            let mut a = vec![0.0f32; ty.block_elems];
            let mut b = vec![0.0f32; ty.block_elems];
            (ty.dequant_row)(src, &mut a);
            (ty.dequant_row)(src, &mut b);
            assert_eq!(a, b, "{name} block {bi} dequant is not deterministic");

            if let Some(dir) = ref_dir.as_deref() {
                let dump = Path::new(dir).join(format!("{name}-{k}.f32"));
                if dump.is_file() {
                    let reference = read_f32_le(&dump);
                    assert_eq!(
                        reference.len(),
                        a.len(),
                        "{name}-{k}: ref len {} vs dequant {}",
                        reference.len(),
                        a.len()
                    );
                    for (j, (got, exp)) in a.iter().zip(reference.iter()).enumerate() {
                        if allow_ulp {
                            let d = ulp_distance(*got, *exp);
                            assert!(
                                d <= 1,
                                "{name}-{k}[{j}]: {got} vs {exp} = {d} ULP (limit 1)"
                            );
                        } else {
                            assert_eq!(
                                got.to_bits(),
                                exp.to_bits(),
                                "{name}-{k}[{j}]: {got} vs {exp} (set BFI_GOLDEN_ULP=1 to allow 1 ULP)"
                            );
                        }
                    }
                }
            }
        }
        eprintln!(
            "{name}: sampled {} blocks from {} ({} total blocks)",
            idxs.len(),
            tensor.name,
            n_blocks
        );
    }
    assert!(
        found_any,
        "GGUF {path} had none of the M2 types {:?}",
        GOLDEN_TYPES.iter().map(|t| t.1).collect::<Vec<_>>()
    );
}
