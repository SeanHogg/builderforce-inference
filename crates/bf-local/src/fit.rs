//! Desktop fit path: call [`bfi_runtime::fit`] when a GGUF is on disk; otherwise the
//! catalog bits-per-weight estimate, returned as the same [`FitReport`] the engine uses.
//!
//! Hub listing used to do the bits math inline. Folding it here means a recommendation
//! is "the highest-quality variant whose `fit.fits` is true", whether or not the file
//! has been downloaded yet.

use std::path::Path;

use bfi_runtime::{FitBreakdown, FitReport};

use crate::hub::Quant;

/// Catalog-only overhead (weights + working memory) until a GGUF header exists.
/// Matches the figure the hub tests pin (~1.2 GB on top of the weights).
pub const CATALOG_OVERHEAD_BYTES: u64 = 1_200_000_000;

/// Default context the catalog path assumes; header-based fit uses the same.
pub const DEFAULT_CTX: usize = 4096;

/// Bits-per-weight catalog estimate as a [`FitReport`].
pub fn estimate_from_bits(params_b: f64, quant: Quant, budget_bytes: u64) -> FitReport {
    let weights = (params_b * 1e9 * quant.bits() / 8.0) as u64;
    let overhead = CATALOG_OVERHEAD_BYTES;
    let need = weights.saturating_add(overhead);
    FitReport {
        fits: need <= budget_bytes,
        need_bytes: need,
        budget_bytes,
        breakdown: FitBreakdown {
            weights,
            kv_cache: 0,
            recurrent_state: 0,
            activations: 0,
            overhead,
        },
        max_ctx_that_fits: DEFAULT_CTX,
        device: "cpu".into(),
    }
}

/// Prefer [`bfi_runtime::fit`] (header-only GGUF) when `gguf` exists; otherwise bits-per-weight.
pub fn catalog_fit(params_b: f64, quant: Quant, budget_bytes: u64, gguf: Option<&Path>) -> FitReport {
    if let Some(path) = gguf {
        if path.is_file() {
            if let Ok(report) = bfi_runtime::fit(path, DEFAULT_CTX, budget_bytes) {
                return report;
            }
        }
    }
    estimate_from_bits(params_b, quant, budget_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    #[test]
    fn bits_path_marks_fits_from_budget() {
        let small = estimate_from_bits(7.6, Quant::Q4, GB);
        assert!(!small.fits);
        assert!(small.need_bytes > 5 * GB);
        let big = estimate_from_bits(7.6, Quant::Q4, 8 * GB);
        assert!(big.fits);
        assert_eq!(big.device, "cpu");
        assert_eq!(big.breakdown.overhead, CATALOG_OVERHEAD_BYTES);
    }

    #[test]
    fn invalid_gguf_falls_back_to_bits_per_weight() {
        let dir = tempfile::tempdir().unwrap();
        let junk = dir.path().join("not.gguf");
        std::fs::write(&junk, b"nope").unwrap();
        let bits = estimate_from_bits(7.6, Quant::Q4, 8 * GB);
        let via = catalog_fit(7.6, Quant::Q4, 8 * GB, Some(&junk));
        assert_eq!(via.need_bytes, bits.need_bytes);
        assert_eq!(via.fits, bits.fits);
    }

    #[test]
    fn missing_path_uses_catalog_bits() {
        let missing = Path::new("/no/such/model.gguf");
        let bits = estimate_from_bits(3.2, Quant::Q8, 16 * GB);
        let via = catalog_fit(3.2, Quant::Q8, 16 * GB, Some(missing));
        assert_eq!(via.need_bytes, bits.need_bytes);
    }
}
