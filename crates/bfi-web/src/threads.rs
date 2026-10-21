//! The THREADED artifact's pool (feature `threads`, built by `build-threads.mjs` on the
//! pinned nightly with atomics + shared memory). `initThreads(n)` starts
//! wasm-bindgen-rayon's Web Worker pool; from then on `bfi-core`'s `par` helpers run on
//! rayon, and `support().backends.cpu.threads` reports `n`. `threadCheck` is the shipped
//! self-check: a Q1_0 matmul on the pool, bit-compared with the same matmul on one
//! thread, plus how many pool threads actually took work.
//!
//! Requires a cross-origin-isolated context (`SharedArrayBuffer`); the host's loader
//! picks this artifact only there. In the default artifact this module only reports 1.

#[cfg(feature = "threads")]
use std::sync::atomic::{AtomicU32, Ordering};

#[cfg(feature = "threads")]
static THREADS: AtomicU32 = AtomicU32::new(0);

/// Threads the CPU kernels run on.
pub fn count() -> u32 {
    #[cfg(feature = "threads")]
    return THREADS.load(Ordering::Relaxed).max(1);
    #[cfg(not(feature = "threads"))]
    1
}

/// `initThreads(n)`: start the pool of `n` workers (once; a second call rejects).
#[cfg(feature = "threads")]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = initThreads)]
pub async fn init_threads(n: u32) -> Result<f64, wasm_bindgen::JsValue> {
    let n = n.max(1);
    wasm_bindgen_futures::JsFuture::from(wasm_bindgen_rayon::init_thread_pool(n as usize)).await?;
    THREADS.store(n, Ordering::Relaxed);
    Ok(n as f64)
}

/// `threadCheck(rows)`: `{ threads, threadsUsed, identical, parallelMs, sequentialMs }`
/// for a `rows × 2048` Q1_0 matmul over 4 tokens.
#[cfg(feature = "threads")]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = threadCheck)]
pub fn thread_check(rows: u32) -> Result<wasm_bindgen::JsValue, wasm_bindgen::JsValue> {
    use bfi_core::{with_sequential, Backend, CpuBackend, DType};
    use wasm_bindgen::JsValue;

    // Waiting on the pool blocks; a browser's main thread may not block (Atomics.wait
    // throws there), so this runs only in a worker — where hosts run bfi anyway.
    if crate::js::present(&crate::js::get(&js_sys::global(), "document")) {
        return Err(crate::errors::error(
            crate::errors::Kind::InvalidRequest,
            "threadCheck must run in a worker, not on the page's main thread",
        ));
    }

    let (n_out, n_in, n_tokens) = ((rows.max(64) as usize) / 64 * 64, 2048usize, 4usize);
    let mut seed = 0x9E37_79B9u32;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let w: Vec<f32> = (0..n_out * n_in).map(|_| next()).collect();
    let mut bytes = vec![0u8; n_out * n_in / 128 * 18];
    for (b, blk) in bytes.chunks_exact_mut(18).enumerate() {
        bfi_gguf::quantize_row_q1_0(&w[b * 128..(b + 1) * 128], blk);
    }
    let xs: Vec<f32> = (0..n_tokens * n_in).map(|_| next()).collect();
    let cpu = CpuBackend;
    let weights = cpu.upload_quant(&[n_out, n_in], DType::Q1_0, &bytes).expect("weights");
    let x = cpu.upload_f32(&[n_tokens, n_in], &xs).expect("activations");

    let now = || js_sys::Date::now();
    let t0 = now();
    let one = with_sequential(|| cpu.matmul_q(&weights, &x).expect("sequential matmul"));
    let t1 = now();
    let many = cpu.matmul_q(&weights, &x).expect("parallel matmul");
    let t2 = now();

    // Which pool threads take work (each task records its rayon thread index).
    let used = std::sync::Mutex::new(std::collections::BTreeSet::new());
    rayon::scope(|s| {
        for _ in 0..(4 * count()) {
            s.spawn(|_| {
                let mut spin = 0u64;
                for i in 0..200_000u64 {
                    spin = spin.wrapping_add(i * i);
                }
                std::hint::black_box(spin);
                used.lock().unwrap().insert(rayon::current_thread_index());
            });
        }
    });
    let identical = one
        .as_f32()
        .unwrap()
        .iter()
        .zip(many.as_f32().unwrap())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    Ok(crate::js::object(&[
        ("threads", JsValue::from_f64(count() as f64)),
        ("threadsUsed", JsValue::from_f64(used.into_inner().unwrap().len() as f64)),
        ("identical", JsValue::from_bool(identical)),
        ("sequentialMs", JsValue::from_f64(t1 - t0)),
        ("parallelMs", JsValue::from_f64(t2 - t1)),
        ("rows", JsValue::from_f64(n_out as f64)),
    ]))
}
