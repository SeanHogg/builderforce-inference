//! The CPU kernels on many threads equal the same kernels on one, bit for bit — the
//! property the threaded browser build relies on (`bfi-web` `threadCheck` proves the
//! same thing in wasm). Natively rayon is real threads; `with_sequential` is the
//! single-thread reference.

use bfi_core::{with_sequential, Backend, CpuBackend, DType};
use bfi_gguf::quantize_row_q1_0;

fn floats(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn q1_0_matmul_on_four_threads_equals_one_thread() {
    let (n_out, n_in, n_tokens) = (512usize, 1024usize, 3usize);
    let w = floats(n_out * n_in, 1);
    let mut bytes = vec![0u8; n_out * n_in / 128 * 18];
    for (b, blk) in bytes.chunks_exact_mut(18).enumerate() {
        quantize_row_q1_0(&w[b * 128..(b + 1) * 128], blk);
    }
    let cpu = CpuBackend;
    let weights = cpu.upload_quant(&[n_out, n_in], DType::Q1_0, &bytes).unwrap();
    let x = cpu.upload_f32(&[n_tokens, n_in], &floats(n_tokens * n_in, 2)).unwrap();

    let one = with_sequential(|| cpu.matmul_q(&weights, &x).unwrap());
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let (many, threads_used) = pool.install(|| {
        let used = std::sync::Mutex::new(std::collections::HashSet::new());
        let out = cpu.matmul_q(&weights, &x).unwrap();
        // Prove the pool really runs work on several threads.
        rayon::scope(|s| {
            for _ in 0..64 {
                s.spawn(|_| {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                    used.lock().unwrap().insert(rayon::current_thread_index());
                });
            }
        });
        (out, used.into_inner().unwrap().len())
    });
    assert!(threads_used > 1, "pool used {threads_used} thread(s)");
    let (a, b) = (one.as_f32().unwrap(), many.as_f32().unwrap());
    assert_eq!(a.len(), n_out * n_tokens);
    assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()), "threaded result differs");
}
