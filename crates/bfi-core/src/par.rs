//! Data-parallel helpers: `rayon` natively and in the threaded browser build, plain
//! loops in the default wasm32 build.
//!
//! The browser has threads only on a cross-origin-isolated page with a worker pool:
//! `bfi-web`'s threaded artifact (`+atomics`, `wasm-bindgen-rayon`) runs these on rayon;
//! the default artifact runs them sequentially. Every CPU kernel goes through these
//! shapes so the split lives here, once. Each element / chunk is computed by the same
//! code on any thread, so results are bit-identical whatever the thread count.
//!
//! [`with_sequential`] runs a closure with parallelism off on the calling thread — the
//! single-thread reference the threaded build's self-check compares against.

use std::cell::Cell;

#[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
use rayon::prelude::*;

thread_local! {
    static SEQUENTIAL: Cell<bool> = const { Cell::new(false) };
}

/// Run `f` with every helper below sequential on this thread (nested calls included).
pub fn with_sequential<R>(f: impl FnOnce() -> R) -> R {
    let before = SEQUENTIAL.with(|s| s.replace(true));
    let out = f();
    SEQUENTIAL.with(|s| s.set(before));
    out
}

/// Parallel here: threads exist in this build and [`with_sequential`] is not active.
#[inline]
#[cfg_attr(all(target_arch = "wasm32", not(target_feature = "atomics")), allow(dead_code))]
fn parallel() -> bool {
    cfg!(any(not(target_arch = "wasm32"), target_feature = "atomics")) && !SEQUENTIAL.with(|s| s.get())
}

/// `out[i] = f(i, &mut out[i])` over every element.
pub fn for_each_indexed<T: Send>(out: &mut [T], f: impl Fn(usize, &mut T) + Sync + Send) {
    #[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
    if parallel() {
        out.par_iter_mut().enumerate().for_each(|(i, v)| f(i, v));
        return;
    }
    out.iter_mut().enumerate().for_each(|(i, v)| f(i, v));
}

/// `f(chunk_index, chunk)` over `chunk`-sized slices of `out`.
pub fn for_each_chunk_indexed<T: Send>(
    out: &mut [T],
    chunk: usize,
    f: impl Fn(usize, &mut [T]) + Sync + Send,
) {
    #[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
    if parallel() {
        out.par_chunks_mut(chunk).enumerate().for_each(|(i, c)| f(i, c));
        return;
    }
    out.chunks_mut(chunk).enumerate().for_each(|(i, c)| f(i, c));
}

/// `xs.map(f).collect()`.
pub fn map<T: Sync, U: Send>(xs: &[T], f: impl Fn(&T) -> U + Sync + Send) -> Vec<U> {
    #[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
    if parallel() {
        return xs.par_iter().map(f).collect();
    }
    xs.iter().map(f).collect()
}

/// `zip(a, b).map(f).collect()`.
pub fn zip_map<A: Sync, B: Sync, U: Send>(
    a: &[A],
    b: &[B],
    f: impl Fn(&A, &B) -> U + Sync + Send,
) -> Vec<U> {
    #[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
    if parallel() {
        return a.par_iter().zip(b.par_iter()).map(|(x, y)| f(x, y)).collect();
    }
    a.iter().zip(b.iter()).map(|(x, y)| f(x, y)).collect()
}

/// `(0..n).map(f).collect()`.
pub fn range_map<U: Send>(n: usize, f: impl Fn(usize) -> U + Sync + Send) -> Vec<U> {
    #[cfg(any(not(target_arch = "wasm32"), target_feature = "atomics"))]
    if parallel() {
        return (0..n).into_par_iter().map(f).collect();
    }
    (0..n).map(f).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_match_sequential() {
        let mut v = vec![0usize; 10];
        for_each_indexed(&mut v, |i, x| *x = i * 2);
        assert_eq!(v, (0..10).map(|i| i * 2).collect::<Vec<_>>());

        let mut c = vec![0usize; 6];
        for_each_chunk_indexed(&mut c, 2, |i, ch| ch.fill(i));
        assert_eq!(c, vec![0, 0, 1, 1, 2, 2]);

        assert_eq!(map(&[1, 2, 3], |x| x + 1), vec![2, 3, 4]);
        assert_eq!(zip_map(&[1, 2], &[10, 20], |a, b| a + b), vec![11, 22]);
        assert_eq!(range_map(3, |i| i * i), vec![0, 1, 4]);
    }

    #[test]
    fn with_sequential_turns_parallelism_off_and_restores_it() {
        assert_eq!(parallel(), cfg!(any(not(target_arch = "wasm32"), target_feature = "atomics")));
        with_sequential(|| {
            assert!(!parallel());
            with_sequential(|| assert!(!parallel()));
            assert!(!parallel());
        });
        assert_eq!(parallel(), cfg!(any(not(target_arch = "wasm32"), target_feature = "atomics")));
    }
}
