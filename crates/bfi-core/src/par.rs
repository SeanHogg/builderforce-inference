//! Data-parallel helpers: `rayon` natively, plain loops on wasm32.
//!
//! The browser has no threads unless the page opts into cross-origin isolation
//! plus a worker pool, and `bfi-web` runs its heavy math on WebGPU anyway. Every
//! CPU kernel goes through these five shapes so the target split lives here, once.

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// `out[i] = f(i, &mut out[i])` over every element.
pub fn for_each_indexed<T: Send>(out: &mut [T], f: impl Fn(usize, &mut T) + Sync + Send) {
    #[cfg(not(target_arch = "wasm32"))]
    out.par_iter_mut().enumerate().for_each(|(i, v)| f(i, v));
    #[cfg(target_arch = "wasm32")]
    out.iter_mut().enumerate().for_each(|(i, v)| f(i, v));
}

/// `f(chunk_index, chunk)` over `chunk`-sized slices of `out`.
pub fn for_each_chunk_indexed<T: Send>(
    out: &mut [T],
    chunk: usize,
    f: impl Fn(usize, &mut [T]) + Sync + Send,
) {
    #[cfg(not(target_arch = "wasm32"))]
    out.par_chunks_mut(chunk).enumerate().for_each(|(i, c)| f(i, c));
    #[cfg(target_arch = "wasm32")]
    out.chunks_mut(chunk).enumerate().for_each(|(i, c)| f(i, c));
}

/// `xs.map(f).collect()`.
pub fn map<T: Sync, U: Send>(xs: &[T], f: impl Fn(&T) -> U + Sync + Send) -> Vec<U> {
    #[cfg(not(target_arch = "wasm32"))]
    return xs.par_iter().map(f).collect();
    #[cfg(target_arch = "wasm32")]
    return xs.iter().map(f).collect();
}

/// `zip(a, b).map(f).collect()`.
pub fn zip_map<A: Sync, B: Sync, U: Send>(
    a: &[A],
    b: &[B],
    f: impl Fn(&A, &B) -> U + Sync + Send,
) -> Vec<U> {
    #[cfg(not(target_arch = "wasm32"))]
    return a.par_iter().zip(b.par_iter()).map(|(x, y)| f(x, y)).collect();
    #[cfg(target_arch = "wasm32")]
    return a.iter().zip(b.iter()).map(|(x, y)| f(x, y)).collect();
}

/// `(0..n).map(f).collect()`.
pub fn range_map<U: Send>(n: usize, f: impl Fn(usize) -> U + Sync + Send) -> Vec<U> {
    #[cfg(not(target_arch = "wasm32"))]
    return (0..n).into_par_iter().map(f).collect();
    #[cfg(target_arch = "wasm32")]
    return (0..n).map(f).collect();
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
}
