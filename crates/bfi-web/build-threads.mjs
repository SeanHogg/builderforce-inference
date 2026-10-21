#!/usr/bin/env node
/**
 * The THREADED bfi-web artifact — `crates/bfi-web/pkg-threads/` — a SECOND, optional
 * build beside the default `pkg/` (`wasm-pack build crates/bfi-web --target web
 * --release --out-dir pkg`, stable, single-threaded, WebGPU + CPU).
 *
 * This one is CPU-only (`--no-default-features --features threads`) and runs the CPU
 * kernels on rayon over wasm-bindgen-rayon's Web Worker pool. Wasm threads need the
 * standard library rebuilt with atomics, which is nightly-only (`-Z build-std`), so the
 * toolchain is PINNED here — and only here: native builds and the default artifact stay
 * on stable (there is deliberately no repo-wide rust-toolchain.toml).
 *
 * A page may load it only when cross-origin isolated (`SharedArrayBuffer`); hosts call
 * `initThreads(navigator.hardwareConcurrency)` inside their bfi worker.
 *
 *   node crates/bfi-web/build-threads.mjs     (from the repo root or this directory)
 */
import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/** The nightly the threaded artifact is built with. Bump deliberately, never "nightly". */
export const THREADS_TOOLCHAIN = 'nightly-2026-10-15';
/** wasm32 target features: shared memory + atomics for the pool, SIMD128 as the default build. */
export const THREADS_RUSTFLAGS = [
  '-C', 'target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128',
  // A SHARED, imported memory (atomics alone leave it unshared, and the pool's
  // Atomics.wait then throws "not a shared typed array"), declaring its maximum —
  // wasm32's whole 4 GiB — plus the TLS exports wasm-bindgen's thread start needs.
  '-C', 'link-arg=--shared-memory',
  '-C', 'link-arg=--import-memory',
  '-C', 'link-arg=--max-memory=4294967296',
  '-C', 'link-arg=--export=__wasm_init_tls',
  '-C', 'link-arg=--export=__tls_size',
  '-C', 'link-arg=--export=__tls_align',
  '-C', 'link-arg=--export=__tls_base',
].join(' ');
export const THREADS_OUT_DIR = 'pkg-threads';

const crateDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(crateDir, '..', '..');

export function buildThreads() {
  const args = [
    'build', 'crates/bfi-web',
    '--target', 'web', '--release', '--out-dir', THREADS_OUT_DIR,
    '--no-default-features', '--features', 'threads',
  ];
  console.log(`[bfi-web threads] RUSTUP_TOOLCHAIN=${THREADS_TOOLCHAIN} wasm-pack ${args.join(' ')}`);
  // The toolchain and `-Z build-std` go through the environment, which every cargo /
  // rustc proxy wasm-pack spawns inherits (a `rustup run` wrapper or `-- -Z …` does
  // not reach the cargo wasm-pack resolves). RUSTFLAGS replaces `.cargo/config.toml`'s
  // wasm32 flags for this build only.
  const run = spawnSync('wasm-pack', args, {
    cwd: repoRoot,
    stdio: 'inherit',
    shell: process.platform === 'win32',
    env: {
      ...process.env,
      RUSTUP_TOOLCHAIN: THREADS_TOOLCHAIN,
      CARGO_UNSTABLE_BUILD_STD: 'panic_abort,std',
      // APPENDED to any RUSTFLAGS the caller set (an env RUSTFLAGS replaces
      // `.cargo/config.toml`'s, so the SIMD128 flag is carried in THREADS_RUSTFLAGS too).
      RUSTFLAGS: [process.env.RUSTFLAGS, THREADS_RUSTFLAGS].filter(Boolean).join(' '),
    },
  });
  if (run.status !== 0) {
    console.error(`[bfi-web threads] build failed — is ${THREADS_TOOLCHAIN} installed with rust-src and wasm32-unknown-unknown?`);
    process.exit(run.status ?? 1);
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) buildThreads();
