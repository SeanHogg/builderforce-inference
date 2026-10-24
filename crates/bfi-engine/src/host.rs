//! What an engine needs from the realm it runs in.

use bfi_models::kv_plan::{DEFAULT_CPU_KV_BUDGET_BYTES, DEFAULT_KV_BUDGET_BYTES, WASM32_MEMORY_BYTES};

/// The realm's clock and scheduler.
#[allow(async_fn_in_trait)]
pub trait Host {
    /// Seconds since the Unix epoch — a completion's `created` (wasm32 has no `SystemTime`).
    fn now_secs(&self) -> u64;
    /// Let the realm's event loop run between CPU steps. A browser yields a macrotask so an
    /// `AbortSignal` fired by a click is seen; a native engine thread has nothing to
    /// yield to (its host's event loop runs on another thread) and returns at once.
    async fn yield_now(&self);
}

/// The CPU backend's memory rules in this realm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostProfile {
    /// The KV budget a CPU `load` (and `contextFor({ backend: 'cpu' })`) uses by default.
    pub cpu_kv_budget_bytes: u64,
    /// The hard ceiling weights + KV + working set must fit under on the CPU: wasm32's
    /// 4 GiB linear memory, or natively the RAM share the host plans to use.
    pub cpu_memory_cap: u64,
}

impl HostProfile {
    /// wasm32: 512 MiB of f32 KV by default, everything inside the 4 GiB linear memory.
    pub const WASM: Self = Self {
        cpu_kv_budget_bytes: DEFAULT_CPU_KV_BUDGET_BYTES,
        cpu_memory_cap: WASM32_MEMORY_BYTES,
    };

    /// A native process: native memory is not a 4 GiB arena, and the window is what lets
    /// a tool-heavy prompt fit, so the CPU's KV budget grows with the machine — a quarter
    /// of `cpu_memory_cap` (the RAM share the host reads off it), from wasm's 512 MiB up
    /// to the GPU's 2 GiB. A small machine keeps a model it can hold; a large one gets
    /// the long window.
    pub fn native(cpu_memory_cap: u64) -> Self {
        Self {
            cpu_kv_budget_bytes: (cpu_memory_cap / 4).clamp(DEFAULT_CPU_KV_BUDGET_BYTES, DEFAULT_KV_BUDGET_BYTES),
            cpu_memory_cap,
        }
    }
}
