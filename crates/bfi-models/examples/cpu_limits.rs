//! Prints the CPU backend's planning constants as JSON — the ONE source the TypeScript
//! fit reads. `Builderforce.ai/scripts/vendor-bfi-web.mjs` runs this and writes
//! `packages/local-inference/src/bfiLimits.generated.ts` from it (recorded in
//! `packages/bfi-web/SOURCE.json`, held by `--check`).
//!
//!   cargo run -q -p bfi-models --example cpu_limits

use bfi_models::kv_plan::{
    ASSUMED_DEVICE_MEMORY_BYTES, CPU_WORKING_SET_BYTES, DEFAULT_CPU_KV_BUDGET_BYTES,
    DEVICE_MEMORY_SHARE_DIVISOR, WASM32_MEMORY_BYTES,
};

fn main() {
    println!(
        "{{\"cpuKvBudgetBytes\":{DEFAULT_CPU_KV_BUDGET_BYTES},\"cpuWorkingSetBytes\":{CPU_WORKING_SET_BYTES},\"wasm32MemoryBytes\":{WASM32_MEMORY_BYTES},\"deviceMemoryShareDivisor\":{DEVICE_MEMORY_SHARE_DIVISOR},\"assumedDeviceMemoryBytes\":{ASSUMED_DEVICE_MEMORY_BYTES}}}"
    );
}
