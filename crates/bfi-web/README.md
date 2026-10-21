# bfi-web

bfi's own engine in the browser. Runs PrismML's 1-bit **Bonsai** (`qwen3`
architecture, GGUF `Q1_0`) on **WebGPU** — or, with no GPU, on the **CPU** —
using the same crates as the native `bfi` binary (`bfi-gguf`, `bfi-core`,
`bfi-models`, `bfi-runtime`), exposed as the `BfiEngine` class from Builderforce
spec 34 §3.1. The same `.wasm` runs in a Node `worker_threads` worker (CPU only).

Published to npm as **`@seanhogg/bfi-web`** (see `package.json`; it ships the
`pkg/` that `wasm-pack` writes).

## Build

```sh
# once: rustup target add wasm32-unknown-unknown && cargo install wasm-pack
wasm-pack build crates/bfi-web --target web --release --out-dir pkg
```

Output lands in `crates/bfi-web/pkg/` (`bfi_web.js`, `bfi_web_bg.wasm`,
`bfi_web.d.ts`). The workspace `.cargo/config.toml` builds wasm32 with
`+simd128` (a `RUSTFLAGS` env var replaces it — keep the flag there). Type-check only:

```sh
cargo check -p bfi-web --target wasm32-unknown-unknown
```

Natively the crate is empty on purpose: all its dependencies are wasm32-only,
so `cargo build --workspace` never turns on `bfi-core/webgpu` for the native
`bfi` binary.

## API

```ts
import init, { BfiEngine } from "@seanhogg/bfi-web";
await init();

const s = await BfiEngine.support();        // adapter probe, no device
// Top level = the WebGPU adapter (contract v2, unchanged). Per backend:
// s.backends.webgpu = { supported, reason?, maxBufferBytes, maxStorageBufferBindingSizeBytes, f16 }
// s.backends.cpu    = { supported: true, simd, threads, kvBudgetBytes, maxMemoryBytes, deviceMemoryBytes? }

const engine = await BfiEngine.create();    // { backend?: "webgpu" | "cpu" | "auto" }; auto = WebGPU if an
                                            // adapter exists, else CPU. "webgpu" rejects "webgpu_unavailable: …"
engine.backend;                             // "webgpu" | "cpu"
const res = await fetch(".../Bonsai-1.7B-Q1_0.gguf");
// Predict the served context before downloading (same math as load):
const ctx = BfiEngine.contextFor({ layers: 28, kvHeads: 8, headDim: 128,
  trainedContext: 32768, f16: s.f16 });          // 18688 with f16, 9216 without
const cpuCtx = BfiEngine.contextFor({ layers: 28, kvHeads: 8, headDim: 128,
  trainedContext: 32768, backend: "cpu" });      // 2304 (f32 KV, 512 MiB budget)

const info = await engine.load(res.body!, Number(res.headers.get("content-length")),
  (loaded, total) => console.log(loaded / total),
  { kvBudgetBytes: 2 * 2 ** 30 });               // optional; 2 GiB WebGPU / 512 MiB CPU default
// info = { arch: "qwen3", backend, contextTokens, vocab, params, kvF16 }

const ac = new AbortController();
const final = await engine.chat(JSON.stringify({
  messages: [{ role: "user", content: "Hi" }],
  stream: true, max_tokens: 256, temperature: 0.7,
}), (chunkJson) => console.log(JSON.parse(chunkJson)), ac.signal);
// final = chat.completion JSON with usage + finish_reason

engine.unload();
```

### CPU backend

The CPU backend runs the native `bfi`'s own `CpuBackend` kernels (`Q1_0`
matmuls on the packed weights, no dequantized copy) through `bfi-models`'
`Qwen3StreamLoader` → `Session`; the chat loop is shared with WebGPU. It
computes on the calling thread and yields a macrotask (`setTimeout(0)`) after
every decode step and every 4 prompt tokens, so an `AbortSignal` lands
promptly — host it in a Worker to keep a page responsive.

- **SIMD**: the `Q1_0` row dot, the f32 dot and attention's `axpy` have wasm
  SIMD128 paths (`bfi-core/src/simd.rs`), proven bit-identical to lane-order
  scalar twins and within float rounding of the sequential reference
  (`wasm-pack test --node crates/bfi-core`).
- **Threads**: single-threaded (`threads: 1`). `wasm-bindgen-rayon` needs
  `-C target-feature=+atomics,+bulk-memory` and `-Z build-std` — a nightly
  toolchain — plus a cross-origin-isolated page; this build stays on stable.
- **Memory**: weights stay packed in wasm memory; the KV cache is f32,
  budgeted at `kvBudgetBytes` (default 512 MiB → Bonsai 1.7B 2 304 tokens,
  4B / 8B 1 792). wasm32 stops at 4 GiB: when weights + KV + a 256 MiB working
  set exceed it, `load` rejects `cpu_memory_exceeded:` from the header alone,
  before any weight is placed.

### `chat` request mapping

| OpenAI field | bfi |
|---|---|
| `messages` (string or content parts; `tool_calls`, `tool_call_id`, `name`) | rendered with the GGUF `tokenizer.chat_template` (minijinja + Python-method shims), ChatML fallback |
| `tools`, `tool_choice` | template `tools`; `required` / named choice → JSON-constrained decode; output parsed into `tool_calls` (`<tool_call>` XML or JSON) |
| `max_tokens` / `max_completion_tokens` | capped by the loaded context; omitted = until context end |
| `temperature` (default 0.8), `top_p`, `top_k`, `min_p`, `presence_penalty`, `seed` | `bfi_runtime::SampleParams` |
| `stop` (string or array) | held back while partially matched, cut on match |
| `stream` | `true` → `onChunk` receives `chat.completion.chunk` JSON; `false` → only the resolved completion |
| `stream_options.include_usage` | final usage-only chunk |
| `response_format: json_object / json_schema` | JSON-object-constrained decode |

The KV cache keeps the previous turn's tokens: a follow-up request that repeats
the conversation so far only prefills the new suffix.

### Errors

`Error.message` starts with a stable prefix:

| Prefix | When |
|---|---|
| `webgpu_unavailable:` | `create({ backend: "webgpu" })` with no `navigator.gpu` / no adapter, or the device was lost (create a new engine) |
| `cpu_memory_exceeded:` | CPU backend: packed weights + KV cache + working set exceed wasm32's 4 GiB (lower `kvBudgetBytes`) |
| `unsupported_type:` | a tensor type the WebGPU engine cannot run (names the tensor); WebGPU runs `Q1_0`, `Q8_0`, `F16`, `F32` matrices |
| `context_exceeded:` | the prompt does not fit the loaded context |
| `aborted` | `signal` fired, or `unload()` during `chat` |
| `invalid_request:` / `not_loaded:` / `busy:` / `load_failed:` / `engine_error:` | malformed request, chat before load, concurrent call, bad/truncated GGUF or architecture other than `qwen3`, other engine failures |

### Memory (WebGPU)

Weights stay packed on the GPU, one buffer per tensor (Bonsai 8B ≈ 1.15 GB,
1.7B ≈ 0.25 GB). The KV cache is f16 when the adapter has `shader-f16`
(else f32), budgeted at `kvBudgetBytes` total (default 2 GiB). If allocation
fails, `load` halves the context down to 2048 tokens. `contextTokens` reports
what was allocated: at the default budget with f16, Bonsai 8B / 4B ≈ 14 336
tokens and 1.7B ≈ 18 688 (f32: 7 168 / 9 216). The formula is in
`docs/PRD-bfi-web.md` §6. The host streams the file;
bfi-web holds at most the header plus one tensor in wasm memory while loading.
