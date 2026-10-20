# bfi-web: 1-bit Bonsai in the browser on bfi's own crates

| | |
|---|---|
| Status | Implemented, pending first wasm32 build + browser run |
| Contract | Builderforce.ai `docs/specs/34-prd-local-models-everywhere.md` §3.1 (`BfiEngine`) |
| Companion | `docs/PRD-native-engine.md` (native `bfi`; this doc does not change its scope) |
| Written | 2026-10-17 |

## 1. Goal

Run PrismML's 1-bit **Bonsai** (`prism-ml/Bonsai-{1.7B,4B,8B}-gguf`, file
`Bonsai-<N>-Q1_0.gguf`; `general.architecture = qwen3`) in a browser tab on
WebGPU, with the same crates the native engine uses, behind the `BfiEngine` JS
class: `create`, `support`, `load(ReadableStream, totalBytes, onProgress)`,
`chat(requestJson, onChunk, signal)`, `unload`.

Model facts used: 8B = 36 layers, GQA 32 q / 8 kv heads, head_dim 128, trained
context 65 536, untied `output.weight`; 1.7B / 4B tie the lm_head to
`token_embd.weight`.

## 2. `Q1_0` (verified against the fork)

Source: `PrismML-Eng/llama.cpp`, branch `prism`, commit
`0082ac24a89ae62132041b40e2c0170e278a98f4` (MIT, "The ggml authors").

| | |
|---|---|
| GGML type id | **41** (`ggml.h`: `GGML_TYPE_Q1_0 = 41`, upstream slot; `Q2_0 = 42`) |
| Block | `#define QK1_0 128`; `typedef struct { ggml_half d; uint8_t qs[QK1_0/8]; } block_q1_0;` — **18 bytes / 128 weights** (1.125 bpw) |
| Bit order | weight `j` = `(qs[j/8] >> (j%8)) & 1`, LSB first (`dequantize_row_q1_0`) |
| Value | bit set ⇒ `+d`, clear ⇒ `-d` |
| Reference quantizer | `d = mean(|x|)`, bit set when `x >= 0` (`quantize_row_q1_0_ref`) |

Registry row + one file: `crates/bfi-gguf/src/types/q1_0.rs` (dequant, fused
`vec_dot`, reference quantizer, synthetic-block tests). The golden-dequant test
lists `Q1_0` so a live Bonsai GGUF is checked like the other types.

## 3. Why a resident GPU path (and not `Backend`)

`bfi_core::Backend` is synchronous over host tensors: `GpuBackend` uploads
weights per call and reads every result back with a blocking
`client.read_one`. A browser cannot block on a GPU readback (CubeCL's
`read_sync` fails on wasm), and per-op round trips would be hopeless anyway.

So `bfi-core` gains **`resident`** (feature `webgpu`): weights, activations and
the KV cache live on the device; a forward pass only *enqueues* kernels; the
host awaits exactly one readback per token (`Gpu::read_f32`, the logits). The
same module runs natively on wgpu (Vulkan / Metal / DX12), which is how its
kernels are tested against `CpuBackend` without a browser.

## 4. Layering (one module per concern)

| Crate | Added | Notes |
|---|---|---|
| `bfi-gguf` | `types/q1_0.rs`; `source.rs` (`ByteSource`: mmap native, owned on wasm32); `stream.rs` (`TensorStream` push parser) | `memmap2` is a non-wasm32 dependency. Peak memory while streaming = header + one tensor. |
| `bfi-core` | `kernels/` (`#[cube]`, feature `cube`): `bytes` (in-shader byte / f16 unpack), `gemm` (Q1_0, Q8_0, F16, F32), `rows` (get_rows), `ops` (RMSNorm, NEOX RoPE, causal GQA attention, SwiGLU, add, copy); `resident/` (feature `webgpu`); `par.rs` (rayon native, sequential wasm32); `DType::Q1_0`; `RopeParams::neox` | The Q1_0 GEMM kernel is shared: native `GpuBackend::matmul_q` (cuda/rocm/vulkan/metal) launches it through `gpu_kernels::gemv_q1_or_cpu`. |
| `bfi-models` | `kv_plan.rs` (pure served-context math, always compiled); `qwen3_resident.rs` (feature `webgpu`): `Qwen3ResidentLoader` (streaming upload, shape checks, `kv_shape`), `Qwen3Resident` (forward, prefix rewind, f16/f32 KV) | Untied lm_head supported (falls back to `token_embd` when `output.weight` is absent). |
| `bfi-runtime` | `openai.rs` (request → `ChatPlan`; `ChatTurn` sampling / stop strings / tool calls / chunk JSON); `jinja_compat.rs`; `byte_level.rs`; ChatML fallback in `chat.rs` | `store` (filesystem) and `dirs` are native-only. `bfi-server` now uses the shared `content_text`, `apply_tools`, `prepend_tools_block`, `render_prompt`, `completion_value`, `chunk_value`. |
| `bfi-web` (new) | `engine.rs` (the class), `support.rs`, `loader.rs`, `chat.rs`, `errors.rs`, `js.rs` | All deps are wasm32-only, so the native workspace never unifies `bfi-core/webgpu` into `bfi`. |

### Kernel design

- Weights are the GGUF block bytes, padded to whole `u32` words, one buffer per
  tensor. Bytes and f16 scales are unpacked with integer ops (WGSL has no byte
  loads; `shader-f16` is optional). Nothing widens a matrix to f32; only 1-D norm
  vectors are dequantized on the host (a few KB per layer).
- GEMM: one 64-lane workgroup per output row, a tile of 8 tokens per workgroup
  (prefill reads a weight row once per 8 tokens), lanes stride the row, a
  shared-memory tree reduces. Q1_0 lanes take whole sign bytes:
  `acc += d · (Σ set x − Σ clear x)`. Rows fold into grid X·Z so a 151 936-row
  lm_head stays under WebGPU's 65 535-per-dimension limit.
- Attention: scores (`[rows, n_past + t]`, causal mask as −3e38), row softmax,
  weighted sum that never reads cache rows past the causal limit. Prefill runs
  in chunks of 64 tokens to bound the score buffer.
- Barriers only in uniform control flow (WGSL uniformity analysis).

## 5. wasm async device init

CubeCL's normal client path (`DeviceService::init`) calls
`future::block_on(create_setup_for_device(..))`, which panics on wasm.
`resident::open_webgpu()` instead awaits
`cubecl::wgpu::init_setup_async::<WebGpu>(&WgpuDevice::DefaultDevice, RuntimeOptions::default())`:
it requests a high-performance adapter, then a device with the adapter's own
limits and features (so `shader-f16` is enabled when offered), and registers
the server for `DefaultDevice`; `WgpuRuntime::client(&DefaultDevice)` then
loads that server. Shader compile errors on wasm are logged from a
`spawn_local` task (CubeCL), not returned.

CubeCL `expect`s an adapter and a wasm panic aborts the module, so
`BfiEngine.create()` first probes `navigator.gpu.requestAdapter()` itself and
rejects with `webgpu_unavailable:` before touching CubeCL. `support()` uses the
same probe and never creates a device.

## 6. Memory and context

- Weights: Bonsai 8B ≈ 1.15 GB, 1.7B ≈ 0.25 GB of packed buffers. The largest
  single buffer is the 8B embedding table (151 936 × 4 096 Q1_0 ≈ 88 MB), under
  WebGPU's default 128 MiB `maxStorageBufferBindingSize`.
- **KV cache precision.** f16 (two halves per `u32` word, packed in-shader so
  it needs no WGSL `f16`) when the adapter offers `shader-f16`; f32 otherwise.
  Attention reads either layout (`attn_scores_f16` / `attn_values_f16`).
- **Served context** (`bfi_models::kv_plan::served_context`, also exposed as
  `BfiEngine.contextFor` so the host can predict it before downloading):

  ```text
  bytes_per_elem = f16 ? 2 : 4
  per_token      = layers · 2 · kvHeads · headDim · bytes_per_elem
  per_buffer_row = kvHeads · headDim · bytes_per_elem          (one layer's K or V)
  ctx = min(trainedContext,
            floor(kvBudgetBytes  / per_token),
            floor(maxBufferBytes / per_buffer_row))
  ctx = ctx ≥ 256 ? floor(ctx / 256) · 256 : ctx
  ```

  `kvBudgetBytes` caps *total* KV memory (default 2 GiB, `load(…, { kvBudgetBytes })`);
  every layer's K and V are separate buffers, so the per-binding limit only
  applies to one of them. `maxBufferBytes` = `min(maxStorageBufferBindingSize,
  maxBufferSize)` of the adapter (`contextFor` takes the min of
  `maxStorageBufferBindingSizeBytes` and `maxBufferSizeBytes` — or its
  deprecated alias `maxBufferBytes` — when given, and defaults to WebGPU's
  guaranteed 128 MiB). The attention-score scratch does not cap the context: long contexts
  use smaller prefill chunks (`max_binding / (n_head · ctx · 4)`, at most 64).
- **Allocation retry.** Before allocating, `load` probes `2 · layers` buffers of
  the planned size inside WebGPU out-of-memory + validation error scopes (CubeCL
  does not expose scopes, so bfi-core keeps the raw `wgpu::Device` from
  `init_setup_async`), then allocates the real cache inside the same scopes
  (weight uploads are scoped too). On either failure it halves the context
  (rounded to 256) down to `min(2048, trainedContext)`; a planned context
  already below that floor (small budget or binding) is tried as-is. If every
  attempt fails, `load` rejects `load_failed:` listing the contexts tried.
  A lost device (`device.lost`) drops the model; later calls reject
  `webgpu_unavailable:`. `BfiModelInfo.contextTokens` is what was allocated, and
  `kvF16` says which layout.
- Predicted served context at the default 2 GiB budget (128 MiB binding, no
  OOM). Shapes from the HF `config.json` of `prism-ml/Bonsai-{8B,4B,1.7B}-unpacked`;
  asserted in `kv_plan` tests:

  | Model | layers · kvHeads · headDim | trained | KV / token f16 | f16 ctx | f32 ctx |
  |---|---|---|---|---|---|
  | Bonsai 8B | 36 · 8 · 128 | 65 536 | 144 KiB | **14 336** | 7 168 |
  | Bonsai 4B | 36 · 8 · 128 | 32 768 | 144 KiB | **14 336** | 7 168 |
  | Bonsai 1.7B | 28 · 8 · 128 | 32 768 | 112 KiB | **18 688** | 9 216 |

  A 4 GiB budget roughly doubles these (8B f16 → 28 928), capped by the trained context.
- Loading never holds the file: chunks → `TensorStream` → upload → drop.

## 7. Tokenizer correctness fixes found on the way (native too)

- **Byte-level BPE.** `tokenizer.ggml.model = "gpt2"` vocabularies (Qwen) store
  tokens in GPT-2's byte-level alphabet (`Ġ` = space, `Ċ` = newline). The
  tokenizer mapped raw text to tokens directly, which fails on the first space
  of a real Qwen GGUF and decodes `Ġ` literally. `byte_level.rs` now maps both
  ways when the model is `gpt2`.
- **Special tokens.** Control / user-defined tokens (`token_type` 3/4, e.g.
  `<|im_start|>`) are now matched verbatim before BPE, as llama.cpp does for
  templated prompts.
- **RoPE layout.** llama.cpp uses NEOX RoPE for `qwen3` (dim `i` pairs with
  `i + n_rot/2`); the CPU reference rotated adjacent pairs. `RopeParams::neox`
  is set for `qwen3`; `qwen35` keeps its current path (its multi-section RoPE
  should be re-checked against the fork's `rope_multi` separately).
- **Chat templates.** HF templates call Python string/dict methods and
  `tojson`; `jinja_compat.rs` adds the subset Qwen uses and minijinja's `json`
  feature is on. ChatML-shaped templates that still fail fall back to ChatML,
  not `role: content`.

## 8. Verification

Native (no browser needed):

- `cargo test -p bfi-gguf` — Q1_0 registry / bit order / vec_dot / quantizer,
  `TensorStream` chunking, `GgufFile::from_bytes`.
- `cargo test -p bfi-core` — `par`, NEOX RoPE, Q1_0 fused GEMV vs CPU.
- `cargo test -p bfi-core --features webgpu` and
  `cargo test -p bfi-models --features webgpu` — resident kernels and a two-layer
  all-Q1_0 Bonsai-shaped qwen3 vs the CPU qwen3 (prompt + decode logits). Each
  test returns early when no wgpu adapter enumerates.
- `cargo test -p bfi-runtime -p bfi-server` — tokenizer, templates, `openai`
  turn (streaming, stop strings, tool calls, length, context errors).

wasm32:

- `cargo check -p bfi-web --target wasm32-unknown-unknown`
- `wasm-pack build crates/bfi-web --target web --release --out-dir pkg`

Still to do in a browser: load the real `Bonsai-1.7B-Q1_0.gguf` and `-8B`,
compare greedy output with the PrismML fork on a few prompts, and measure
tok/s on an integrated and a discrete GPU.

## 9. Known limits

- Single sequence; prefix reuse across turns, no batching.
- Resident weight types: `Q1_0`, `Q8_0`, `F16`, `F32` (others → `unsupported_type:`).
- Without `shader-f16` the KV cache is f32 (half the f16 context). The f16
  path packs with integer ops, so it could run everywhere; it is gated on the
  adapter feature to stay conservative.
- Upload copies a tensor twice on the host (`create_from_slice` copies), so wasm
  memory peaks near 3× the largest tensor (~260 MB for 8B) during load.
