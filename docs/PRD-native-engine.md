# PRD: native inference engine (`bfi`) — Synapse replaces Ollama

| | |
|---|---|
| Status | Product cut-over shipped (M2–M6). Remaining: M1 70% GPU tok/s unmeasured; M3 live Bonsai 27B greedy vs fork not run; M7 signed GitHub Release binaries |
| Owner | Synapse / builderforce-inference |
| Written | 2026-10-11 |
| Tracks | Builderforce.ai `ROADMAP.md` → "Local LLM execution" row (🔄 Partial until the three remaining items below) |

**Delivery (2026-10-14).** The product behaviour this PRD asked for is in the tree. Synapse supervises `bfi serve`; `bf-local/src/ollama.rs` and `SYNAPSE_USE_OLLAMA` are gone (`crates/bfi/src/ollama.rs` is the import-ollama CLI only). CubeCL GPU backends compile (`open_backend` for cuda/rocm/metal/vulkan); `mix_decode` + `tools`/`tool_choice` round-trip; VSIX discovers Synapse via `~/.builderforce/desktop.json`; blog `synapse-runs-models-itself.md` is live. **Not claimed:** 70% GPU tok/s vs llama-bench (ticket #3762 / #4465 — needs Metal + NVIDIA hardware); live Bonsai 27B greedy vs the PrismML fork (ticket #3764 — no 27B weights in CI); signed GitHub Release binaries (tickets #3782 / #4481 — owner signing secrets). Do not move the ROADMAP row to DONE until those three land.

## 1. Why

*(Historical — written 2026-10-11. Cut-over shipped; see Delivery above. The three costs below are why we built `bfi`.)*

Synapse ran local models by driving a separately installed Ollama (`crates/bf-local/src/ollama.rs`). That had three costs:

1. **An install step outside our control.** A person must install Ollama before Synapse's local models work. If it is missing, the Models view shows a download link to someone else's product.
2. **The formats we most want don't run on it.** PrismML's Ternary Bonsai 2 27B ships as `PQ2_0` / `PTQ1_0` GGUF, and stock llama.cpp (which Ollama embeds) rejects both types. Only PrismML's llama.cpp fork runs them.
3. **Sizing guesses.** `bf-local/src/hub.rs` estimates memory from a fixed bits-per-weight table (`Q4_K_M` = 4.85, `Q8_0` = 8.5, `F16` = 16). It knows nothing about ternary formats, hybrid-attention KV cost or the runtime's real overhead.

**Goal:** a pure-Rust engine in this repo, `bfi`, that Synapse ships and supervises. It loads any GGUF we write a loader for, including the ternary formats. It runs on every GPU Synapse's users have: Vulkan (any Windows GPU), CUDA, ROCm, Metal, plus a CPU fallback. It speaks the OpenAI, Anthropic and Ollama APIs, so every existing client works unchanged. After cut-over, Synapse has no Ollama dependency.

## 2. Already done (context, not scope)

Steps 1–4 of the extraction finished on 2026-10-11:
- Synapse lives here and Spawn lives in `builderforce-spawn`. Both pass `cargo check --workspace`.
- Each repo has a `release.yml` triggered by `v*` tags. The signing actions live here.
- The Spawn release was carried over as `v2026.10.2`.
- Builderforce.ai's links point here, and its `desktop/` folder is gone.
- Synapse's in-app update check reads this repo's `v*` releases.

**Outstanding from that work:** the owner must add the signing secrets to both repos: Azure Trusted Signing or a `.pfx`, Apple Developer ID plus notarization, and the Tauri updater key.

## 3. Facts this design rests on (verified 2026-10-11)

| Fact | Source |
|---|---|
| PrismML's fork (`PrismML-Eng/llama.cpp`, default branch `prism`) is **MIT**: "Copyright (c) 2023-2026 The ggml authors". Kernels may be ported if each ported file keeps the MIT notice. | GitHub API licence + `LICENSE` |
| The Bonsai model weights are **Apache-2.0**. | HF model card |
| Bonsai 2 27B's GGUF `general.architecture` is **`qwen35`**: hybrid attention, about **75% linear attention (gated delta net)** and 25% full attention, plus SwiGLU, RMSNorm and multi-section RoPE (`ggml_rope_multi`). It has 27.36B parameters (24.35B backbone, 2.54B embedding/LM head, 0.46B vision) and a 262K context. | HF card; fork `src/models/qwen35.cpp` |
| The full-attention GGML ops used by `qwen35`: `get_rows`, `rope_multi`, `swiglu_split`, `sigmoid`, `softplus`, `silu`, `l2_norm`, `ssm_conv`, `concat`. The linear layers also use the delta-net base (`llm_build_delta_net_base`). | fork `src/models/qwen35.cpp` (785 lines) |
| Files: `PQ2_0` is 7.21 GB (2.13 bpw); `PTQ1_0` is 5.95 GB (1.75 bpw). The vision projector `mmproj` (Q8_0, 0.63 GB) is optional. | HF card |
| The headline speed is **~47 tok/s decode (tg128) on an M5 Max**, PQ2_0, 765 tok/s prompt processing (pp512). It came from a *pre-rotation* build and is marked "pending re-measurement". Other figures: RTX 4090 = 81.2 (PQ2_0) / 91.1 (PTQ1_0); M5 Pro = 28.1. | HF card |

### 3.1 Exact block layouts (port these bit-for-bit)

GGML type ids: **`PQ2_0 = 142`**, **`PTQ1_0 = 143`** (fork `ggml.h`). Upstream `Q2_0` is 42 (group 64) and `TQ1_0` is 34 (group 256). Neither can represent a group-128 checkpoint losslessly.

**`block_pq2_0`**: 128 weights in 34 bytes.
```
f16  d;          // scale
u8   qs[32];     // 2 bits per weight, little-end first: weight j = (qs[j/4] >> (2*(j%4))) & 3
value = (q - 1) * d       // 00=-1, 01=0, 10=+1, 11=+2 (11 unused by ternary checkpoints)
```

**`block_ptq1_0`**: 128 weights in 28 bytes. Note that the scale comes **last**.
```
u8   qs[24];     // 5 trits per byte -> 120 weights
u8   qh[2];      // 4 trits per byte ->   8 weights
f16  d;          // scale
```
Decoding uses the fork's `dequantize_row_ptq1_0`, with stages `{32, 16, 8}` and `pow3 = {1,3,9,27,81,243}`:
- The 24-byte `qs` is consumed in stages. A 32-byte stage doesn't fit, so the 16-byte stage covers bytes 0..16 and the 8-byte stage covers bytes 16..24.
- Within a stage of `c` bytes starting at byte `b`, the output order is `for n in 0..5 { for m in 0..c { trit(qs[b+m], n) } }`.
- Then `for n in 0..4 { for h in 0..2 { trit(qh[h], n) } }`.
- `trit(byte, n) = ((u16)((u8)(byte * pow3[n])) * 3) >> 8`. The value is `(trit - 1) * d`.

**Golden test:** take 3 blocks from each file of the real Bonsai GGUF. Dequantize them with the fork's C function (compiled once, with output committed as a fixture) and with `bfi-gguf`. The results must match exactly, bit-for-bit in f32.

## 4. Non-goals (this PRD)

- Training or fine-tuning. Quantizing models ourselves (`bfi` reads GGUF and never writes it).
- Our own model registry or hosting. Models come straight from Hugging Face.
- Vision input (`mmproj`). This is a later PRD; the loader must *skip* the file cleanly, not fail.
- Multi-GPU / tensor parallelism. Speculative decoding with the MTP layers comes later; the loader keeps those tensors and doesn't run them.
- Embedding Synapse's engine in-process. Synapse supervises a child process (§6).

## 5. Architecture

```
            ┌──────────────────────── builderforce-inference ────────────────────────┐
 HF Hub ──► │ bfi CLI ─ pull/run/serve/ps/fit/rm/bench                              │
            │   │                                                                     │
            │   ▼                                                                     │
            │ bfi-server  (OpenAI /v1/*, Anthropic /v1/messages, Ollama /api/*)       │
            │   │                                                                     │
            │ bfi-runtime (scheduler · KV/recurrent state · sampler · tokenizer ·     │
            │   │          chat templates · model store)                              │
            │ bfi-models  (registry: arch name → layer graph; qwen3, qwen35 first)    │
            │   │                                                                     │
            │ bfi-core    (Tensor + Backend trait; CubeCL kernels: wgpu/Vulkan,       │
            │   │          CUDA, ROCm/HIP, Metal; CPU SIMD reference)                 │
            │ bfi-gguf    (GGUF v3 reader, mmap, type registry incl. PQ2_0/PTQ1_0)    │
            └─────────────────────────────────────────────────────────────────────────┘
 Synapse app ── bf-local::runtime supervises `bfi serve` (child process) ──► loopback
```

Each crate depends only on the crates below it in the diagram. **No crate may grow into a god module.** The rules that enforce this are below.

### 5.1 `bfi-gguf`
- GGUF v2/v3 reader. Header, metadata KV (all value types including arrays) and tensor infos are read from a `memmap2` mapping. Tensor data is zero-copy (`&[u8]` slices with the 32-byte default alignment, or `general.alignment`).
- **The type registry is data.** It is a table of `GgmlType { id, name, block_elems, block_bytes, dequant_row: fn, vec_dot: Option<fn> }`. Supported at M2: `F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, PQ2_0, PTQ1_0`. Adding a type means adding one table row and one file, with no edits elsewhere.
- An unknown type gives an error that names the type id and the tensor. It must never produce silent garbage (stock llama.cpp's failure mode on Bonsai).
- `ModelHeader::bytes_by_tensor()` is the exact source of truth for `bfi fit`.

### 5.2 `bfi-core`
- `Tensor { shape, dtype, storage: Storage }` and the `trait Backend`: `alloc`, `upload`, `matmul_q(weights: QTensor, x)`, `rms_norm`, `rope_multi`, `softmax`, `silu`, `swiglu`, `sigmoid`, `softplus`, `l2_norm`, `conv1d_causal` (for `ssm_conv`), `gated_delta_step`, `attention(q,k,v, mask)`, `add`, `mul`, `get_rows`.
- **The CPU backend is the reference.** It is plain Rust plus `std::simd` / `pulp`, multithreaded with `rayon`. Every GPU kernel is tested against it within tolerance.
- **CubeCL backend:**
  - Each kernel is written once with `#[cube]` and compiled to the `wgpu` (Vulkan/Metal/DX12), `cuda` and `hip` runtimes behind cargo features `vulkan`, `cuda`, `rocm`, `metal`.
  - Pin one exact CubeCL version in `[workspace.dependencies]`. Pick the newest release that is at least two weeks old when M1 starts.
- **The kernels that matter, in order:**
  1. a fused dequant+GEMV for `PQ2_0` and `PTQ1_0` (decode, batch 1)
  2. dequant+GEMM for prompt processing
  3. `Q4_K` / `Q6_K` / `Q8_0` GEMV
  4. flash-style attention decode
  5. gated-delta-net recurrent step
- Device selection is automatic: CUDA, then ROCm, then Metal, then Vulkan, then CPU. The order can be overridden with `BFI_BACKEND`. `bfi ps` reports the chosen device.

### 5.3 `bfi-models`
- `trait Architecture { fn name() -> &'static str; fn from_gguf(&ModelHeader) -> Result<Box<dyn Model>>; }`. Architectures self-register through `inventory`, which keeps the registry open/closed. Each architecture is one module, and a new model never edits the engine.
- `trait Model { fn forward(&self, batch: &Batch, state: &mut SeqState) -> Logits; fn state_layout(&self) -> StateLayout; }`
- **M2:** `qwen3` (dense, all full attention, Q/K-norm per head, GQA, SwiGLU). This is the correctness baseline against small public GGUFs such as Qwen3-0.6B/1.7B.
- **M3:** `qwen35`, the Bonsai architecture.
  - Full-attention layers use multi-section RoPE.
  - Linear-attention layers are gated delta net: a causal `ssm_conv` over the mixed QKV, `l2_norm` on q/k, a `softplus`/`sigmoid` gate, and a per-head recurrent state update.
  - The layer kind comes from the GGUF metadata (the fork marks recurrent layers on load; mirror its logic in `qwen35.cpp` lines 8–30).
  - MTP tensors are loaded but not run.

### 5.4 `bfi-runtime`
- **Tokenizer:** built from GGUF metadata (`tokenizer.ggml.model = "gpt2"`, `tokens`, `merges`, `token_type`, `pre`). It is a byte-level BPE with the Qwen pre-tokenizer regex (`fancy-regex`). Required test: our token ids equal llama.cpp's `llama-tokenize` output on a 2,000-line multilingual and code corpus.
- **Chat templates:** render `tokenizer.chat_template` with `minijinja`, including tool-call and thinking blocks. There is a per-architecture fallback only when the GGUF carries no template.
- **Sequence state:**
  - A paged KV cache for full-attention layers.
  - A fixed-size recurrent state per sequence for delta-net layers. This is why 262K context is affordable on `qwen35`: only 25% of layers grow with the context.
  - `StateLayout` (from the model) tells the runtime what to allocate.
- **Sampler:** greedy, temperature, top-k, top-p, min-p, repetition/presence penalty and seed. JSON-schema/grammar-constrained decoding is M4 (needed for reliable tool calls).
- **Scheduler:** M3 is a single sequence. Sequential `generate_batch` (reuse one loaded Engine; no concurrent token mixing) and prompt-prefix caching landed 2026-10-12 (`e4c07ee`). Concurrent decode mixing / paged attention remains.
- **Model store:** `~/.builderforce/models/` (shared with Synapse). The layout follows the HF cache (`<repo>/<revision>/<file>`), plus `aliases.toml`, which ships in the repo and can be overridden by the user. Example: `bonsai2:27b = "prism-ml/Ternary-Bonsai-2-27B-gguf:PTQ1_0"`.

### 5.5 `bfi-server`
- Uses `axum` with `tokio`. It needs SSE and concurrent streams, which tiny_http can't do well. This is the only async crate.
- **OpenAI:**
  - `GET /v1/models`
  - `POST /v1/chat/completions` (stream and non-stream, `tools`, `tool_choice`, `response_format`)
  - `POST /v1/completions`
  - `POST /v1/embeddings` (M4; it returns 501 until then)
- **Anthropic:** `POST /v1/messages`, streaming event types included, with `tools` and `tool_use`/`tool_result` blocks.
- **Ollama:**
  - `GET /api/version`, `GET /api/tags`, `GET /api/ps`
  - `POST /api/show`, `POST /api/pull` (NDJSON progress `{status,total,completed}`, the exact shape `bf-local` parses today), `DELETE /api/delete`, `POST /api/chat`, `POST /api/generate`
- `GET /health` is unauthenticated.
- **Bind and auth:**
  - Binds to `127.0.0.1` only. The default port is 11434; if that port is taken (Ollama running), it uses the next free port. `--port` and `BFI_HOST` override this.
  - The chosen address is written to `~/.builderforce/bfi.json` (`{port, pid, version, token}`, user-only permissions, the same scheme as `bf-context/src/discovery.rs`).
  - Requests other than `/health` need `Authorization: Bearer <token>` when started with `--auth`. Synapse always passes `--auth`.

### 5.6 `bfi` CLI
| Command | Behaviour |
|---|---|
| `bfi pull <repo[:quant] \| alias>` | Resumable ranged download from HF with sha256 verification. Supports `HF_TOKEN` for gated repos. |
| `bfi run <model> [prompt]` | Interactive chat in the terminal. |
| `bfi serve [--port --auth --models-dir]` | The server (§5.5). |
| `bfi ps` | Loaded models, device, VRAM/RAM use, tok/s of the last request. |
| `bfi fit <model> [--ctx N] [--json]` | Exact memory need vs this machine (§5.7). Works **before** download by reading only the GGUF header with an HTTP range request. |
| `bfi rm`, `bfi ls` | Store management. |
| `bfi bench <model> [--pp 512 --tg 128]` | Same metric names as `llama-bench`, so the numbers compare directly. |
| `bfi import-ollama` | Links models already in Ollama's blob store (manifests → GGUF blobs) into our store, so switching re-downloads nothing. |

### 5.7 `bfi fit` (replaces `hub.rs` sizing)
`need = Σ tensor bytes (exact, from the header) + KV(ctx) + recurrent_state + activations(batch) + runtime overhead (measured per backend, kept as data)`, where:
- `KV(ctx) = full_attn_layers × 2 × n_kv_heads × head_dim × ctx × kv_dtype_bytes`
- `recurrent_state` = the delta-net layers × the per-head state size

The available budget is VRAM when a GPU backend is chosen (from CubeCL device info), or system RAM × 0.75 when on CPU. The `--json` output is `{ fits, needBytes, budgetBytes, device, maxCtxThatFits, breakdown{weights,kv,state,activations,overhead} }`. `maxCtxThatFits` lets the Models view offer the largest context that fits instead of refusing the model.

## 6. Synapse integration (`app/`, `crates/bf-local`)

1. **Ship `bfi` with Synapse** as a Tauri sidecar (`bundle.externalBin`). `release.yml` builds `bfi` for each target before `tauri-action`. The GPU features per target:
   - Windows: `vulkan,cuda`
   - macOS: `metal`
   - Linux: `vulkan,cuda,rocm`
2. **Replace `bf-local/src/ollama.rs` with `bf-local/src/runtime.rs`,** the supervisor:
   - It spawns `bfi serve --auth --models-dir ~/.builderforce/models`, reads `bfi.json` and health-checks the server.
   - It restarts with backoff on crash, at most 5 times in 10 minutes before showing an error. It stops the child when Synapse quits.
   - It exposes the same `installed / pull / delete / version` functions over bfi's `/api/*`. Callers in `app/src-tauri` change only the module name.
   - Delete `ollama.rs` and `DOWNLOAD_URL`, plus every "Install Ollama" string in `app/ui` (all 5 locales).
3. **Fold `hub.rs` sizing into `bfi fit`.** `hub.rs` keeps only the curated catalog as data (purpose, repo, quant tags, alias). `Quant::bits()` and the bits-per-weight estimator are deleted. A recommendation becomes "the highest-quality variant whose `fit.fits` is true".
4. **`api.rs` (port 11435)** keeps its contract (OpenAI + Anthropic, Synapse's key from `bf-vault`) and proxies to bfi instead of Ollama. `chat.rs` is unchanged apart from its error text ("is Ollama running?" becomes the supervisor's state).
5. **Starter model:** the curated default for machines that fit it is `bonsai2:27b` (PTQ1_0). Below that budget, the default is the largest `qwen3` that `bfi fit` approves.
6. **Discovery for other tools:** extend `bf-context`'s `Discovery` (`~/.builderforce/desktop.json`) with an optional field:
   `models: { port: u16, token: String, version: String }`, which is the 11435 endpoint and its key.
   - The file is already user-only, so it has the same trust level as the context token.
   - Written only when the endpoint is enabled. Old readers ignore the field.

## 7. VS Code extension (Builderforce.ai `clients/vscode`, `packages/brain-ui`)

1. **Discovery:**
   - `desktopContext.ts` → `liveDesktop()` already reads `desktop.json`. Add `desktopModelsEndpoint(): Promise<{ baseUrl, token } | null>`, which reads the new `models` field and probes `/health`.
   - Add `"synapse"` to `LocalProviderId` / `LOCAL_PROVIDER_IDS` in `localModels.ts`. Its endpoint comes from discovery, not settings, so there is nothing for the user to configure.
   - `localTransport` sends the token as a Bearer header. This is the only change to the transport.
2. **Model tab** (brain-ui `PromptOptionsMenu` → Model tab):
   - When Synapse is live, list its installed models (`/v1/models`) under a "This machine (Synapse)" group.
   - List the curated catalog from Synapse (new `GET /v1/synapse/catalog` on 11435, returning catalog plus `fit` per entry) with an **Install** action. Install calls `POST /api/pull` through 11435 and shows the NDJSON progress inline. When the pull finishes, the model is selectable.
   - The fixed-list rule applies: these are data rows, not a model call.
   - All new strings go through brain-ui's existing i18n catalogs in all five locales. Styling uses theme tokens only (both themes, narrow widths).
3. When Synapse isn't running, the group doesn't render (the component decides its own visibility). The existing Ollama provider keeps working against any Ollama-compatible server, including a standalone `bfi serve`.

## 8. Milestones and exit criteria

| # | Milestone | Exit criteria (all must hold) |
|---|---|---|
| **M1** | **Ternary kernel benchmark** — the go/no-go | (a) The CubeCL fused GEMV for PQ2_0 and PTQ1_0 matches the CPU reference (max abs err ≤ 1e-3 relative). (b) `bfi bench-kernel` measures the GEMV on Bonsai-27B-shaped matrices: hidden 5120, and every projection shape read from the real GGUF header. (c) It projects decode tok/s as Σ layer GEMV time + measured attention/norm overhead. (d) On the **same machine**, PrismML's `llama-bench -m Ternary-Bonsai-2-27B-PQ2_0.gguf -p 0 -n 128` is run. **Go = projected ≥ 70% of the fork's tg128 on Metal and on Vulkan/CUDA.** The M5 Max reference figure is ~47 tok/s, so ≥ 33 tok/s there. A no-go triggers the fallback in §10. |
| M2 | GGUF + CPU reference, `qwen3` | Golden dequant tests (§3.1) pass. On Qwen3-0.6B Q8_0 and Q4_K_M, top-1 tokens match llama.cpp's for 256 greedy tokens on 20 prompts, and logits agree to within 1e-2 cosine distance. The tokenizer test passes. |
| M3 | `qwen35` + GPU end to end | Bonsai 2 27B (PQ2_0 and PTQ1_0) runs with greedy output identical to the fork's for 128 tokens on 10 prompts. `bfi bench` tg128 is at least the M1 projection on Metal and on Vulkan or CUDA. A 32K context runs in budget. |
| M4 | Server + CLI complete | Every §5.5 endpoint has a contract test. Tool calls round-trip on all three dialects. Continuous batching works with 4 concurrent streams. `bfi fit` is within ±5% of the measured peak RSS/VRAM on 3 models. `import-ollama` works. |
| M5 | Synapse cut-over | The sidecar ships, `ollama.rs` and the estimator are deleted, and the supervisor's restart path is tested. The Models view install → chat flow works on Windows (Vulkan), macOS (Metal) and Linux (CPU). `desktop.json` carries `models`. |
| M6 | VSIX | Discovery plus Model-tab install, tested against a live Synapse. Unit tests cover the discovery parser and the catalog view model. Released as a VSIX version bump. |
| M7 | Release | `bfi` is published standalone (GitHub release binaries, plus `cargo install bfi`). The Synapse release notes and a Builderforce.ai blog post are written by the method template. Builderforce.ai ROADMAP's "Local LLM execution" row moves to DONE.md. |

## 9. Testing and CI

- **Standard runners:** CPU and Vulkan via lavapipe/SwiftShader on ubuntu-22.04, and Metal on `macos-latest` (Apple silicon), including the golden, tokenizer and contract tests. Small models only (Qwen3-0.6B, cached).
- **CUDA / ROCm:** these need self-hosted or paid GPU runners. **Blocker owned by the repo owner:** provide at least one NVIDIA and one AMD runner, or accept manual pre-release runs (a documented checklist in `docs/gpu-checklist.md`).
- **Performance regression:** `bfi bench` results are stored per release in `bench/<version>.json`. A decode regression above 5% on the same hardware blocks the release.

## 10. Risks and fallbacks

| Risk | Mitigation |
|---|---|
| CubeCL kernels miss the 70% bar (M1 no-go) | **Fallback A:** hand-written backends for the hot GEMV only: Metal (MSL via `metal-rs`) and CUDA (PTX via `cudarc`). CubeCL stays for everything else and for Vulkan. **Fallback B** (decision by the owner): fork mistral.rs/Candle for Mac and NVIDIA and accept CPU on AMD/Intel Windows. |
| CubeCL API churn | Pin an exact version and wrap it behind `bfi-core::Backend`, so an upgrade is contained to one crate. |
| `qwen35` drift (the "rotation" rebuild the HF card mentions) | Treat the fork's `qwen35.cpp` at a pinned commit as the spec. Record the commit SHA in `bfi-models/src/qwen35/SPEC.md`. |
| Tokenizer edge cases | Corpus parity test (§5.4) runs in CI. |
| Port 11434 collision with Ollama | Next-free-port, written to `bfi.json`. Synapse never assumes 11434. |
| Licence hygiene | Ported files carry the MIT header of "The ggml authors". `THIRD-PARTY-NOTICES.md` lists the fork's commit. The repo is dual MIT/Apache-2.0 (change `[workspace.package] license` to `MIT OR Apache-2.0` and add `LICENSE-APACHE`). |
| GPU driver crash | It kills only the `bfi` child. The supervisor restarts it, and Synapse keeps running. |

## 11. Blockers that need the owner

1. **The M1 reference run on an M5 Max** (or the closest Apple-silicon machine available, with the figure scaled against the HF card's table). No such machine is in this environment.
2. **GPU CI runners** for CUDA/ROCm (§9).
3. **Signing secrets** in both repos (§2). Unsigned sidecars trip SmartScreen/Gatekeeper more often than the app alone.

## 12. Definition of done

- Synapse installs, runs and serves local models with **no Ollama on the machine**, on Windows, macOS and Linux.
- Bonsai 2 27B runs on Metal and on Vulkan/CUDA at or above the M1 bar.
- Every client that worked against Ollama works against `bfi` unchanged.
- The VS Code extension finds Synapse's models and installs one from the Model tab.
- `ollama.rs` and the bits-per-weight estimator are deleted.
- ROADMAP's row has moved to DONE.md.
