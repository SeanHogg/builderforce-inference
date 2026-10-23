# builderforce-inference

> Part of [Builderforce open source](https://builderforce.ai/open-source): the MIT projects Builderforce.ai is built on.

**Synapse** — the Builderforce desktop app and local AI runtime. Run LLMs on your own machine, keep a private Evermind memory, host MCP connectors, and hand work to agents.

Synapse now runs models itself. Ollama is not required. The desktop app supervises `bfi serve`; a separately installed Ollama is not on the path. Keep `crates/bfi` `import-ollama` if you are converting an existing Ollama library.

> Status: Synapse ships and supervises the native `bfi` engine for local models (see [docs/PRD-native-engine.md](docs/PRD-native-engine.md)). CPU sidecar ships. GPU 70% vs llama-bench and live Bonsai 27B identity are not claimed.

## Install

```sh
cargo install --git https://github.com/SeanHogg/builderforce-inference bfi
```

GitHub Release binaries are not signed yet (owner signing secrets).

## Layout

| Path | What |
|---|---|
| `app/` | Synapse desktop app (Tauri: `src-tauri/`, `ui/`, `console/`) |
| `crates/bfi-gguf` | GGUF v2/v3 reader and GGML type registry (including PrismML `PQ2_0` / `PTQ1_0`) |
| `crates/bfi-core` | Tensor + Backend trait; CPU reference kernels |
| `crates/bfi-models` | Architecture registry (`qwen3` first, then `qwen35`) |
| `crates/bfi-runtime` | Tokenizer, sampler, `mix_decode` concurrent token mixing and `generate_batch`, prefix cache, grammar-constrained decode, embeddings, model store, `bfi fit` |
| `crates/bfi-server` | OpenAI / Anthropic / Ollama HTTP API |
| `crates/bfi` | CLI: `pull`, `run`, `serve`, `fit`, `ls`, `rm`, `ps`, `bench` |
| `crates/bf-local` | Local models: catalog, hardware fit, OpenAI/Anthropic-compatible loopback API |
| `crates/bf-context` | Loopback context service used by the Builderforce VS Code extension |
| `crates/bf-index` | Code index (tree-sitter, BM25, local embeddings, reference-ranked repo map) |
| `crates/bf-compress` | LLMLingua-2 token-level prompt compression; code fences, paths and numbers are never touched (ONNX model behind the `compress` feature) |
| `crates/bf-mcp` | MCP client/host for connectors |
| `crates/bf-memory` | Evermind memory client |
| `crates/bf-teach` | Teach-once UI recording for self-directed agents |
| `crates/bf-vault` | Secrets in the OS credential store |
| `crates/bf-cloud` | builderforce.ai client (device sign-in, platform API, LLM gateway) |

## Build

```sh
cargo build --workspace
```

The default build has no ONNX Runtime and downloads no model. LLMLingua-2 compression is opt-in:

```sh
cargo build -p bf-context --features compress
```

### `bfi` engine options

`bfi run`, `bfi serve` and `bfi bench` take the same engine flags. `bfi serve` also reads them from the environment, so the desktop app can set them on the sidecar:

| Flag | Env (`bfi serve`) | What |
|---|---|---|
| `--spec-lookup` | `BFI_SPEC_LOOKUP=1\|true\|on` (`0\|false\|off` turns it off) | Prompt-lookup speculative decoding: draft tokens from n-gram matches in the context and verify them in one forward pass. Off by default. `bfi bench --spec-lookup` also times it against plain decode and reports the acceptance rate. |
| `--draft-len N` | | Tokens drafted per verify pass with `--spec-lookup` (default 4). |
| `--prefix-cache-mb N` | `BFI_PREFIX_CACHE_MB=N` | Prefix-cache budget in MiB: the cached K/V of earlier prompts, reused when a request shares their prefix (default 1024). |

### `bf-context` commands

```sh
bf-context serve                         # run the service (what the desktop app does, minus the UI)
bf-context mcp [--root <dir>]            # MCP stdio server for Claude Code / Cursor (default root: cwd)
bf-context index <dir>                   # index a workspace once and print its status
bf-context search <dir> <query>          # search a workspace and print the hits as JSON
bf-context repo-map --root <dir> [--max-tokens N] [--focus <path>…] [--query <task>…]
                                         # print the reference-ranked repo map
bf-context compress [--rate R] < file    # LLMLingua-2 compress stdin to stdout (`compress` feature)
```

`repo-map` ranks files by how much the rest of the code depends on them (PageRank over the reference graph) and lists each file's most-referenced definitions with their signature lines, within the token budget (default 1500, max 8000). `--focus` paths and the files, file names and symbols named in `--query` personalize the ranking: they rank first and their neighbourhood next. The same options are on `POST /v1/repo-map` (`maxTokens`, `focus`, `query`) and the MCP `repo_map` tool.

`compress` keeps the share `R` of prose tokens (default 0.5) that the LLMLingua-2 classifier marks as load-bearing, in their original order; fenced code, paths, URLs, inline code and numbers are copied verbatim. Token counts go to stderr, so the output pipes cleanly. The same operation is `POST /v1/compress` and the MCP `compress_context` tool, which is listed only when the binary is built with `--features compress`.

The code index parses Rust, TypeScript, JavaScript, Python, Go, Java, C# (`.cs`), Ruby (`.rb`), PHP (`.php`), Kotlin (`.kt`, `.kts`) and Swift (`.swift`) with tree-sitter.

### Golden dequant (PRD §8 M2)

CI `cargo test -p bfi-gguf` does **not** need weights. To sample 8 real blocks of each M2 type (`Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `F16`, `PQ2_0`, `PTQ1_0`) from a Bonsai / qwen3 GGUF:

```sh
BFI_GOLDEN_GGUF=/path/to/model.gguf cargo test -p bfi-gguf --test golden_dequant -- --ignored --nocapture
```

To compare against llama.cpp C-reference dumps (`<TYPE>-<i>.f32`, little-endian f32, one block each), set `BFI_GOLDEN_REF=/path/to/dumps`. Bit-exact by default; `BFI_GOLDEN_ULP=1` allows ≤1 ULP.

Tokenizer (2,000-line mixed corpus) and greedy 20-prompt identity vs llama.cpp:

```sh
BFI_GOLDEN_GGUF=/path/to/qwen3.gguf cargo test -p bfi-runtime --test golden_tokenize -- --ignored --nocapture
BFI_GOLDEN_GGUF=/path/to/qwen3-0.6b.gguf cargo test -p bfi-runtime --test golden_greedy -- --ignored --nocapture
```

Optional: `BFI_LLAMA_TOKENIZE` (llama-tokenize binary), `BFI_LLAMA_IDS` / `BFI_LLAMA_LOGITS` dump dirs.

History was extracted from `SeanHogg/Builderforce.ai` (`desktop/`).

## License

MIT
