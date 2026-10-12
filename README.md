# builderforce-inference

**Synapse** — the Builderforce desktop app and local AI runtime. Run LLMs on your own machine, keep a private Evermind memory, host MCP connectors, and hand work to agents.

> Status: Synapse ships and supervises the native `bfi` engine for local models (see [docs/PRD-native-engine.md](docs/PRD-native-engine.md)).

## Layout

| Path | What |
|---|---|
| `app/` | Synapse desktop app (Tauri: `src-tauri/`, `ui/`, `console/`) |
| `crates/bfi-gguf` | GGUF v2/v3 reader and GGML type registry (including PrismML `PQ2_0` / `PTQ1_0`) |
| `crates/bfi-core` | Tensor + Backend trait; CPU reference kernels |
| `crates/bfi-models` | Architecture registry (`qwen3` first, then `qwen35`) |
| `crates/bfi-runtime` | Tokenizer, sampler, sequential `generate_batch` scheduler (one loaded model, no concurrent token mixing), prefix cache, grammar-constrained decode, embeddings, model store, `bfi fit` |
| `crates/bfi-server` | OpenAI / Anthropic / Ollama HTTP API |
| `crates/bfi` | CLI: `pull`, `run`, `serve`, `fit`, `ls`, `rm`, `ps`, `bench` |
| `crates/bf-local` | Local models: catalog, hardware fit, OpenAI/Anthropic-compatible loopback API |
| `crates/bf-context` | Loopback context service used by the Builderforce VS Code extension |
| `crates/bf-index` | Code index (tree-sitter, BM25, local embeddings) |
| `crates/bf-mcp` | MCP client/host for connectors |
| `crates/bf-memory` | Evermind memory client |
| `crates/bf-teach` | Teach-once UI recording for self-directed agents |
| `crates/bf-vault` | Secrets in the OS credential store |
| `crates/bf-cloud` | builderforce.ai client (device sign-in, platform API, LLM gateway) |

## Build

```sh
cargo build --workspace
```

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
