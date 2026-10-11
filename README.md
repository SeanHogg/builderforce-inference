# builderforce-inference

**Synapse** — the Builderforce desktop app and local AI runtime. Run LLMs on your own machine, keep a private Evermind memory, host MCP connectors, and hand work to agents.

> Status: Synapse currently drives an external Ollama install for local models. The native, pure-Rust engine that replaces it (`bfi`) is specified in [docs/PRD-native-engine.md](docs/PRD-native-engine.md) and is being implemented as the `bfi*` crates below.

## Layout

| Path | What |
|---|---|
| `app/` | Synapse desktop app (Tauri: `src-tauri/`, `ui/`, `console/`) |
| `crates/bfi-gguf` | GGUF v2/v3 reader and GGML type registry (including PrismML `PQ2_0` / `PTQ1_0`) |
| `crates/bfi-core` | Tensor + Backend trait; CPU reference kernels |
| `crates/bfi-models` | Architecture registry (`qwen3` first, then `qwen35`) |
| `crates/bfi-runtime` | Tokenizer, sampler, scheduler, model store, `bfi fit` |
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

History was extracted from `SeanHogg/Builderforce.ai` (`desktop/`).

## License

MIT
