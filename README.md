# builderforce-inference

**Synapse** — the Builderforce desktop app and local AI runtime. Run LLMs on your own machine, keep a private Evermind memory, host MCP connectors, and hand work to agents.

> Status: Synapse currently drives an external Ollama install for local models. The roadmap is a native, pure-Rust inference engine in this repo so no Ollama install is required.

## Layout

| Path | What |
|---|---|
| `app/` | Synapse desktop app (Tauri: `src-tauri/`, `ui/`, `console/`) |
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
