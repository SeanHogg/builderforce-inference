# bfi-web

bfi's own engine in the browser. Runs PrismML's 1-bit **Bonsai** (`qwen3`
architecture, GGUF `Q1_0`) on **WebGPU** using the same crates as the native
`bfi` binary (`bfi-gguf`, `bfi-core`, `bfi-models`, `bfi-runtime`), exposed as
the `BfiEngine` class from Builderforce spec 34 §3.1.

Published to npm as **`@seanhogg/bfi-web`** (see `package.json`; it ships the
`pkg/` that `wasm-pack` writes).

## Build

```sh
# once: rustup target add wasm32-unknown-unknown && cargo install wasm-pack
wasm-pack build crates/bfi-web --target web --release --out-dir pkg
```

Output lands in `crates/bfi-web/pkg/` (`bfi_web.js`, `bfi_web_bg.wasm`,
`bfi_web.d.ts`). Type-check only:

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
if (!s.supported) throw new Error(s.reason);

const engine = await BfiEngine.create();    // rejects "webgpu_unavailable: …"
const res = await fetch(".../Bonsai-1.7B-Q1_0.gguf");
const info = await engine.load(res.body!, Number(res.headers.get("content-length")),
  (loaded, total) => console.log(loaded / total));
// info = { arch: "qwen3", contextTokens, vocab, params }

const ac = new AbortController();
const final = await engine.chat(JSON.stringify({
  messages: [{ role: "user", content: "Hi" }],
  stream: true, max_tokens: 256, temperature: 0.7,
}), (chunkJson) => console.log(JSON.parse(chunkJson)), ac.signal);
// final = chat.completion JSON with usage + finish_reason

engine.unload();
```

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
| `webgpu_unavailable:` | no `navigator.gpu`, no adapter |
| `unsupported_type:` | a tensor type the WebGPU engine cannot run (names the tensor); WebGPU runs `Q1_0`, `Q8_0`, `F16`, `F32` matrices |
| `context_exceeded:` | the prompt does not fit the loaded context |
| `aborted` | `signal` fired, or `unload()` during `chat` |
| `invalid_request:` / `not_loaded:` / `busy:` / `load_failed:` / `engine_error:` | malformed request, chat before load, concurrent call, bad/truncated GGUF or architecture other than `qwen3`, other engine failures |

### Memory

Weights stay packed on the GPU, one buffer per tensor (Bonsai 8B ≈ 1.15 GB,
1.7B ≈ 0.25 GB). The f32 KV cache gets a 1 GiB budget and is also bounded by
the adapter's `maxStorageBufferBindingSize`; `contextTokens` reports what was
allocated (Bonsai 8B ≈ 3.5K tokens, 1.7B ≈ 4.5K). The host streams the file;
bfi-web holds at most the header plus one tensor in wasm memory while loading.
