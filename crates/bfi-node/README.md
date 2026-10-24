# bfi-node

bfi's own engine inside a Node process — the editor's extension host. Runs PrismML's
1-bit **Bonsai** (`qwen3`, GGUF `Q1_0`) on **native wgpu** (Vulkan; Metal on macOS) or on
**every CPU core**, exposed as the same `BfiEngine` class as `@seanhogg/bfi-web`
(Builderforce spec 34 §3.1), so a host's `bfi` page engine takes either module.

The engine is `bfi-engine` — the state machine, load, chat and `contextFor` that
`bfi-web` runs in a browser. This crate is its napi face: one engine thread per engine
(the JS thread never decodes), a native support probe (a wgpu adapter, the machine's
RAM) and the pushed-chunk byte source that keeps one GGUF chunk in flight.

## Build

```sh
node crates/bfi-node/build.mjs                         # this machine → bfi_node.<platform>-<arch>.node
node crates/bfi-node/build.mjs --target aarch64-apple-darwin
node crates/bfi-node/smoke.mjs [model.gguf] [--backend webgpu|cpu|auto]
```

`bfi-node` is a workspace member but not a default one: it turns on bfi-core's `webgpu`
feature, which must not unify into the native `bfi` binary in a bare `cargo build`.

## API

```js
const { loadBfiNode } = require('@seanhogg/bfi-node');
const { BfiEngine } = loadBfiNode();            // or loadBfiNode('/path/to/bfi_node.node')

const support = await BfiEngine.support();      // wgpu adapter probe, never a device
const engine = await BfiEngine.create({ backend: 'auto' }); // 'webgpu' = the GPU via wgpu
const info = await engine.load(readableStream, totalBytes, onProgress, { kvBudgetBytes });
const completion = await engine.chat(JSON.stringify(request), (chunkJson) => {}, signal);
engine.unload();
engine.dispose();                               // stop the engine thread
```

Differences from `bfi-web`, all in the device record: the GPU is native wgpu (its real
limits and `shader-f16`); the CPU runs on rayon's pool, plans against half the machine's
RAM, and its default KV budget is a quarter of that share (512 MiB–2 GiB) rather than
wasm's 512 MiB. A panic inside the engine is the native twin of a wasm trap: the call
rejects with a `RuntimeError` and the engine is poisoned — create a new one.
