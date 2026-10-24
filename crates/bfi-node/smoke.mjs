// The addon end to end on this machine: support, contextFor, an engine on each backend
// — and, given a GGUF path, a load + one streamed chat turn (with a tool, to prove the
// grammar-constrained tool call comes back structured).
//
//   node crates/bfi-node/smoke.mjs [path/to/Bonsai-1.7B-Q1_0.gguf] [--backend cpu|webgpu|auto]
import { createReadStream, statSync } from 'node:fs';
import { createRequire } from 'node:module';
import { Readable } from 'node:stream';
import { Worker } from 'node:worker_threads';

const require = createRequire(import.meta.url);
const { loadBfiNode } = require('./index.js');
const { BfiEngine } = loadBfiNode();

const args = process.argv.slice(2);
const at = args.indexOf('--backend');
const backend = at >= 0 ? args[at + 1] : 'auto';
const gguf = args.find((a, i) => !a.startsWith('--') && args[i - 1] !== '--backend');

const support = await BfiEngine.support();
console.log('support', JSON.stringify(support));
const ctx = BfiEngine.contextFor({ layers: 28, kvHeads: 8, headDim: 128, trainedContext: 32768, ...support });
const cpuCtx = BfiEngine.contextFor({ layers: 28, kvHeads: 8, headDim: 128, trainedContext: 32768, backend: 'cpu' });
console.log('contextFor', { gpu: ctx, cpu: cpuCtx });
if (!(cpuCtx > 0)) throw new Error('contextFor returned nothing');

const cpu = await BfiEngine.create({ backend: 'cpu' });
console.log('cpu engine', cpu.backend);
cpu.dispose();

// The editor's topology: the addon loaded on the main thread AND in worker after worker
// (a trap starts a fresh one) — each gets the same support and a working engine.
for (const n of [1, 2]) {
  const worker = new Worker(
    `const { parentPort } = require('node:worker_threads');
     const { BfiEngine } = require(${JSON.stringify(require.resolve('./index.js'))}).loadBfiNode();
     (async () => {
       const s = await BfiEngine.support();
       const e = await BfiEngine.create({ backend: 'auto' });
       const backend = e.backend;
       e.dispose();
       parentPort.postMessage({ supported: s.supported, backend });
     })().catch((err) => parentPort.postMessage({ error: String(err) }));`,
    { eval: true },
  );
  const answer = await new Promise((resolve, reject) => worker.once('message', resolve).once('error', reject));
  await worker.terminate();
  if (answer.error || answer.supported !== support.supported) throw new Error(`worker ${n}: ${JSON.stringify(answer)}`);
  console.log(`worker ${n}`, answer);
}

const engine = await BfiEngine.create({ backend });
console.log('engine', engine.backend);
if (!gguf) {
  engine.dispose();
  console.log('smoke ok (no GGUF given: load/chat skipped)');
  process.exit(0);
}

const bytes = statSync(gguf).size;
let last = 0;
const loading = engine.load(Readable.toWeb(createReadStream(gguf)), bytes, (loaded, total) => {
  const pct = Math.floor((loaded / total) * 10);
  if (pct !== last) console.log(`load ${pct * 10}%`), (last = pct);
});
// A second load while one runs is `busy:` — and never touches its own stream.
const second = Readable.toWeb(createReadStream(gguf));
await engine.load(second, bytes).then(
  () => {
    throw new Error('expected busy');
  },
  (e) => {
    if (!/^busy:/.test(e.message) || second.locked) throw new Error(`expected an untouched busy: ${e.message}`);
    console.log('busy ok');
  },
);
await second.cancel();
const info = await loading;
console.log('loaded', JSON.stringify(info));

const request = {
  stream: true,
  max_tokens: 96,
  temperature: 0,
  messages: [{ role: 'user', content: 'List the files in the current folder.' }],
  tools: [
    {
      type: 'function',
      function: {
        name: 'run_command',
        description: 'Run a shell command and return its output',
        parameters: { type: 'object', properties: { command: { type: 'string' } }, required: ['command'] },
      },
    },
  ],
};
let chunks = 0;
const started = Date.now();
const completion = JSON.parse(await engine.chat(JSON.stringify(request), () => void chunks++));
const message = completion.choices?.[0]?.message;
console.log('chat', {
  chunks,
  ms: Date.now() - started,
  finish: completion.choices?.[0]?.finish_reason,
  toolCalls: message?.tool_calls?.map((c) => `${c.function.name}(${c.function.arguments})`),
  text: message?.content?.slice(0, 200),
  usage: completion.usage,
});

// Abort lands: a long turn stopped right away ends `aborted`.
const ac = new AbortController();
const long = engine.chat(JSON.stringify({ ...request, tools: undefined, max_tokens: 2048 }), () => ac.abort(), ac.signal);
await long.then(
  () => {
    throw new Error('expected aborted');
  },
  (e) => {
    if (e.message !== 'aborted') throw e;
    console.log('abort ok');
  },
);
engine.unload();
engine.dispose();
console.log('smoke ok');
