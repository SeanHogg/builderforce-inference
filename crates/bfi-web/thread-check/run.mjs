#!/usr/bin/env node
/**
 * The THREADED artifact's proof, in a real browser: serve `pkg-threads/` with the
 * cross-origin-isolation headers wasm threads need (COOP same-origin + COEP
 * require-corp), open it in headless Chrome, and inside a dedicated worker — where hosts
 * run bfi — start the rayon pool and run `threadCheck`: a Q1_0 matmul on the pool,
 * bit-compared with the same matmul on one thread, plus how many pool threads took work.
 * The page POSTs the result back; this prints it and exits non-zero unless it holds.
 *
 *   node crates/bfi-web/build-threads.mjs && node crates/bfi-web/thread-check/run.mjs
 *
 * `CHROME` overrides the browser path. `THREADS` caps the pool (default: all cores).
 */
import { spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const crateDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const pkgDir = join(crateDir, 'pkg-threads');
if (!existsSync(join(pkgDir, 'bfi_web.js'))) {
  console.error('pkg-threads/ is missing — run `node crates/bfi-web/build-threads.mjs` first.');
  process.exit(2);
}

const PAGE = `<!doctype html><meta charset="utf-8"><script type="module">
const report = (body) => fetch('/result', { method: 'POST', body: JSON.stringify(body) });
const worker = new Worker('/worker.js', { type: 'module' });
worker.onmessage = (e) => report(e.data);
worker.onerror = (e) => report({ error: String(e.message ?? e) });
worker.postMessage({ threads: ${Number(process.env.THREADS) || 0} });
</script>`;

const WORKER = `
self.onmessage = async (e) => {
  try {
    const mod = await import('/pkg/bfi_web.js');
    await mod.default({ module_or_path: '/pkg/bfi_web_bg.wasm' });
    const n = e.data.threads || navigator.hardwareConcurrency || 4;
    await mod.initThreads(n);
    const check = mod.threadCheck(2048);
    const support = await mod.BfiEngine.support();
    postMessage({ crossOriginIsolated: self.crossOriginIsolated, hardwareConcurrency: navigator.hardwareConcurrency, check, cpu: support.backends.cpu });
  } catch (err) {
    postMessage({ error: String(err && err.stack || err) });
  }
};`;

const TYPES = { '.js': 'text/javascript', '.wasm': 'application/wasm', '.html': 'text/html' };
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Resource-Policy': 'same-origin',
};

let chrome;
const finish = (code, text) => {
  console.log(text);
  chrome?.kill();
  server.close();
  process.exit(code);
};

const server = createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost');
  if (req.method === 'POST' && url.pathname === '/result') {
    let body = '';
    req.on('data', (c) => (body += c));
    req.on('end', () => {
      res.end('ok');
      const result = JSON.parse(body);
      const ok = !result.error && result.check?.identical === true && result.check?.threadsUsed > 1 && result.cpu?.threads > 1;
      finish(ok ? 0 : 1, JSON.stringify(result, null, 2));
    });
    return;
  }
  let file;
  let type;
  if (url.pathname === '/') [file, type] = [null, TYPES['.html']];
  else if (url.pathname === '/worker.js') [file, type] = [null, TYPES['.js']];
  else if (url.pathname.startsWith('/pkg/')) {
    file = normalize(join(pkgDir, url.pathname.slice('/pkg/'.length)));
    if (!file.startsWith(pkgDir) || !existsSync(file)) {
      res.writeHead(404, isolation).end();
      return;
    }
    type = TYPES[extname(file)] ?? 'application/octet-stream';
  } else {
    res.writeHead(404, isolation).end();
    return;
  }
  res.writeHead(200, { ...isolation, 'Content-Type': type });
  res.end(file ? readFileSync(file) : url.pathname === '/' ? PAGE : WORKER);
});

server.listen(0, '127.0.0.1', () => {
  const { port } = server.address();
  const candidates = [
    process.env.CHROME,
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    '/usr/bin/google-chrome',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ].filter(Boolean);
  const browser = candidates.find((p) => existsSync(p));
  if (!browser) finish(2, 'no Chrome found — set CHROME');
  chrome = spawn(browser, [
    '--headless=new',
    '--disable-gpu',
    '--no-first-run',
    `--user-data-dir=${join(tmpdir(), `bfi-thread-check-${port}`)}`,
    `http://127.0.0.1:${port}/`,
  ], { stdio: 'ignore' });
  setTimeout(() => finish(3, 'timed out after 120 s'), 120_000);
});
