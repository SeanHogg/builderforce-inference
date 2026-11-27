#!/usr/bin/env node
// Builds Synapse for one platform from the pinned Code-OSS tag. See app/README.md.
//   node app/scripts/build.mjs --platform win32 --arch x64 --version 2026.11.27 \
//        --builderforce-vsix <path> --spawn-vsix <path> [--stage fetch|package|installer|all]
import { spawnSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { builtinExtensionsPath, gulpTask, installerName, mergeProduct } from './lib.mjs';
import { makeDmg } from './make-dmg.mjs';

// Stages: `fetch` (clone upstream, so CI can read its .nvmrc), `package` (up to the packaged app with the
// built-in extensions in it), `installer` (upstream installer tasks / the .dmg), `all` (the default).
// CI signs the packaged app between `package` and `installer`.
const STAGES = ['fetch', 'package', 'installer', 'all'];

const APP = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const BUILD = resolve(process.env.SYNAPSE_BUILD_DIR || join(APP, '.build'));
const SRC = join(BUILD, 'vscode');
const OUT = join(BUILD, 'out');
const UPSTREAM_REPO = 'https://github.com/microsoft/vscode';

const fail = (msg) => { console.error(`build: ${msg}`); process.exit(1); };
const readJson = (p) => JSON.parse(readFileSync(p, 'utf8'));

function run(cmd, args, cwd = SRC, env = {}) {
  console.log(`> ${cmd} ${args.join(' ')}`);
  const r = spawnSync(cmd, args, { cwd, stdio: 'inherit', shell: process.platform === 'win32', env: { ...process.env, ...env } });
  if (r.status !== 0) fail(`"${cmd} ${args.join(' ')}" failed (exit ${r.status ?? r.error?.message})`);
}

function parseArgs(argv) {
  const opts = {};
  for (let i = 0; i < argv.length; i += 2) opts[argv[i].replace(/^--/, '')] = argv[i + 1];
  opts.stage ??= 'all';
  if (!STAGES.includes(opts.stage)) fail(`--stage must be one of ${STAGES.join(', ')}`);
  const needsVsix = opts.stage === 'package' || opts.stage === 'all';
  for (const k of ['platform', 'arch', 'version', ...(needsVsix ? ['builderforce-vsix', 'spawn-vsix'] : [])]) {
    if (!opts[k]) fail(`missing --${k}`);
  }
  if (needsVsix) {
    for (const k of ['builderforce-vsix', 'spawn-vsix']) {
      if (!existsSync(opts[k])) fail(`--${k} not found: ${opts[k]}`);
    }
  }
  return opts;
}

/** Step 1: shallow clone at the pinned tag, or reuse a checkout already at it. */
function fetchUpstream(tag) {
  if (existsSync(join(SRC, '.git'))) {
    const head = spawnSync('git', ['describe', '--tags', '--exact-match'], { cwd: SRC, encoding: 'utf8' }).stdout?.trim();
    if (head === tag) {
      console.log(`reusing checkout at ${tag}`);
      run('git', ['config', 'core.autocrlf', 'false']);
      run('git', ['checkout', '--', '.']);
      return;
    }
    rmSync(SRC, { recursive: true, force: true });
  }
  mkdirSync(BUILD, { recursive: true });
  run('git', ['clone', '--config', 'core.autocrlf=false', '--depth', '1', '--branch', tag, UPSTREAM_REPO, SRC], BUILD);
}

/** Step 2: Synapse's product.json merged over upstream's. */
function writeProduct() {
  const upstream = readJson(join(SRC, 'product.json'));
  writeFileSync(join(SRC, 'product.json'), JSON.stringify(mergeProduct(upstream, readJson(join(APP, 'product.json'))), null, 2));
}

/** Step 3: patches in filename order. */
function applyPatches() {
  const dir = join(APP, 'patches');
  for (const p of readdirSync(dir).filter((f) => f.endsWith('.patch')).sort()) run('git', ['apply', '--whitespace=nowarn', join(dir, p)]);
}

/** Step 4: Synapse icons over upstream's (standard resources paths). */
function copyIcons(platform) {
  const map = {
    win32: ['icon.ico', 'resources/win32/code.ico'],
    darwin: ['icon.icns', 'resources/darwin/code.icns'],
    linux: ['icon.png', 'resources/linux/code.png'],
  };
  const [from, to] = map[platform];
  cpSync(join(APP, 'icons', from), join(SRC, to));
}

/**
 * Unzip a .zip/.vsix. `unzip` where present (Linux/macOS); on Windows the bsdtar that ships with it
 * (System32/tar.exe, which reads zip), NOT whatever `tar` Git Bash puts first on PATH (GNU tar cannot read zip).
 */
function unzip(file, dest) {
  mkdirSync(dest, { recursive: true });
  if (process.platform === 'win32') {
    const tar = join(process.env.SystemRoot ?? 'C:/Windows', 'System32', 'tar.exe');
    run(`"${tar}"`, ['-xf', file, '-C', dest], APP);
    return;
  }
  const hasUnzip = spawnSync('unzip', ['-v'], { stdio: 'ignore' }).status === 0;
  run(hasUnzip ? 'unzip' : 'tar', hasUnzip ? ['-q', file, '-d', dest] : ['-xf', file, '-C', dest], APP);
}

/**
 * Step 5 (after the `-min` task): a VSIX keeps its files under `extension/`; they become a built-in
 * extension INSIDE the packaged app. Never into the source `extensions/` folder gulp compiles.
 */
function installVsix(vsix, extensionId, { platform, arch }) {
  const tmp = mkdtempSync(join(tmpdir(), 'synapse-vsix-'));
  try {
    unzip(resolve(vsix), tmp);
    const manifest = readJson(join(tmp, 'extension', 'package.json'));
    const actualId = `${manifest.publisher}.${manifest.name}`;
    if (actualId !== extensionId) throw new Error(`Expected ${extensionId}, received ${actualId}`);
    const extensionsDir = join(BUILD, ...builtinExtensionsPath(platform, arch));
    if (!existsSync(extensionsDir)) throw new Error(`Packaged app has no built-in extensions folder: ${extensionsDir}`);
    const dest = join(extensionsDir, extensionId);
    rmSync(dest, { recursive: true, force: true });
    cpSync(join(tmp, 'extension'), dest, { recursive: true });
    console.log(`installed ${extensionId}@${manifest.version} into ${dest}`);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
}

const GULP_ENV = { NODE_OPTIONS: '--max-old-space-size=8192' };

/**
 * Step 6: upstream's own build up to the packaged app (microsoft/vscode build pipelines use npm; `npm ci`).
 * ASSUMPTION: still npm at the pinned tag. Needs the Node version in upstream's `.nvmrc`.
 */
function buildApp({ platform, arch }) {
  run('npm', ['ci'], SRC, GULP_ENV);
  for (const task of gulpTask(platform, arch).app) run('npx', ['gulp', task], SRC, GULP_ENV);
}

/** Step 7: upstream's installer tasks, in order (after the packaged app is signed, where signing applies). */
function buildInstaller({ platform, arch }) {
  if (!existsSync(join(SRC, 'node_modules'))) fail('no upstream checkout with dependencies: run --stage package first');
  for (const task of gulpTask(platform, arch).installer) run('npx', ['gulp', task], SRC, GULP_ENV);
}

const find = (dir, pred) => (existsSync(dir) ? readdirSync(dir).find(pred) : undefined);

/** Step 8: rename upstream's outputs to Synapse's installer names in app/.build/out. */
function collectOutputs({ platform, arch, version }) {
  mkdirSync(OUT, { recursive: true });
  const target = `${platform}-${arch}`;
  const outputs = [];
  if (platform === 'win32') {
    outputs.push([join(SRC, '.build', target, 'user-setup', 'VSCodeSetup.exe'), installerName(version, platform, arch, 'setup')]);
  } else if (platform === 'darwin') {
    // The .app was signed between the package and installer stages; the .dmg is signed and notarized after this.
    const dmg = join(OUT, installerName(version, platform, arch, 'dmg'));
    makeDmg(join(BUILD, `VSCode-${target}`), dmg);
    return [dmg];
  } else {
    const debDir = join(SRC, '.build', 'linux', 'deb', arch === 'x64' ? 'amd64' : 'arm64', 'deb');
    const deb = find(debDir, (f) => f.endsWith('.deb'));
    if (!deb) fail(`no .deb found in ${debDir}`);
    outputs.push([join(debDir, deb), installerName(version, platform, arch, 'deb')]);
  }
  return outputs.map(([from, name]) => {
    if (!existsSync(from)) fail(`expected build output missing: ${from}`);
    const to = join(OUT, name);
    renameSync(from, to);
    return to;
  });
}

function main() {
  const opts = parseArgs(process.argv.slice(2));
  const ctx = { platform: opts.platform, arch: opts.arch, version: opts.version };
  gulpTask(ctx.platform, ctx.arch); // fail fast on an unsupported target
  const { stage } = opts;
  if (stage === 'fetch' || stage === 'package' || stage === 'all') fetchUpstream(readJson(join(APP, 'upstream.json')).vscodeTag);
  if (stage === 'package' || stage === 'all') {
    writeProduct();
    applyPatches();
    copyIcons(ctx.platform);
    buildApp(ctx);
    installVsix(opts['builderforce-vsix'], 'builderforce.builderforce-ai', ctx);
    installVsix(opts['spawn-vsix'], 'builderforce.spawn', ctx);
  }
  if (stage === 'installer' || stage === 'all') {
    buildInstaller(ctx);
    for (const f of collectOutputs(ctx)) console.log(`built ${f}`);
  }
}

main();
