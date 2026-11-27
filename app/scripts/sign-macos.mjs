#!/usr/bin/env node
// Signs a packaged macOS .app for notarization. `codesign --deep` does NOT sign loose Mach-O files under
// Contents/Resources (`*.node`, `rg`, native code inside bundled extensions), so they are found by magic
// bytes and signed inside-out: every Mach-O file (deepest first), then nested bundles (helper .apps and
// frameworks, deepest first), then the main bundle, then `codesign --verify --deep --strict`.
//   node app/scripts/sign-macos.mjs --app <Synapse.app> --identity <id> --entitlements <plist>
import { spawnSync } from 'node:child_process';
import { closeSync, openSync, readSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// 32/64-bit Mach-O in both byte orders, plus the fat (universal) header.
const MACHO_MAGICS = new Set([0xfeedface, 0xfeedfacf, 0xcefaedfe, 0xcffaedfe]);
// Fat headers; 0xcafebabe is also the Java .class magic, told apart by the next big-endian word
// (nfat_arch is small; a .class file has its minor/major version there, 45 or more).
const FAT_MAGICS = new Set([0xcafebabe, 0xcafebabf]);
const MAX_FAT_ARCHS = 30;

/** True when the file starts with a Mach-O / fat magic number. Unreadable or short files are not Mach-O. */
export function isMachO(file) {
  let fd;
  try {
    fd = openSync(file, 'r');
    const buf = Buffer.alloc(8);
    const n = readSync(fd, buf, 0, 8, 0);
    if (n < 4) return false;
    const magic = buf.readUInt32BE(0);
    if (MACHO_MAGICS.has(magic)) return true;
    if (!FAT_MAGICS.has(magic) || n < 8) return false;
    const nfatArch = buf.readUInt32BE(4);
    return nfatArch > 0 && nfatArch < MAX_FAT_ARCHS;
  } catch {
    return false;
  } finally {
    if (fd !== undefined) closeSync(fd);
  }
}

/** Walks `root` (never following symlinks) and returns every regular file that is Mach-O. */
export function findMachO(root) {
  const found = [];
  const walk = (dir) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const full = join(dir, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.isFile() && isMachO(full)) found.push(full);
    }
  };
  walk(root);
  return found;
}

/** Nested bundles (`*.app`, `*.framework`) strictly inside `root`, never following symlinks. */
export function findBundles(root) {
  const found = [];
  const walk = (dir) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const full = join(dir, entry.name);
      if (/\.(app|framework)$/.test(entry.name)) found.push(full);
      walk(full);
    }
  };
  walk(root);
  return found;
}

const depth = (p) => p.split(/[\\/]+/).filter(Boolean).length;

/** Inside-out: deepest paths first (ties broken by name so the order is stable). */
export function signingOrder(paths) {
  return [...paths].sort((a, b) => depth(b) - depth(a) || (a < b ? -1 : a > b ? 1 : 0));
}

/** The full ordered list of paths to sign: Mach-O files, then nested bundles, then the main bundle last. */
export function planSigning(app, machO, bundles) {
  const main = resolve(app);
  return [...signingOrder(machO), ...signingOrder(bundles.filter((b) => resolve(b) !== main)), app];
}

function codesign(args) {
  console.log(`> codesign ${args.join(' ')}`);
  const r = spawnSync('codesign', args, { stdio: 'inherit' });
  if (r.status !== 0) {
    console.error(`sign-macos: codesign failed (exit ${r.status ?? r.error?.message})`);
    process.exit(1);
  }
}

function main() {
  const argv = process.argv.slice(2);
  const opts = {};
  for (let i = 0; i < argv.length; i += 2) opts[argv[i].replace(/^--/, '')] = argv[i + 1];
  for (const k of ['app', 'identity', 'entitlements']) {
    if (!opts[k]) { console.error(`sign-macos: missing --${k}`); process.exit(1); }
  }
  const { app, identity, entitlements } = opts;
  const plan = planSigning(app, findMachO(app), findBundles(app));
  console.log(`sign-macos: signing ${plan.length} items`);
  for (const target of plan) {
    codesign(['--force', '--options', 'runtime', '--timestamp', '--entitlements', entitlements, '--sign', identity, target]);
  }
  codesign(['--verify', '--deep', '--strict', '--verbose=2', app]);
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) main();
