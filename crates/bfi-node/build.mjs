// Build the addon for THIS machine (or `--target <rust triple>` in CI) and place it as
// `bfi_node.<platform>-<arch>.node` beside index.js — the name `loadBfiNode` resolves —
// with `.bfi-build-stamp.<platform>-<arch>.json` beside it: the commit it was built from
// and whether the crates were dirty, which Builderforce's `vendor-bfi-node.mjs --from`
// records (the `bfi-node` workflow uploads both).
//
//   node crates/bfi-node/build.mjs                 # this machine, release
//   node crates/bfi-node/build.mjs --target aarch64-apple-darwin
import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..', '..');

/** Rust triple → Node's `process.platform` / `process.arch`. */
const TARGETS = {
  'x86_64-pc-windows-msvc': ['win32', 'x64'],
  'aarch64-pc-windows-msvc': ['win32', 'arm64'],
  'x86_64-apple-darwin': ['darwin', 'x64'],
  'aarch64-apple-darwin': ['darwin', 'arm64'],
  'x86_64-unknown-linux-gnu': ['linux', 'x64'],
  'aarch64-unknown-linux-gnu': ['linux', 'arm64'],
};

const at = process.argv.indexOf('--target');
const target = at >= 0 ? process.argv[at + 1] : undefined;
const [platform, arch] = target ? TARGETS[target] ?? [] : [process.platform, process.arch];
if (!platform) throw new Error(`unknown target ${target}; known: ${Object.keys(TARGETS).join(', ')}`);

execFileSync('cargo', ['build', '-p', 'bfi-node', '--release', ...(target ? ['--target', target] : [])], {
  cwd: root,
  stdio: 'inherit',
});

const out = join(root, 'target', ...(target ? [target] : []), 'release');
const built = { win32: 'bfi_node.dll', darwin: 'libbfi_node.dylib', linux: 'libbfi_node.so' }[platform];
const from = join(out, built);
if (!existsSync(from)) throw new Error(`cargo produced no ${from}`);
const to = join(here, `bfi_node.${platform}-${arch}.node`);
copyFileSync(from, to);

const git = (...args) => execFileSync('git', ['-C', root, ...args], { encoding: 'utf8' }).trim();
const stamp = { commit: git('rev-parse', 'HEAD'), dirty: git('status', '--porcelain', '--', 'crates', 'Cargo.toml', 'Cargo.lock') !== '' };
writeFileSync(join(here, `.bfi-build-stamp.${platform}-${arch}.json`), `${JSON.stringify(stamp, null, 2)}
`);
console.log(`bfi-node: ${to} (${stamp.commit.slice(0, 7)}${stamp.dirty ? ', dirty' : ''})`);
