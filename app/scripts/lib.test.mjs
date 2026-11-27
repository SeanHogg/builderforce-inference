import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mergeProduct, installerName, gulpTask, builtinExtensionsPath, expectedInstallers } from './lib.mjs';

// Mirrors the resolver in Builderforce.ai/api/src/application/desktop/synapseInstallers.ts.
const WINDOWS_SUFFIXES = ['_x64-setup.exe', '_x64_en-US.msi'];
const MAC_ARM_SUFFIXES = ['_aarch64.dmg'];
const MAC_INTEL_SUFFIXES = ['_x64.dmg'];
const endsWithAny = (name, suffixes) => suffixes.some((s) => name.endsWith(s));

test('mergeProduct deep-merges objects and replaces arrays', () => {
  const upstream = { nameShort: 'Code', gallery: { serviceUrl: 'a', cacheUrl: 'c' }, list: [1, 2], keep: true };
  const merged = mergeProduct(upstream, { nameShort: 'Synapse', gallery: { serviceUrl: 'b' }, list: [3] });
  assert.deepEqual(merged, { nameShort: 'Synapse', gallery: { serviceUrl: 'b', cacheUrl: 'c' }, list: [3], keep: true });
  assert.equal(upstream.nameShort, 'Code', 'upstream is not mutated');
});

test('mergeProduct lets an override replace an object with a scalar', () => {
  assert.deepEqual(mergeProduct({ a: { b: 1 } }, { a: 'x' }), { a: 'x' });
});

test('mergeProduct: an explicit empty array wins and a null key is deleted', () => {
  const upstream = { builtInExtensions: [{ name: 'ms-vscode.js-debug' }], defaultChatAgent: { extensionId: 'GitHub.copilot' }, gallery: { a: 1, b: 2 } };
  const merged = mergeProduct(upstream, { builtInExtensions: [], defaultChatAgent: null, gallery: { a: null }, absent: null });
  assert.deepEqual(merged, { builtInExtensions: [], gallery: { b: 2 } });
  assert.ok(!('defaultChatAgent' in merged));
  assert.ok(upstream.defaultChatAgent, 'upstream is not mutated');
});

test('installerName produces the resolver-matched names', () => {
  const v = '2026.11.27';
  assert.equal(installerName(v, 'win32', 'x64', 'setup'), `Synapse_${v}_x64-setup.exe`);
  assert.equal(installerName(v, 'darwin', 'arm64', 'dmg'), `Synapse_${v}_aarch64.dmg`);
  assert.equal(installerName(v, 'darwin', 'x64', 'dmg'), `Synapse_${v}_x64.dmg`);
  assert.equal(installerName(v, 'linux', 'x64', 'deb'), `Synapse_${v}_amd64.deb`);
  assert.throws(() => installerName(v, 'linux', 'x64', 'appimage'));
});

test('installerName matches the site resolver suffixes', () => {
  assert.ok(endsWithAny(installerName('1.0.0', 'win32', 'x64', 'setup'), WINDOWS_SUFFIXES));
  assert.ok(endsWithAny(installerName('1.0.0', 'darwin', 'arm64', 'dmg'), MAC_ARM_SUFFIXES));
  assert.ok(endsWithAny(installerName('1.0.0', 'darwin', 'x64', 'dmg'), MAC_INTEL_SUFFIXES));
  // The Intel suffix must not capture the arm64 name, nor the arm suffix the Intel one.
  assert.ok(!endsWithAny(installerName('1.0.0', 'darwin', 'arm64', 'dmg'), MAC_INTEL_SUFFIXES));
  assert.ok(!endsWithAny(installerName('1.0.0', 'darwin', 'x64', 'dmg'), MAC_ARM_SUFFIXES));
});

test('installerName rejects unknown combinations', () => {
  assert.throws(() => installerName('1.0.0', 'win32', 'x64', 'dmg'));
});

test('gulpTask returns the upstream task names, installer tasks in run order', () => {
  assert.deepEqual(gulpTask('win32', 'x64'), {
    app: ['vscode-win32-x64-min', 'vscode-win32-x64-inno-updater'],
    installer: ['vscode-win32-x64-user-setup'],
  });
  assert.deepEqual(gulpTask('darwin', 'arm64'), { app: ['vscode-darwin-arm64-min'], installer: [] });
  assert.deepEqual(gulpTask('linux', 'x64'), {
    app: ['vscode-linux-x64-min'],
    installer: ['vscode-linux-x64-prepare-deb', 'vscode-linux-x64-build-deb'],
  });
  assert.throws(() => gulpTask('freebsd', 'x64'));
});

test('builtinExtensionsPath points into the packaged app, never the source tree', () => {
  assert.deepEqual(builtinExtensionsPath('win32', 'x64'), ['VSCode-win32-x64', 'resources', 'app', 'extensions']);
  assert.deepEqual(builtinExtensionsPath('linux', 'x64'), ['VSCode-linux-x64', 'resources', 'app', 'extensions']);
  assert.deepEqual(builtinExtensionsPath('darwin', 'arm64'), ['VSCode-darwin-arm64', 'Synapse.app', 'Contents', 'Resources', 'app', 'extensions']);
});

test('expectedInstallers lists one installer per matrix leg', () => {
  assert.deepEqual(expectedInstallers('1.2.3'), [
    'Synapse_1.2.3_x64-setup.exe',
    'Synapse_1.2.3_aarch64.dmg',
    'Synapse_1.2.3_x64.dmg',
    'Synapse_1.2.3_amd64.deb',
  ]);
});
