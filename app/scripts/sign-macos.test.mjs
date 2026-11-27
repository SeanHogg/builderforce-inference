import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { isMachO, findMachO, findBundles, signingOrder, planSigning } from './sign-macos.mjs';

const MAGICS = { feedface: 'feedface', feedfacf: 'feedfacf', cafebabe: 'cafebabe', cafebabf: 'cafebabf', cefaedfe: 'cefaedfe', cffaedfe: 'cffaedfe' };

function withTmp(fn) {
  const dir = mkdtempSync(join(tmpdir(), 'sign-macos-'));
  try { fn(dir); } finally { rmSync(dir, { recursive: true, force: true }); }
}

test('isMachO accepts every Mach-O and fat magic', () => {
  withTmp((dir) => {
    for (const [name, hex] of Object.entries(MAGICS)) {
      const f = join(dir, name);
      writeFileSync(f, Buffer.concat([Buffer.from(hex, 'hex'), Buffer.from([0, 0, 0, 2])]));
      assert.equal(isMachO(f), true, name);
    }
  });
});

test('isMachO rejects scripts, short and missing files', () => {
  withTmp((dir) => {
    writeFileSync(join(dir, 'script.sh'), '#!/bin/sh\necho hi\n');
    writeFileSync(join(dir, 'short'), Buffer.from([0xfe, 0xed]));
    writeFileSync(join(dir, 'empty'), '');
    assert.equal(isMachO(join(dir, 'script.sh')), false);
    assert.equal(isMachO(join(dir, 'short')), false);
    assert.equal(isMachO(join(dir, 'empty')), false);
    assert.equal(isMachO(join(dir, 'missing')), false);
  });
});

test('findMachO and findBundles walk the tree', () => {
  withTmp((dir) => {
    const res = join(dir, 'Synapse.app', 'Contents', 'Resources', 'app', 'node_modules');
    const helper = join(dir, 'Synapse.app', 'Contents', 'Frameworks', 'Helper.app', 'Contents', 'MacOS');
    mkdirSync(res, { recursive: true });
    mkdirSync(helper, { recursive: true });
    const macho = Buffer.from('cffaedfe00000000', 'hex');
    writeFileSync(join(res, 'a.node'), macho);
    writeFileSync(join(res, 'readme.txt'), 'text');
    writeFileSync(join(helper, 'Helper'), macho);
    const app = join(dir, 'Synapse.app');
    assert.deepEqual(findMachO(app).sort(), [join(helper, 'Helper'), join(res, 'a.node')].sort());
    assert.deepEqual(findBundles(app), [join(app, 'Contents', 'Frameworks', 'Helper.app')]);
  });
});

test('signingOrder signs the deepest paths first', () => {
  assert.deepEqual(signingOrder(['/a/b', '/a/b/c/d', '/a', '/a/b/c']), ['/a/b/c/d', '/a/b/c', '/a/b', '/a']);
  assert.deepEqual(signingOrder(['/a/z', '/a/b']), ['/a/b', '/a/z'], 'ties are name-ordered');
});

test('planSigning: Mach-O files, then nested bundles, then the main bundle last', () => {
  const app = '/x/Synapse.app';
  const plan = planSigning(
    app,
    ['/x/Synapse.app/Contents/Resources/a.node', '/x/Synapse.app/Contents/Frameworks/H.app/Contents/MacOS/H'],
    ['/x/Synapse.app/Contents/Frameworks/H.app', '/x/Synapse.app/Contents/Frameworks/E.framework', app],
  );
  assert.deepEqual(plan, [
    '/x/Synapse.app/Contents/Frameworks/H.app/Contents/MacOS/H',
    '/x/Synapse.app/Contents/Resources/a.node',
    '/x/Synapse.app/Contents/Frameworks/E.framework',
    '/x/Synapse.app/Contents/Frameworks/H.app',
    app,
  ]);
});

test('isMachO rejects a Java .class header and degenerate fat headers', () => {
  withTmp((dir) => {
    const cases = {
      'Foo.class': [0xca, 0xfe, 0xba, 0xbe, 0, 0, 0, 52],
      zeroArchs: [0xca, 0xfe, 0xba, 0xbe, 0, 0, 0, 0],
      'fat64.class': [0xca, 0xfe, 0xba, 0xbf, 0, 0, 0, 45],
      truncated: [0xca, 0xfe, 0xba, 0xbe],
    };
    for (const [name, bytes] of Object.entries(cases)) {
      writeFileSync(join(dir, name), Buffer.from(bytes));
      assert.equal(isMachO(join(dir, name)), false, name);
    }
  });
});

test('isMachO accepts a fat header with 2 archs', () => {
  withTmp((dir) => {
    writeFileSync(join(dir, 'fat'), Buffer.from([0xca, 0xfe, 0xba, 0xbe, 0, 0, 0, 2]));
    assert.equal(isMachO(join(dir, 'fat')), true);
  });
});
