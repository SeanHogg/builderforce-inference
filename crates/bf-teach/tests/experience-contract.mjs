// Generate or check the native contract using a built builderforce-memory-engine entry.
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const [engineEntry, command] = process.argv.slice(2);
if (!engineEntry) throw new Error('Pass the built memory engine JS entry, then optionally --check.');
const { trainOnce } = await import(pathToFileURL(resolve(engineEntry)).href);

const target = {
  name: 'Amount',
  automationId: 'amount',
  controlType: 'Edit',
  className: 'TextBox',
  windowTitle: 'Invoice',
  processName: 'ledger',
  relX: 25,
  relY: -12,
};
const episode = {
  id: '0000000000000001',
  name: 'Record invoice',
  program: 'ledger.exe',
  args: ['--sandbox'],
  startedAt: 1730000000000,
  endedAt: 1730000001000,
  steps: [
    { id: 'open', atMs: 0, action: { kind: 'click', target }, screenshot: 'open.jpg' },
    { id: 'amount', atMs: 50, action: { kind: 'setValue', target, value: '250', secret: false }, screenshot: null },
    { id: 'fixed', atMs: 100, action: { kind: 'setValue', target, value: 'note', secret: false }, screenshot: null },
    { id: 'secret', atMs: 150, action: { kind: 'setValue', target, value: '', secret: true }, screenshot: null },
    { id: 'next', atMs: 200, action: { kind: 'keys', keys: 'Enter', target: null }, screenshot: null },
  ],
};
const skill = trainOnce(episode, { fixed: ['fixed'], approvals: { open: true } }, episode.endedAt);
skill.id = '0000000000000002';
skill.routine = { schedule: { every: 'minutes', minutes: 5 }, values: {}, enabled: true };
const encoded = JSON.stringify({ episode, skill }, null, 2) + '\n';
const fixture = new URL('./fixtures/experience-contract.json', import.meta.url);
if (command === '--check') {
  if (readFileSync(fixture, 'utf8').replace(/\r\n/g, '\n') !== encoded) {
    throw new Error('The Teach helper fixture differs from the built engine contract.');
  }
} else {
  writeFileSync(fixture, encoded);
}
