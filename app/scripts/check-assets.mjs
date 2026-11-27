// Fails when a release is missing any expected installer.
//   node app/scripts/check-assets.mjs <version> <file listing the release's asset names, one per line>
import { readFileSync } from 'node:fs';
import { expectedInstallers } from './lib.mjs';

const [version, listing] = process.argv.slice(2);
if (!version || !listing) { console.error('usage: check-assets.mjs <version> <assets-file>'); process.exit(2); }
const have = new Set(readFileSync(listing, 'utf8').split(/\r?\n/).map((s) => s.trim()).filter(Boolean));
const missing = expectedInstallers(version).filter((n) => !have.has(n));
if (missing.length) {
  console.error(`::error::v${version} is missing installers, so it stays a draft: ${missing.join(', ')}`);
  process.exit(1);
}
console.log(`v${version} has every expected installer`);
