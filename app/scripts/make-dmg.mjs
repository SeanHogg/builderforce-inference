// Creates the macOS .dmg from the (already signed) Synapse.app directory.
//   node app/scripts/make-dmg.mjs <dir containing Synapse.app> <output.dmg>
import { spawnSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';

export function makeDmg(appDir, dmg) {
  const r = spawnSync('hdiutil', ['create', '-volname', 'Synapse', '-srcfolder', appDir, '-ov', '-format', 'UDZO', dmg], { stdio: 'inherit' });
  if (r.status !== 0) throw new Error(`hdiutil failed (exit ${r.status ?? r.error?.message})`);
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [appDir, dmg] = process.argv.slice(2);
  if (!appDir || !dmg) { console.error('usage: make-dmg.mjs <app dir> <output.dmg>'); process.exit(1); }
  makeDmg(appDir, dmg);
}
