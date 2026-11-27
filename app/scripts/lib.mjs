// Pure helpers for build.mjs (no IO, so they are unit-tested in lib.test.mjs).

const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);

/**
 * Deep-merge `overrides` over `upstream`. Objects merge key by key; arrays and scalars in overrides
 * replace (so an explicit `[]` wins over upstream's list). A key set to `null` in overrides is DELETED
 * from the result, which is how an upstream object (for example `defaultChatAgent`) is removed.
 */
export function mergeProduct(upstream, overrides) {
  if (!isObject(upstream) || !isObject(overrides)) return overrides === undefined ? upstream : overrides;
  const out = { ...upstream };
  for (const [key, value] of Object.entries(overrides)) {
    if (value === null) delete out[key];
    else out[key] = isObject(value) ? mergeProduct(isObject(out[key]) ? out[key] : {}, value) : value;
  }
  return out;
}

/**
 * Installer file names. The suffixes must keep matching the site's download resolver
 * (Builderforce.ai/api/src/application/desktop/synapseInstallers.ts).
 * `kind`: 'setup' (win32), 'dmg' (darwin), 'deb' (linux; no AppImage: Electron's sandbox aborts there).
 */
export function installerName(version, platform, arch, kind) {
  const base = `Synapse_${version}`;
  const key = `${platform}/${arch}/${kind}`;
  switch (key) {
    case 'win32/x64/setup': return `${base}_x64-setup.exe`;
    case 'darwin/arm64/dmg': return `${base}_aarch64.dmg`;
    case 'darwin/x64/dmg': return `${base}_x64.dmg`;
    case 'linux/x64/deb': return `${base}_amd64.deb`;
    default: throw new Error(`No installer name for ${key}`);
  }
}

/** Every installer a complete release carries (one per CI matrix leg); publish refuses a release missing any. */
export function expectedInstallers(version) {
  return [
    installerName(version, 'win32', 'x64', 'setup'),
    installerName(version, 'darwin', 'arm64', 'dmg'),
    installerName(version, 'darwin', 'x64', 'dmg'),
    installerName(version, 'linux', 'x64', 'deb'),
  ];
}

const PLATFORMS = { win32: ['x64', 'arm64'], darwin: ['x64', 'arm64'], linux: ['x64', 'arm64'] };

/**
 * Upstream gulp tasks, in the order they must run: the ordered `app` list (the packaged app; on win32 it
 * ends with `inno-updater` so `inno_updater.exe` exists before the signing step) and the ordered `installer`
 * list (empty for darwin: the .dmg is made from the signed .app by make-dmg.mjs).
 */
export function gulpTask(platform, arch) {
  if (!PLATFORMS[platform]?.includes(arch)) throw new Error(`Unsupported target ${platform}-${arch}`);
  const target = `${platform}-${arch}`;
  const installer = {
    win32: [`vscode-${target}-user-setup`],
    linux: [`vscode-${target}-prepare-deb`, `vscode-${target}-build-deb`],
    darwin: [],
  }[platform];
  const app = [`vscode-${target}-min`, ...(platform === 'win32' ? [`vscode-${target}-inno-updater`] : [])];
  return { app, installer };
}

/** Path segments, relative to the directory holding the checkout, of the packaged app's built-in extensions folder. */
export function builtinExtensionsPath(platform, arch) {
  const root = `VSCode-${platform}-${arch}`;
  return platform === 'darwin'
    ? [root, 'Synapse.app', 'Contents', 'Resources', 'app', 'extensions']
    : [root, 'resources', 'app', 'extensions'];
}
