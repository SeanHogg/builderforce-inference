# Synapse desktop app (developer docs)

Synapse is built from the Code-OSS source (the open-source base of VS Code) using a build recipe,
the way VSCodium does. This repo does **not** contain the editor source: `scripts/build.mjs` clones
the pinned upstream tag into `app/.build/vscode` (gitignored), overlays Synapse's branding and
extensions, and runs upstream's own packaging. Design: `Builderforce.ai/docs/specs/tasks/synapse-vscode-fork.md` section 1.

## Files

| Path | Purpose |
|---|---|
| `upstream.json` | The pinned upstream tag (`vscodeTag`). Confirm the tag exists and still matches the extension's `chatSessionsProvider` proposal before the first CI build. |
| `product.json` | Merged over upstream's `product.json`: names, ids, `synapse:` URL scheme, Open VSX gallery (`serviceUrl`, `itemUrl`, `resourceUrlTemplate`; `controlUrl`/`latestUrlTemplate` are deliberately omitted, as their Open VSX values are unconfirmed), enabled API proposals. Upstream's `builtInExtensions` (the JS debugger) are kept. No Microsoft telemetry keys. The `{{GUID}` win32 ids are Inno Setup escaped braces: keep that form. |
| `patches/` | Minimal `*.patch` files applied with `git apply`, in order. Empty today. See `patches/README.md`. |
| `icons/` | Synapse icons (`icon.ico`, `icon.icns`, `icon.png`, sized PNGs), copied over upstream's `resources/*/code.*`. |
| `scripts/lib.mjs` | Pure helpers: `mergeProduct`, `installerName`, `gulpTask`. |
| `scripts/lib.test.mjs` | `node:test` tests for the helpers (`node --test app/scripts`). |
| `scripts/build.mjs` | The build CLI. |
| `scripts/sign-macos.mjs` | Inside-out macOS signing of the packaged `.app` (tested in `sign-macos.test.mjs`). |
| `../.github/workflows/synapse.yml` | CI matrix: win32-x64, darwin-arm64, darwin-x64, linux-x64. |

Installer names (`Synapse_<v>_x64-setup.exe`, `_aarch64.dmg`, `_x64.dmg`, `_amd64.deb`) are matched by the
site's download resolver, `Builderforce.ai/api/src/application/desktop/synapseInstallers.ts`. Change them in both places or not at all.

## Build locally

Prerequisites: git, the Node version in upstream's `.nvmrc`, and upstream's native toolchain for your OS
(Windows: VS Build Tools + Python; Linux: `libx11-dev libxkbfile-dev libsecret-1-dev libkrb5-dev fakeroot dpkg`; macOS: Xcode CLI tools).
You also need the two extension packages as `.vsix` files (the Builderforce extension and the Spawn extension).

```
node app/scripts/build.mjs --platform win32 --arch x64 --version 2026.11.27 \
  --builderforce-vsix path/to/builderforce-ai.vsix --spawn-vsix path/to/spawn.vsix
```

The build runs in stages (`--stage fetch|package|installer|all`, default `all`) so CI can sign in between:

1. `fetch`: clone upstream at `vscodeTag` (or reuse a checkout already at it), so CI can read its `.nvmrc`.
2. `package`: write the merged `product.json` (a `null` value deletes an upstream key, an array replaces), apply the patches, copy the
   icons, run `npm ci` and the upstream `-min` gulp task, then unpack both VSIX files into the PACKAGED app's built-in extensions folder
   (`VSCode-<plat>-<arch>/resources/app/extensions/<publisher>.<name>`; on macOS `Synapse.app/Contents/Resources/app/extensions/`).
3. CI signs the packaged app here: macOS `scripts/sign-macos.mjs` (every Mach-O file found by magic bytes, inside-out, then helper bundles, then the main bundle, then `codesign --verify --deep --strict`) with `build/darwin/entitlements.plist` (includes microphone access for dictation; one entitlements file signs the main app and every helper, whereas upstream's `build/darwin/sign.js` uses separate Renderer/GPU/Plugin sets, which are not reproduced offline); Windows
   `scripts/sign-windows.ps1` over every `.exe`/`.dll` (looks up `signtool.exe` in the Windows SDK for the `.pfx` path).
4. `installer`: upstream's installer tasks in order (Windows `user-setup`, after `inno-updater` ran in the package stage so it is signed; Linux `prepare-deb` then `build-deb`), or on
   macOS `scripts/make-dmg.mjs` (`hdiutil`), then rename into `app/.build/out/`. CI then signs the installer (macOS: sign, notarize,
   staple the `.dmg`). Linux ships only the `.deb`: the AppImage was dropped because Electron's sandbox aborts there.

Unit tests (no network): `node --test app/scripts`.

Assumptions to confirm at the first real build: upstream still uses npm (`npm ci`) at the pinned tag; the gulp task names
(`vscode-<os>-<arch>-min`, `vscode-win32-<arch>-user-setup`, `vscode-linux-<arch>-build-deb`) and the output paths
(`.build/win32-x64/user-setup/VSCodeSetup.exe`, `.build/linux/deb/<arch>/deb/*.deb`, `VSCode-<os>-<arch>` next to the checkout) are unchanged.

The build must keep `ELECTRON_RUN_AS_NODE` working (the extension spawns `process.execPath` with it) and the `vscode-webview://`
origin unchanged (the gateway's CORS allows it).

## Cut a release

1. Make sure the Builderforce and Spawn `.vsix` release assets exist.
2. Either push a tag `synapse-v<version>` (for example `synapse-v2026.11.27`), or run the **Synapse desktop** workflow by hand with
   `version` and the two VSIX URLs. Tag-triggered runs read the URLs from the repository variables `BUILDERFORCE_VSIX_URL` and `SPAWN_VSIX_URL`.
3. The workflow builds all four targets, signs them with the composite actions in `.github/actions/`, and attaches the installers to
   the release `v<version>`, which it creates as a DRAFT (at the workflow's commit). The `publish` job checks that every expected
   installer is attached (`expectedInstallers` in `lib.mjs`) and only then runs `gh release edit v<version> --draft=false --latest`; if any
   leg failed it fails and the release stays a draft. The headless `release.yml` (`bfi`, `bf-context`) publishes with `--latest=false`.
   darwin-x64 builds on `macos-15-intel`; if that label goes away, cross-build x64 on `macos-latest` with `npm_config_arch=x64`.

## Known gaps

- **No self-update.** `updateUrl` is deliberately unset: there is no Synapse update server yet, so an installed Synapse does not
  self-update until an update endpoint exists (users reinstall from the release page). Do not point it at a made-up URL.
- Hard-coded "Visual Studio Code" walkthrough strings need upstream patches; see `patches/README.md`.
- Product fields that need checking against the pinned tag on the first build: the `win32ContextMenu` shape (only the `clsid`s are
  overridden; `appx` names come from upstream) and whether the `win32Tunnel*Mutex` / `darwinProfile*UUID` keys exist upstream.
