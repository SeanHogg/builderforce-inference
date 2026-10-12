# bfi sidecar

Synapse ships `bfi` next to the app binary (Tauri `bundle.externalBin`).

Once CI produces a real engine for the target, drop it here as:

```
bfi-<target-triple>[.exe]
```

for example `bfi-x86_64-pc-windows-msvc.exe`, `bfi-aarch64-apple-darwin`.

`tauri.conf.json` already has `"bundle.externalBin": []`. When the files exist, set it to:

```json
"externalBin": ["binaries/bfi"]
```

Do not put `binaries/bfi` in `externalBin` until those files exist — `tauri-build` fails the compile when the sidecar is missing.

At runtime `bf-local` looks next to the Synapse executable for `bfi` / `bfi.exe` and for `bfi-<target-triple>`, so a copied sidecar is found even before the bundle table is wired. `BFI_BIN` still wins. `SYNAPSE_USE_OLLAMA=1` keeps the deprecated Ollama fallback.
