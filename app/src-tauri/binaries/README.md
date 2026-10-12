# bfi sidecar

Synapse ships `bfi` next to the app binary (Tauri `bundle.externalBin`).

`release.yml` builds a CPU `bfi` per target and stages it here as:

```
bfi-<target-triple>[.exe]
```

for example `bfi-x86_64-pc-windows-msvc.exe`, `bfi-aarch64-apple-darwin`. It then sets `externalBin` for that job only.

`tauri.conf.json` stays `"bundle.externalBin": []` in git. Do not put `binaries/bfi` in the committed table until those files exist in the tree — `tauri-build` fails the compile when the sidecar is missing.

At runtime `bf-local` looks next to the Synapse executable for `bfi` / `bfi.exe` and for `bfi-<target-triple>`, so a copied sidecar is found even before the bundle table is wired. `BFI_BIN` still wins.
