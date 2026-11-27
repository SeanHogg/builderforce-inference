# Patches

Each `*.patch` file here is a unified diff (the output of `git diff` or `git format-patch`)
made against the upstream checkout pinned in `../upstream.json`.

- `scripts/build.mjs` applies them with `git apply`, in filename order. Number them
  (`0001-branding.patch`, `0002-update-url.patch`) to control the order.
- Keep each patch minimal and single-purpose: branding strings and the update URL only.
  Anything that can be expressed in `../product.json` goes there instead.
- A patch that no longer applies after a tag bump fails the build; regenerate it.
- An empty set (only `.gitkeep`) is valid. There are no patches yet.

## First CI-time task

Upstream hard-codes "Visual Studio Code" in the Welcome / walkthrough strings (and a few other UI strings).
`product.json` cannot override those, and a patch cannot be authored without a checkout. So the first task, on
the first real CI build, is to clone the pinned tag, grep for "Visual Studio Code" / "VS Code" in user-visible
strings, and commit the minimal `0001-branding.patch` that renames them to "Synapse".
