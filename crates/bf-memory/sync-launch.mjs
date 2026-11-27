import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

// Build builderforce-memory before running this script. A custom checkout path
// can be supplied after --check or on its own.
const flags = process.argv.slice(2);
const check = flags.includes("--check");
const checkout = flags.find((arg) => arg !== "--check");
const sibling = new URL("../../../builderforce-memory/", import.meta.url);
const memory = checkout ? pathToFileURL(resolve(checkout) + "/") : sibling;
const entry = new URL("packages/memory-mcp/dist/install/server-spec.js", memory);
const { buildServerSpec } = await import(entry.href);
const target = new URL("src/generated-memory-launch.json", import.meta.url);
const generated = JSON.stringify({
  posix: buildServerSpec({ platform: "linux" }),
  windows: buildServerSpec({ platform: "win32" }),
}, null, 2) + "\n";

if (check) {
  if ((await readFile(target, "utf8")).replaceAll("\r\n", "\n") !== generated) {
    throw new Error("Memory launch specs changed. Run node crates/bf-memory/sync-launch.mjs.");
  }
} else {
  await writeFile(target, generated);
}
