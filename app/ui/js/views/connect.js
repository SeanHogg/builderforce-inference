// Show the VS Code extension’s connection to the local index.
import { h } from "../bridge.js";
import { lastSeen } from "../clients.js";
import { relTime, t } from "../i18n.js";
import { subscribe } from "../store.js";

function vscodeConnection() {
  const status = h("span");
  const el = h(
    "section",
    { class: "card tool" },
    h("div", { class: "row between wrap" }, h("h2", { text: t("connect.vscode") }), status),
    h("p", { class: "muted", text: t("connect.vscodeBody") }),
  );
  return {
    el,
    update(entries) {
      const at = lastSeen(entries, "vscode");
      status.replaceChildren(
        h("span", { class: `pill ${at ? "ok" : "idle"}`, text: at ? t("connect.lastRequest", { when: relTime(at) }) : t("connect.noRequests") }),
      );
    },
  };
}

export function render(host) {
  const connection = vscodeConnection();
  host.append(
    h("header", { class: "page-head" }, h("h1", { text: t("connect.title") })),
    connection.el,
    h("p", { class: "muted small privacy", text: t("connect.privacy") }),
  );
  return subscribe((s) => connection.update(s.activity));
}
