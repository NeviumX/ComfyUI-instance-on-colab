"""The stop button the proxy adds to ComfyUI's page, and carrying out what it asks for."""

import json
import threading

from . import drive, filesystem, paths, service
from .messages import t


STOP_PATH = "/_colab/stop"
STOP_SCRIPT_PATH = "/_colab/stop.js"
# Where the GitHub mark in the stop dialog leads. Empty: no mark.
REPOSITORY_URL = "https://github.com/NeviumX/ComfyUI-instance-on-colab"
# Added to the ComfyUI page by the proxy: a small button in the tab row, and the dialog it opens.
STOP_SCRIPT = r"""(() => {
  if (window.top !== window || document.getElementById("colab-stop-button")) return;
  const text = __TEXTS__;
  const style = document.createElement("style");
  style.textContent = `
    #colab-stop-button { position: fixed; top: 7px; right: 56px; z-index: 1000; font: 12px/1 system-ui, sans-serif;
      padding: 5px 10px; border-radius: 6px; border: 1px solid rgba(128, 128, 128, .5); background: rgba(30, 30, 30, .7);
      color: #ddd; cursor: pointer; opacity: .6; }
    #colab-stop-button:hover, #colab-stop-button:focus-visible { opacity: 1; color: #fff; border-color: #f87171; }
    #colab-stop-dialog { font: 14px/1.6 system-ui, sans-serif; width: min(26rem, calc(100vw - 2rem)); padding: 1.25rem;
      border: 1px solid #555; border-radius: 10px; background: #202020; color: #eee;
      box-shadow: 0 12px 40px rgba(0, 0, 0, .6); }
    #colab-stop-dialog::backdrop { background: rgba(0, 0, 0, .6); }
    #colab-stop-dialog h2 { margin: 0 2.25rem .5rem 0; font-size: 1.1rem; }
    #colab-stop-dialog .colab-stop-repository { position: absolute; top: 1.1rem; right: 1.1rem; display: flex; color: #aaa; }
    #colab-stop-dialog .colab-stop-repository:hover, #colab-stop-dialog .colab-stop-repository:focus-visible { color: #fff; }
    #colab-stop-dialog p { margin: 0 0 .75rem; }
    #colab-stop-dialog .colab-stop-status { color: #fcd34d; }
    #colab-stop-dialog .colab-stop-status:empty { display: none; }
    #colab-stop-dialog .colab-stop-actions { display: flex; flex-direction: column; gap: .5rem; }
    #colab-stop-dialog button { font: inherit; padding: .55rem .75rem; border-radius: 6px; border: 1px solid #666;
      background: #333; color: #eee; cursor: pointer; }
    #colab-stop-dialog button:hover:enabled { background: #444; }
    #colab-stop-dialog button:disabled { opacity: .45; cursor: default; }
    #colab-stop-dialog button.colab-stop-danger { background: #b91c1c; border-color: #b91c1c; color: #fff; }
    #colab-stop-dialog button.colab-stop-danger:hover:enabled { background: #dc2626; }
    #colab-stop-dialog button.colab-stop-quiet { background: transparent; }
  `;
  const add = (parent, tag, content, name) => {
    const element = document.createElement(tag);
    if (content) element.textContent = content;
    if (name) element.className = name;
    if (tag === "button") element.type = "button";
    return parent.appendChild(element);
  };
  const dialog = document.createElement("dialog");
  dialog.id = "colab-stop-dialog";
  add(dialog, "h2", text.title);
  add(dialog, "p", text.explain);
  const status = add(dialog, "p", "", "colab-stop-status");
  const actions = add(dialog, "div", "", "colab-stop-actions");
  const remove = add(actions, "button", text.delete, "colab-stop-danger");
  const disconnect = add(actions, "button", text.disconnect);
  const cancel = add(actions, "button", text.cancel, "colab-stop-quiet");
  cancel.autofocus = true;  // Enter right after opening must not be a choice to stop.
  const repository = "__REPOSITORY__";
  if (repository) {
    // Last in the dialog, so the keyboard reaches the choices first; shown in its top right corner.
    const link = add(dialog, "a", "", "colab-stop-repository");
    link.href = repository;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.title = text.repository;
    link.setAttribute("aria-label", text.repository);
    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    for (const [name, value] of [["viewBox", "0 0 16 16"], ["width", "20"], ["height", "20"], ["fill", "currentColor"], ["aria-hidden", "true"]]) {
      icon.setAttribute(name, value);
    }
    const mark = document.createElementNS("http://www.w3.org/2000/svg", "path");
    mark.setAttribute("d", "M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z");
    icon.appendChild(mark);
    link.appendChild(icon);
  }
  const button = document.createElement("button");
  button.id = "colab-stop-button";
  button.type = "button";
  button.textContent = text.stop;
  button.title = text.title;
  let busy = false;
  const lock = (locked) => {
    busy = locked;
    for (const choice of [remove, disconnect, cancel]) choice.disabled = locked;
  };
  const ask = async (deleteRuntime) => {
    lock(true);
    status.textContent = text.working;
    try {
      const answer = await fetch("__STOP_PATH__", {method: "POST", headers: {"Content-Type": "application/json"},
                                               body: JSON.stringify({delete_runtime: deleteRuntime})});
      if (!answer.ok) throw new Error(answer.status);
    } catch (error) {
      status.textContent = text.failed;
      lock(false);
      return;
    }
    const started = Date.now();
    const timer = setInterval(async () => {
      let state;
      try {
        const answer = await fetch("__STOP_PATH__", {cache: "no-store"});
        if (!answer.ok) throw new Error(answer.status);
        state = (await answer.json()).state;
      } catch (error) {
        // Nothing answers any more: the proxy and the tunnel are down.
        clearInterval(timer);
        status.textContent = deleteRuntime ? text.deleted : text.stopped;
        button.remove();
        return;
      }
      if (state === "failed") {
        clearInterval(timer);
        status.textContent = text.failed;
        lock(false);
      } else if (Date.now() - started > 180000) {
        status.textContent = text.slow;
      }
    }, 3000);
  };
  remove.addEventListener("click", () => ask(true));
  disconnect.addEventListener("click", () => ask(false));
  cancel.addEventListener("click", () => dialog.close());
  dialog.addEventListener("cancel", (event) => { if (busy) event.preventDefault(); });
  // Keys pressed in the dialog are not ComfyUI shortcuts.
  dialog.addEventListener("keydown", (event) => event.stopPropagation());
  button.addEventListener("click", () => {
    if (!busy) status.textContent = "";
    dialog.showModal();
  });
  const mount = () => {
    document.head.appendChild(style);
    document.body.append(button, dialog);
  };
  if (document.body) mount(); else document.addEventListener("DOMContentLoaded", mount);
})();
"""


def with_stop_button(page):
    """The ComfyUI page with the script that adds the stop button (unchanged if it has no head)."""
    position = page.lower().rfind(b"</head>")
    tag = b'<script defer src="' + STOP_SCRIPT_PATH.encode() + b'"></script>'
    return page if position < 0 else page[:position] + tag + page[position:]


def stop_button_script():
    texts = {"stop": t("menu.stop"), "title": t("menu.title"), "explain": t("menu.explain"), "delete": t("menu.delete"),
             "disconnect": t("menu.disconnect"), "cancel": t("menu.cancel"), "working": t("menu.working"),
             "stopped": t("menu.stopped"), "deleted": t("menu.deleted"), "failed": t("menu.failed"), "slow": t("menu.slow"),
             "repository": t("menu.repository")}
    repository = REPOSITORY_URL if REPOSITORY_URL.startswith("https://") else ""
    return (STOP_SCRIPT.replace("__TEXTS__", json.dumps(texts)).replace("__STOP_PATH__", STOP_PATH)
            .replace('"__REPOSITORY__"', json.dumps(repository)))


def request_stop(delete_runtime):
    """Leave word, from the proxy, that the ComfyUI page asked to stop. The notebook's own
    process carries it out: only it can save to Drive and give the runtime back to Colab."""
    (paths.PRIVATE / "stop-status.json").unlink(missing_ok=True)
    filesystem.private_write(paths.PRIVATE / "stop-request.json", json.dumps({"delete_runtime": bool(delete_runtime)}))


def stop_request_state():
    if (paths.PRIVATE / "stop-request.json").exists():
        return "pending"
    return "failed" if (paths.PRIVATE / "stop-status.json").exists() else "idle"


def delete_colab_runtime():
    """Finish writing to Drive, then give the runtime back: it stops using compute units."""
    from google.colab import drive as colab_drive, runtime as colab
    drive.hide_drive_models()
    colab_drive.flush_and_unmount()
    colab.unassign()


def carry_out_stop_request(config):
    """Do what the stop button of the ComfyUI page asked for, if it asked. True once stopped."""
    request = paths.PRIVATE / "stop-request.json"
    try:
        wanted = json.loads(request.read_text())
    except (OSError, ValueError):
        return False
    try:
        service.stop(config, save=True)
    except Exception as failure:
        # Nothing was stopped, so the page is still there to say that it did not work.
        filesystem.private_write(paths.PRIVATE / "stop-status.json", json.dumps({"state": "failed"}))
        request.unlink(missing_ok=True)
        print(t("menu.failed"), failure, flush=True)
        return False
    if wanted.get("delete_runtime") is True:
        delete_colab_runtime()
    return True


STOP_WATCH = []
STOP_WATCH_SECONDS = 2


def follow_stop_requests(config):
    """Watch for the stop button from the notebook's process, until stopped one way or another."""
    cancelled = threading.Event()
    STOP_WATCH[:] = [cancelled]

    def watch():
        while not cancelled.wait(STOP_WATCH_SECONDS):
            if carry_out_stop_request(config):
                return
    threading.Thread(target=watch, daemon=True).start()
