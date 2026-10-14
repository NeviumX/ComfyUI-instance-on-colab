"""Pairing codes: how a browser is let in through the public URL (6-2)."""

import hmac
import json
import re
import secrets
import time

from . import filesystem, paths
from .messages import t


PAIRING_PATH = "/_colab/pair"
PAIRING_SECONDS = 600
PAIRING_ATTEMPTS = 5


PAIRING_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{ font-family: system-ui, sans-serif; background: #202020; color: #eee; margin: 0;
       min-height: 100vh; display: flex; align-items: center; justify-content: center; }}
main {{ width: 20rem; padding: 1.5rem; }}
h1 {{ font-size: 1.3rem; }}
input, button {{ font: inherit; width: 100%; box-sizing: border-box; padding: .6rem; border-radius: .4rem; }}
input {{ font-size: 1.4rem; letter-spacing: .2em; text-align: center; background: #111; color: #eee; border: 1px solid #666; }}
button {{ margin-top: .8rem; background: #3b82f6; color: #fff; border: 0; }}
.notice {{ color: #fca5a5; }}
</style></head>
<body><main><h1>{title}</h1><p>{prompt}</p><p class="notice">{notice}</p>
<form method="post" action="{path}">
<input name="code" inputmode="numeric" autocomplete="one-time-code" maxlength="16" autofocus required>
<button type="submit">{button}</button>
</form></main></body></html>
"""


def new_pairing_code():
    """Issue the code that lets one browser in. It replaces any earlier one."""
    code = "".join(secrets.choice("0123456789") for _ in range(8))
    filesystem.private_write(paths.PRIVATE / "pairing.json", json.dumps(
        {"code": code, "expires": time.time() + PAIRING_SECONDS, "attempts": PAIRING_ATTEMPTS}))
    return code


def redeem_pairing_code(supplied):
    """(accepted, tries left). A code is good for one browser, a few minutes and a few wrong
    guesses; after any of these there is no code until the notebook issues another."""
    path = paths.PRIVATE / "pairing.json"
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        return False, 0
    if time.time() > state["expires"]:
        path.unlink(missing_ok=True)
        return False, 0
    if hmac.compare_digest(re.sub(r"[\s-]", "", supplied).encode(), state["code"].encode()):
        path.unlink(missing_ok=True)
        return True, 0
    state["attempts"] -= 1
    if state["attempts"] <= 0:
        path.unlink(missing_ok=True)
        return False, 0
    filesystem.private_write(path, json.dumps(state))
    return False, state["attempts"]


def show_pairing(config):
    code = new_pairing_code()
    print(t("start.tunnel_heading"))
    print(config["remote_url"] + "/")
    print(t("start.pairing_code", code=code[:4] + "-" + code[4:], minutes=PAIRING_SECONDS // 60))
    print(t("start.pairing_note"))


def issue_pairing_code(config):
    """Let one more browser in while ComfyUI runs behind the tunnel."""
    if config.get("mode") != "tunnel" or not config.get("remote_url"):
        raise RuntimeError(t("pairing.not_running"))
    show_pairing(config)
