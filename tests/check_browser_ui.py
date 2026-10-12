"""Open a ComfyUI URL in a headless browser on this PC and report why its splash logo does not clear.

Manual check; needs Chrome or Edge and the dev requirements. The browser goes through whatever
filters this PC applies (ad blockers, security software), which a check from Colab cannot see.

usage: python tests/check_browser_ui.py URL [wait_seconds]
Set PROBE_BROWSER to the path of another Chromium-based browser to test that one.
Set PROBE_MBPS (and PROBE_LATENCY_MS) to emulate a slow link, PROBE_SECOND_LOAD=1 to also time a reload.
Set PROBE_PAIRING_CODE to the code cell 6-2 shows (issue a new one for this check) to pass its pairing page.
"""
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit

import aiohttp

BROWSER = os.environ.get("PROBE_BROWSER") or next(p for p in (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe") if os.path.exists(p))
URL = sys.argv[1]
WAIT = float(sys.argv[2]) if len(sys.argv) > 2 else 40
PORT = 9377
CODE = os.environ.get("PROBE_PAIRING_CODE", "")
ENTER_CODE = ("(() => { const field = document.querySelector('form input[name=code]'); if (!field) return false; "
              "field.value = " + json.dumps(CODE) + "; field.form.submit(); return true; })()")
HOST = urlsplit(URL).hostname


def show(*parts):
    print(re.sub(r"access_key=[A-Za-z0-9_-]+", "access_key=[REDACTED]", " ".join(map(str, parts))))


async def main():
    profile = tempfile.mkdtemp(prefix="cdp-probe-")
    proc = subprocess.Popen([BROWSER, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                             "--no-first-run", "--disable-extensions", "--window-size=1400,900", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        async with aiohttp.ClientSession() as http:
            for _ in range(50):
                try:
                    async with http.get(f"http://127.0.0.1:{PORT}/json") as res:
                        targets = await res.json()
                        break
                except aiohttp.ClientError:
                    await asyncio.sleep(0.2)
            page = next(t for t in targets if t["type"] == "page")
            async with http.ws_connect(page["webSocketDebuggerUrl"], max_msg_size=0) as ws:
                counter = 0
                pending = {}
                requests, console, failures, statuses = {}, [], [], {}
                sockets = []

                async def send(method, **params):
                    nonlocal counter
                    counter += 1
                    future = asyncio.get_running_loop().create_future()
                    pending[counter] = future
                    await ws.send_json({"id": counter, "method": method, "params": params})
                    return await future

                async def reader():
                    async for raw in ws:
                        msg = json.loads(raw.data)
                        if "id" in msg:
                            pending.pop(msg["id"]).set_result(msg.get("result", msg.get("error")))
                            continue
                        method, p = msg.get("method"), msg.get("params", {})
                        if method == "Network.requestWillBeSent":
                            requests[p["requestId"]] = {"url": p["request"]["url"], "method": p["request"]["method"],
                                                        "start": time.monotonic(), "done": False}
                        elif method == "Network.responseReceived":
                            statuses[p["requestId"]] = p["response"]["status"]
                        elif method == "Network.loadingFinished":
                            if p["requestId"] in requests:
                                requests[p["requestId"]]["done"] = True
                                requests[p["requestId"]]["bytes"] = p.get("encodedDataLength", 0)
                        elif method == "Network.loadingFailed":
                            if p["requestId"] in requests:
                                requests[p["requestId"]]["done"] = True
                                failures.append((requests[p["requestId"]]["url"], p.get("errorText"), p.get("blockedReason")))
                        elif method == "Network.webSocketCreated":
                            sockets.append(["created", p["url"]])
                        elif method == "Network.webSocketClosed":
                            sockets.append(["closed", p["requestId"]])
                        elif method == "Network.webSocketFrameReceived":
                            sockets.append(["frame", p["response"]["payloadData"][:120]])
                        elif method == "Runtime.consoleAPICalled":
                            text = " ".join(str(a.get("value", a.get("description", ""))) for a in p["args"])
                            console.append((p["type"], text[:600]))
                        elif method == "Runtime.exceptionThrown":
                            d = p["exceptionDetails"]
                            console.append(("EXCEPTION", (d.get("exception", {}).get("description") or d.get("text", ""))[:800]))
                        elif method == "Log.entryAdded":
                            e = p["entry"]
                            console.append(("log:" + e["level"], (e["text"] + " " + e.get("url", ""))[:400]))

                task = asyncio.create_task(reader())
                for domain in ("Network", "Runtime", "Log", "Page"):
                    await send(domain + ".enable")
                if os.environ.get("PROBE_MBPS"):  # Emulate a slow link, e.g. PROBE_MBPS=2 PROBE_LATENCY_MS=200.
                    rate = float(os.environ["PROBE_MBPS"]) * 125000
                    await send("Network.emulateNetworkConditions", offline=False, downloadThroughput=rate,
                               uploadThroughput=rate, latency=float(os.environ.get("PROBE_LATENCY_MS", 0)))
                async def wait_for_ui():
                    started = time.monotonic()
                    entered = not CODE
                    await send("Page.navigate", url=URL)
                    while time.monotonic() - started < WAIT:
                        await asyncio.sleep(1)
                        if not entered:
                            result = await send("Runtime.evaluate", returnByValue=True, expression=ENTER_CODE)
                            entered = result.get("result", {}).get("value") is True
                            continue
                        # about:blank has no splash either, so require the frontend's own mount point.
                        result = await send("Runtime.evaluate", returnByValue=True, expression=
                                            "!!document.getElementById('vue-app') && !document.getElementById('splash-loader')")
                        if result.get("result", {}).get("value") is True:
                            cleared = time.monotonic() - started
                            await asyncio.sleep(3)
                            return cleared
                cleared_at = await wait_for_ui()
                state = await send("Runtime.evaluate", returnByValue=True, expression="""JSON.stringify({
                    href: location.href, title: document.title,
                    splash: !!document.getElementById('splash-loader'),
                    canvas: !!document.querySelector('canvas'),
                    clientId: window.app && window.app.api ? window.app.api.clientId : null,
                    body: document.body.innerText.slice(0, 300)})""")
                show("RESULT:", "splash cleared after %.1fs" % cleared_at if cleared_at else f"SPLASH STILL SHOWN after {WAIT:.0f}s")
                show("STATE:", state.get("result", {}).get("value"))
                others = sorted({urlsplit(r["url"]).hostname for r in requests.values()} - {HOST, None, ""})
                megabytes = sum(r.get("bytes", 0) for r in requests.values()) / 1048576
                show(f"REQUESTS: {len(requests)} total, {megabytes:.1f} MB transferred; other hosts contacted (a local filter injecting scripts shows up here): {others}")
                bad = [(statuses[i], r["method"], r["url"]) for i, r in requests.items() if statuses.get(i, 0) >= 400]
                show("HTTP >= 400:")
                for item in bad[:40]:
                    show("  ", *item)
                show("FAILED:")
                for item in failures[:40]:
                    show("  ", *item)
                show("STILL PENDING:")
                now = time.monotonic()
                for r in requests.values():
                    if not r["done"]:
                        show("   %5.1fs %s %s" % (now - r["start"], r["method"], r["url"][:200]))
                show("WEBSOCKET:")
                for item in sockets[:12]:
                    show("  ", *item)
                show("CONSOLE (warnings/errors):")
                for kind, text in console:
                    if kind in ("error", "warning", "EXCEPTION", "log:error", "log:warning"):
                        show("   [%s] %s" % (kind, text))
                if os.environ.get("PROBE_SECOND_LOAD"):  # What a reload or reconnect costs once the cache is warm.
                    first = set(requests)
                    again = await wait_for_ui()
                    megabytes = sum(r.get("bytes", 0) for i, r in requests.items() if i not in first) / 1048576
                    show("SECOND LOAD:", "splash cleared after %.1fs" % again if again else "SPLASH STILL SHOWN",
                         f"({len(set(requests) - first)} requests, {megabytes:.1f} MB transferred)")
                task.cancel()
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


asyncio.run(main())
