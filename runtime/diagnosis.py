"""Connection diagnosis (cell 7) and the report on what the application user can reach."""

import asyncio
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.parse import quote

from . import commands, proxy
from .messages import t


# What the frontend awaits before it removes its splash logo.
STARTUP_ROUTES = ("/", "/api/features", "/api/users", "/api/settings", "/api/i18n", "/api/extensions",
                  "/api/object_info", "/api/userdata?dir=workflows&recurse=true&split=false&full_info=true")


async def probe_frontend(base, headers=None, timeout=30):
    """Request the frontend's startup dependencies at base; print each result and return the failure count."""
    from aiohttp import ClientSession, ClientTimeout, DummyCookieJar, WSMsgType
    failures = 0
    compressed = []

    async def fetch(session, route):
        started = time.monotonic()
        try:
            async with session.get(base + route, headers=headers, allow_redirects=False) as res:
                body = await res.read()
                compressed.append("Content-Encoding" in res.headers)
                return res.status, body, time.monotonic() - started
        except Exception as exc:
            return type(exc).__name__, b"", time.monotonic() - started

    def healthy(route, status, seconds):
        missing_is_normal = status == 404 and "userdata" in route
        return isinstance(status, int) and (status < 400 or missing_is_normal) and seconds < 10

    async with ClientSession(timeout=ClientTimeout(total=timeout), cookie_jar=DummyCookieJar()) as session:
        files = set()
        for route in STARTUP_ROUTES:
            status, body, seconds = await fetch(session, route)
            ok = healthy(route, status, seconds)
            failures += not ok
            print(t("diagnosis.route", mark="OK" if ok else "NG", status=status, seconds=seconds, size=len(body),
                    route=route.split("?")[0]))
            if route == "/":
                files.update("/" + name.decode() for name in re.findall(rb'(?:src|href)="\./(assets/[^"]+)"', body))
            elif route == "/api/extensions" and status == 200:
                # The list comes from ComfyUI; only a path can be appended to the address safely.
                files.update(quote(name) for name in json.loads(body) if isinstance(name, str) and name.startswith("/"))
        gate = asyncio.Semaphore(16)

        size = 0

        async def fetch_file(route):
            nonlocal size
            async with gate:
                status, body, seconds = await fetch(session, route)
                size += len(body)
                # Follow the chunks each script imports, as the browser does.
                more = {"/assets/" + name.decode() for name in
                        re.findall(rb"(?:\./|assets/)([\w.-]+-[\w-]{8}\.(?:js|css))", body)} if route.startswith("/assets/") else set()
                return route, status, seconds, more
        del compressed[:]
        started = time.monotonic()
        results, pending, listed = [], sorted(files), set(files)
        while pending:
            batch = await asyncio.gather(*map(fetch_file, pending))
            results += [item[:3] for item in batch]
            found = set().union(*(item[3] for item in batch)) - files
            files |= found
            pending = sorted(found)
        elapsed = time.monotonic() - started
        # A name found inside a script may be just text; only a listed file must exist.
        bad = [item for item in results if not healthy(*item) and (item[0] in listed or item[1] != 404)]
        failures += len(bad)
        # The browser downloads these before the UI appears, so this time predicts the wait.
        print(t("diagnosis.assets", mark="NG" if bad else "OK", total=len(results), failed=len(bad),
                megabytes=size / 1048576, elapsed=elapsed, compressed=sum(compressed)))
        for route, status, seconds in bad[:10]:
            print(t("diagnosis.failed_asset", status=status, seconds=seconds, route=route))
        try:
            # Browsers always offer compression; the first frame carries the client id.
            async with session.ws_connect(base.replace("http", "ws", 1) + "/ws", headers=headers, compress=15) as ws:
                message = await ws.receive(timeout=10)
                kind = json.loads(message.data).get("type") if message.type == WSMsgType.TEXT else message.type.name
        except Exception as exc:
            kind = type(exc).__name__
        failures += kind != "status"
        print(t("diagnosis.websocket", mark="OK" if kind == "status" else "NG", kind=kind))
    return failures


async def diagnose(config):
    """Find the hop where the UI stops loading: ComfyUI, the auth proxy, or the tunnel."""
    if "mode" not in config:
        raise SystemExit(t("diagnosis.not_running"))
    auth = {"Cookie": proxy.COOKIE + "=" + config["cookie_secret"]} if config.get("cookie_secret") else None
    targets = [(t("diagnosis.target_comfyui"), "http://127.0.0.1:8188", None)]
    if config["mode"] != "access":  # There the proxy answers only what Cloudflare Access has signed.
        targets.append((t("diagnosis.target_proxy"), "http://127.0.0.1:8189", auth))
    if config.get("remote_url"):
        targets.append((t("diagnosis.target_public"), config["remote_url"], auth))
    for label, base, headers in targets:
        print("== " + label, flush=True)
        if await probe_frontend(base, headers):
            print(t("diagnosis.failed_at", label=label))
            return label
    # Colab's forwarding and Cloudflare Access only answer a signed-in browser, so they cannot be tried from here.
    if config["mode"] == "colab":
        print(t("diagnosis.colab_ok"))
    elif config["mode"] == "access":
        print(t("diagnosis.access_ok"))
    else:
        print(t("diagnosis.all_ok"))
    return None


def isolation_report():
    """List the local services the application user can reach. Colab runs its own services as
    root; one that answers without authentication would let a custom node leave its account."""
    import socket
    services = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        for line in Path(table).read_text().splitlines()[1:]:
            fields = line.split()
            address, port = fields[1].rsplit(":", 1)
            if fields[3] != "0A":  # Not listening.
                continue
            host = socket.inet_ntoa(bytes.fromhex(address)[::-1]) if len(address) == 8 else "127.0.0.1"
            services.add(("127.0.0.1" if host == "0.0.0.0" else host, int(port, 16)))
    for name, value in sorted(os.environ.items()):
        match = re.fullmatch(r"(\d+\.\d+\.\d+\.\d+):(\d+)", value)
        if match and re.search(r"CRED|TBE|METADATA", name):
            services.add((match.group(1), int(match.group(2))))
    services.add(("169.254.169.254", 80))
    # Unix sockets that accept connections: by path, or "@name" in the abstract namespace.
    sockets = sorted({fields[7] for fields in map(str.split, Path("/proc/net/unix").read_text().splitlines()[1:])
                      if len(fields) > 7 and fields[3] == "00010000"})
    probe = ("import json, socket, sys\n"
             "def ask(host, port, path):\n"
             "    with socket.create_connection((host, port), 3) as s:\n"
             "        try:\n"
             "            s.sendall(('GET ' + path + ' HTTP/1.0\\r\\nHost: localhost\\r\\n\\r\\n').encode())\n"
             "            return s.recv(64).split(b'\\r\\n')[0].decode('latin-1')\n"
             "        except OSError:\n"
             "            return ''\n"
             "tcp, unix = json.load(sys.stdin)\n"
             "for host, port in tcp:\n"
             "    try:\n"
             "        answer = ask(host, port, '/') or 'connected, no HTTP reply'\n"
             "    except OSError as exc:\n"
             "        answer = 'unreachable (' + type(exc).__name__ + ')'\n"
             "    if answer.startswith('HTTP/'):\n"
             "        try:\n"
             "            answer += ' | /api/kernels: ' + ask(host, port, '/api/kernels').partition(' ')[2]\n"
             "        except OSError:\n"
             "            pass\n"
             "    print(f'tcp\\t{host}:{port}\\t{answer}', flush=True)\n"
             "for name in unix:\n"
             "    s = socket.socket(socket.AF_UNIX)\n"
             "    s.settimeout(3)\n"
             "    try:\n"
             "        s.connect('\\0' + name[1:] if name.startswith('@') else name)\n"
             "        print(f'unix\\t{name}\\tconnected', flush=True)\n"
             "    except OSError:\n"
             "        pass\n"
             "    s.close()\n")
    result = subprocess.run(commands.app_command([sys.executable, "-c", probe]), env=commands.application_environment(),
                            input=json.dumps([sorted(services), sockets]), capture_output=True, text=True, timeout=600)
    owners = subprocess.run(["ss", "-H", "-ltnp"], capture_output=True, text=True).stdout if shutil.which("ss") else ""
    print(t("isolation.heading", user=commands.APP_USER))
    reachable = []
    for line in result.stdout.splitlines():
        kind, _, rest = line.partition("\t")
        address, _, answer = rest.partition("\t")
        if kind == "unix":
            reachable.append(address)
            continue
        port = address.rsplit(":", 1)[1]
        owner = sorted(set(re.findall(r'\(\("([^"]+)"', " ".join(
            row for row in owners.splitlines() if row.split()[3].rsplit(":", 1)[-1] == port))))
        note = {"8188": t("isolation.comfyui"), "8189": t("isolation.proxy")}.get(port, ", ".join(owner) or t("common.unknown"))
        print(f"  {address:<24} {answer:<56} {note}")
    print(t("isolation.unix_sockets", listening=len(sockets), reachable=len(reachable)), ", ".join(reachable) or t("common.none"))
    import ctypes
    abi = ctypes.CDLL(None, use_errno=True).syscall(444, None, 0, 1)  # landlock_create_ruleset(version)
    lsm = Path("/sys/kernel/security/lsm")
    print(t("isolation.kernel"), os.uname().release, "| Landlock ABI:", abi if abi > 0 else t("common.none"),
          "| LSM:", lsm.read_text().strip() if lsm.exists() else t("common.unknown"))
    effective = int(re.search(r"CapEff:\s*([0-9a-f]+)", Path("/proc/self/status").read_text()).group(1), 16)
    print(t("isolation.root_capabilities"), ", ".join(f"{name}={t('common.yes') if effective >> bit & 1 else t('common.no')}" for name, bit in
                               (("NET_ADMIN", 12), ("SYS_ADMIN", 21), ("SETUID", 7), ("SYS_PTRACE", 19))))

    def usable(command):
        if not shutil.which(command[0]):
            return t("isolation.no_command")
        return t("isolation.available") if not subprocess.run(command, capture_output=True).returncode else t("isolation.unavailable")
    print("iptables:", usable(["iptables", "-S", "OUTPUT"]), "| network namespace:", usable(["unshare", "--net", "true"]),
          "| mount namespace:", usable(["unshare", "--mount", "true"]), "| ip:", t("common.present") if shutil.which("ip") else t("common.none"),
          "| /dev/net/tun:", t("common.present") if Path("/dev/net/tun").exists() else t("common.none"))
