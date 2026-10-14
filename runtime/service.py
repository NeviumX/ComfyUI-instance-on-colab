"""Starting and stopping ComfyUI with its proxy and tunnel (cells 6 and 10)."""

import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import threading
import time
from urllib.parse import urlsplit

from . import app_tree, cloudflare, commands, environment, filesystem, pairing, paths, stop_button
from .messages import t


# What a session adds to the saved settings, and stopping takes away again.
SESSION_KEYS = ("mode", "cookie_secret", "remote_url", "origins", "access")


def load_config():
    """The settings setup saved for this runtime, with those of the running session if any."""
    return json.loads(paths.CONFIG.read_text())


def run_diagnosis():
    """diagnose() as a process of its own (cell 7), with its output shown in the notebook."""
    commands.run(commands.own_command("diagnose"), cwd=commands.OWN_DIRECTORY, timeout=600)


def backend_command(config):
    repo = paths.BASE / "environment/ComfyUI"
    return [str(paths.BASE / ".venv/bin/python"), str(repo / "main.py"),
            "--listen", "127.0.0.1", "--port", "8188", "--enable-manager",
            "--base-directory", str(repo), "--user-directory", str(repo / "user"),
            "--input-directory", str(paths.BASE / "input"), "--output-directory", str(paths.BASE / "output"),
            "--temp-directory", str(paths.BASE / "temp"), "--extra-model-paths-config", str(paths.BASE / "models.yaml"),
            "--database-url", "sqlite:///" + str(paths.BASE / "database/comfyui.db"),
            # Only where setup installed and tried it: a runtime set up before that has no package.
            *(["--use-sage-attention"] if config.get("sage_attention") else []),
            # One named optimization: --fast alone turns on every experimental one ComfyUI has.
            *(["--fast", "fp16_accumulation"] if config.get("fp16_accumulation") else [])]


def supervise(config):
    # Background process persists independently of the notebook cell.
    def save_loop():
        while True:
            time.sleep(config["checkpoint_seconds"])
            try:
                environment.checkpoint(config)
            except Exception as exc:
                filesystem.private_write(paths.PRIVATE / "checkpoint-status.json", json.dumps({"ok": False, "error": str(exc)}))
                print(t("supervisor.autosave_failed"), flush=True)
    threading.Thread(target=save_loop, daemon=True).start()
    env = commands.application_environment()
    for _ in range(5):
        with (paths.PRIVATE / "comfy.log").open("a") as log:
            child = subprocess.Popen(commands.app_command(backend_command(config)), cwd=paths.BASE / "environment/ComfyUI", env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
            child.wait()
        try:
            environment.checkpoint(config, force=True)
        except Exception:
            print(t("supervisor.save_before_restart_failed"), flush=True)
        print(t("supervisor.restarting"), flush=True)
        time.sleep(10)
    print(t("supervisor.restart_limit"), flush=True)


def redact(text):
    """Hide credentials and the tunnel address in text that is about to be displayed."""
    text = re.sub(r"(?i)(access_key=|Bearer\s+)[A-Za-z0-9_.-]+", r"\1[REDACTED]", text)
    text = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", text)
    text = re.sub(r"eyJ[A-Za-z0-9_+/=-]{20,}(\.[A-Za-z0-9_+/=-]+)*", "[REDACTED]", text)  # Tunnel and Access tokens.
    return re.sub(r"[a-z0-9-]+\.trycloudflare\.com", "[REDACTED].trycloudflare.com", text)


def colab_forwarded_address(port):
    """Where Colab serves a port of this runtime, to the browser signed in as its owner only."""
    from google.colab import output
    parts = urlsplit(str(output.eval_js(f"google.colab.kernel.proxyPort({port})")))
    if parts.scheme != "https" or not parts.netloc:
        raise RuntimeError(t("start.colab_unavailable"))
    return f"https://{parts.netloc}/"


def start_with_access(config, hostname, audience, tunnel_token):
    """Start behind a named Cloudflare tunnel whose hostname Cloudflare Access protects."""
    access = cloudflare.access_settings(hostname, audience)
    token = (tunnel_token or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9+/=_-]{40,4096}", token):
        raise ValueError(t("access.invalid_token"))
    if (paths.PRIVATE / "processes.json").exists():
        raise RuntimeError(t("start.already_running"))
    access["team"] = cloudflare.access_team_domain(access["hostname"])
    start(config, "access", access=access, tunnel_token=token)


def start(config, mode, access=None, tunnel_token=""):
    """Start ComfyUI and make it reachable: mode "colab" through Colab's port forwarding,
    "tunnel" through a Quick Tunnel that asks each browser for a pairing code, "access"
    through a named tunnel on a hostname that Cloudflare Access protects."""
    import requests
    if mode not in ("colab", "tunnel", "access") or (mode == "access") != bool(access and tunnel_token):
        raise ValueError(mode)
    if (paths.PRIVATE / "processes.json").exists():
        raise RuntimeError(t("start.already_running"))
    environment.refuse_unfinished_update()
    for name in SESSION_KEYS:
        config.pop(name, None)
    for name in ("stop-request.json", "stop-status.json"):
        (paths.PRIVATE / name).unlink(missing_ok=True)
    config["mode"] = mode
    if mode == "colab":
        address = colab_forwarded_address(8189)
        config["origins"] = [address.rstrip("/")]
    elif mode == "tunnel":
        # A cookie from an earlier start (browser, Desktop) stops working.
        config["cookie_secret"] = secrets.token_urlsafe(32)
    else:
        config["access"] = access  # Not the tunnel's token: only cloudflared is given that.
    filesystem.private_write(paths.CONFIG, json.dumps(config))
    processes = {}

    def launch(name):
        with (paths.PRIVATE / (name + ".log")).open("a") as log:
            process = subprocess.Popen(commands.own_command(name.replace("supervisor", "supervise")),
                                       cwd=commands.OWN_DIRECTORY, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        processes[name] = process.pid
        filesystem.private_write(paths.PRIVATE / "processes.json", json.dumps(processes))
        return process
    try:
        # A process left behind by a custom node could take the proxy's port and receive the tunnel.
        commands.stop_app_processes()
        proxy = launch("proxy")
        for _ in range(30):
            if proxy.poll() is not None:
                raise RuntimeError(t("start.proxy_exited_port"))
            try:
                # Its answer before ComfyUI runs: no cookie behind the tunnel, no backend behind Colab.
                if requests.get("http://127.0.0.1:8189/", timeout=3).status_code == (502 if mode == "colab" else 401):
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise RuntimeError(t("start.proxy_not_started"))
        launch("supervisor")
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            try:
                if requests.get("http://127.0.0.1:8188/system_stats", timeout=3).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            raise RuntimeError(t("start.comfyui_timeout"))
        if proxy.poll() is not None:
            raise RuntimeError(t("start.proxy_exited"))
        if mode == "colab":
            print(t("start.colab_heading"))
            print(address)
            print(t("start.colab_note"))
            stop_button.follow_stop_requests(config)
            return
        logfile = paths.PRIVATE / "tunnel.log"
        logfile.write_text("")
        with logfile.open("a") as log:
            if mode == "access":
                # The token goes in the environment: a command line can be read by every user here.
                tunnel = subprocess.Popen([str(paths.PRIVATE / "cloudflared"), "tunnel", "--no-autoupdate",
                                           "--protocol", cloudflare.TUNNEL_PROTOCOL, "run"],
                                          env={**os.environ, "TUNNEL_TOKEN": tunnel_token},
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            else:
                tunnel = subprocess.Popen([str(paths.PRIVATE / "cloudflared"), "tunnel", "--no-autoupdate",
                        "--protocol", cloudflare.TUNNEL_PROTOCOL, "--url", "http://127.0.0.1:8189"],
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes["tunnel"] = tunnel.pid
        filesystem.private_write(paths.PRIVATE / "processes.json", json.dumps(processes))
        if mode == "access":
            services = connected = None
            for second in range(60):
                log = logfile.read_text()
                services = cloudflare.tunnel_services(log)
                if connected is None and "Registered tunnel connection" in log:
                    connected = second
                # Cloudflare sends the tunnel its settings right after it connects.
                if services is not None or (connected is not None and second - connected >= 15):
                    break
                if tunnel.poll() is not None:
                    raise RuntimeError(t("access.tunnel_exited"))
                time.sleep(1)
            if services is None:
                print(t("access.target_unknown"))
            elif not cloudflare.tunnel_leads_to_proxy(services):
                raise RuntimeError(t("access.wrong_target", services=", ".join(services)[:300]))
            print(t("access.heading"))
            print("https://" + access["hostname"] + "/")
            print(t("access.note"))
            print(t("start.url_auto_download"))
            stop_button.follow_stop_requests(config)
            return
        for _ in range(90):
            match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", logfile.read_text())
            if match:
                url = match.group(0)
                config["remote_url"] = url
                filesystem.private_write(paths.CONFIG, json.dumps(config))
                pairing.show_pairing(config)
                print(t("start.url_auto_download"))
                stop_button.follow_stop_requests(config)
                return
            if tunnel.poll() is not None:
                raise RuntimeError(t("start.tunnel_exited"))
            time.sleep(1)
        raise RuntimeError(t("start.no_tunnel_url"))
    except Exception:
        stop(config, save=False)
        raise


def stop(config, save=True):
    if save:
        environment.checkpoint(config, force=True)  # A failed save must not look like successful shutdown.
        print(t("stop.saved"))
    file = paths.PRIVATE / "processes.json"
    if file.exists():
        for pid in reversed(list(json.loads(file.read_text()).values())):
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for pid in json.loads(file.read_text()).values():
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                stat = Path(f"/proc/{pid}/stat")
                if not stat.exists():
                    break
                try:
                    if stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                        break
                except FileNotFoundError:
                    break
                time.sleep(0.1)
            else:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        file.unlink()
    commands.stop_app_processes()
    for watch in stop_button.STOP_WATCH:
        watch.set()
    for name in ("pairing.json", "stop-request.json", "stop-status.json"):
        (paths.PRIVATE / name).unlink(missing_ok=True)
    for name in SESSION_KEYS:
        config.pop(name, None)
    filesystem.private_write(paths.CONFIG, json.dumps(config))
    print(t("stop.done"))


def zip_outputs(destination="/content/comfy-output.zip"):
    try:
        with open(destination, "wb") as file:
            app_tree.app_action("zip-output", stdout=file, limit=1024 ** 4)
    except Exception:
        Path(destination).unlink(missing_ok=True)
        raise
    return str(destination)
