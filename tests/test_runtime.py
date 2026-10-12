import asyncio
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile

import nbformat
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import colab_runtime as runtime


@pytest.mark.parametrize("value", ["../secret", "/root/key", "models/../secret", "C:\\key", "a\\b", "./a", "a\x00b"])
def test_reject_escaping_paths(value):
    with pytest.raises(ValueError):
        runtime.safe_relative(value)


@pytest.mark.parametrize("member_type,name,link", [(tarfile.REGTYPE, "../secret", ""),
    (tarfile.SYMTYPE, "ComfyUI/custom_nodes/key", "/content/.comfy-private/rclone.conf"),
    (tarfile.LNKTYPE, "ComfyUI/custom_nodes/key", "../../key"),
    (tarfile.FIFOTYPE, "ComfyUI/fifo", "")])
def test_reject_unsafe_archive(tmp_path, member_type, name, link):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        member = tarfile.TarInfo(name)
        member.type = member_type
        member.linkname = link
        tf.addfile(member)
    dest = tmp_path / "restored"
    dest.mkdir()
    with pytest.raises(ValueError):
        runtime.restore_archive(archive, dest)
    assert not list(dest.iterdir())


def test_snapshot_preserves_node_models_code_and_user_data(tmp_path):
    env = tmp_path / "environment"
    expected = ["ComfyUI/custom_nodes/node/models/model.py", "ComfyUI/custom_nodes/node/.git/config",
                "ComfyUI/user/workflows/workflow.json", "requirements.lock.txt"]
    excluded = ["ComfyUI/models/checkpoints/large.bin", "ComfyUI/output/video.mp4",
                "ComfyUI/custom_nodes/node/__pycache__/a.pyc"]
    for name in expected + excluded:
        path = env / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    archive = tmp_path / "environment.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for child in env.iterdir():
            tf.add(child, arcname=child.name, filter=runtime.archive_filter)
    dest = tmp_path / "restored"
    dest.mkdir()
    runtime.restore_archive(archive, dest)
    for name in expected:
        assert (dest / name).read_text() == name
    for name in excluded:
        assert not (dest / name).exists()


ROOT = Path(__file__).resolve().parents[1]


def embedded_runtime(nb):
    import ast
    cell = next(c.source for c in nb.cells if c.cell_type == "code" and "write_text(" in c.source)
    write_call = next(node for node in ast.walk(ast.parse(cell)) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute) and node.func.attr == "write_text")
    return ast.literal_eval(write_call.args[0])


@pytest.mark.parametrize("language", ["JP", "EN"])
def test_notebook_is_valid_self_contained_and_compilable(language):
    import build_notebook
    path = ROOT / f"ComfyUI_instance_on_Colab_{language}.ipynb"
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    # The committed notebook is exactly what the builder produces from the current sources.
    assert json.loads(path.read_text(encoding="utf-8")) == build_notebook.build(language)
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
            compile(cell.source, "notebook-cell", "exec")
    assert embedded_runtime(nb) == build_notebook.runtime_source(language)
    source ="\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "drive.mount(" in source
    assert "GDRIVE_OAUTH" not in source
    assert "MODEL_URL" in source and "runtime.fetch_model(" in source
    # Optional and destructive steps are checkboxes that start unchecked, shown as a form.
    for switch in ("USE_LATEST_COMFYUI", "DOWNLOAD_NOW", "REFRESH_MODELS", "SAVE_NOW", "DOWNLOAD_OUTPUT_ZIP",
                   "STOP_NOW", "UPDATE_COMFYUI_NOW"):
        assert switch + ' = False #@param {type:"boolean"}' in source
    assert all(c.metadata.get("cellView") == "form" for c in nb.cells if c.cell_type == "code")
    text = "\n".join(c.source for c in nb.cells)
    assert nb.cells[0].source.startswith("# ComfyUI instance on Colab\n")
    assert bool(build_notebook.JAPANESE.search(text)) == (language == "JP")
    prose = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "C:\\" not in prose and "D:\\" not in prose and "docs/" not in prose


def test_both_notebooks_have_the_same_cells_and_code():
    import build_notebook
    japanese, english = (build_notebook.build(language)["cells"] for language in ("JP", "EN"))
    assert [c["cell_type"] for c in japanese] == [c["cell_type"] for c in english]

    def statements(cell):
        # What runs, without the text shown to the reader.
        import ast
        tree = ast.parse(cell["source"])
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = ""
        return ast.dump(tree)
    for left, right in zip(japanese, english):
        if left["cell_type"] == "code" and "write_text(" not in left["source"]:
            assert statements(left) == statements(right)
            assert left["source"].count("#@param") == right["source"].count("#@param")


def test_notebook_texts_come_from_the_translation_file(monkeypatch, tmp_path):
    import build_notebook
    texts = build_notebook.translations()["notebook"]
    lines = 0
    for section, entry in texts.items():
        # A markdown cell is its lines per language; a code cell is one entry per text.
        for name, text in ([(section, entry)] if "JP" in entry else entry.items()):
            assert set(text) == {"JP", "EN"}, name
            assert type(text["JP"]) is type(text["EN"]) and len(text["JP"]) > 0 and len(text["EN"]) > 0, name
            english = text["EN"] if isinstance(text["EN"], str) else "\n".join(text["EN"])
            assert not build_notebook.JAPANESE.search(english), name
            lines += 1
        assert ("JP" in entry) or "title" in entry, section
    assert lines > 30
    # The builder holds no text of its own: its only Japanese is the pattern that detects it.
    builder = (ROOT / "build_notebook.py").read_text(encoding="utf-8")
    assert [line.split(" = ")[0] for line in builder.splitlines() if build_notebook.JAPANESE.search(line)] == ["JAPANESE"]
    # Every text reaches the notebook, and a text no cell refers to is refused.
    for language in ("JP", "EN"):
        notebook = "\n".join(cell["source"] for cell in build_notebook.build(language)["cells"])
        for section, entry in texts.items():
            for text in ([entry] if "JP" in entry else entry.values()):
                shown = text[language] if isinstance(text[language], str) else "\n".join(text[language])
                assert shown in notebook, section
    changed = build_notebook.translations()
    changed["notebook"]["stop"]["title"]["EN"] = "10. Finish"
    changed["notebook"]["stop"]["skipped"]["EN"] = 'Check "STOP_NOW" to finish.'
    changed["notebook"]["save"]["leftover"] = {"JP": "未使用", "EN": "Unused"}
    copy = tmp_path / "translations.json"
    copy.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(build_notebook, "TRANSLATIONS", copy)
    with pytest.raises(ValueError, match="save.leftover"):
        build_notebook.build("EN")
    del changed["notebook"]["save"]["leftover"]
    copy.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
    def stop_cell(language):
        return next(cell["source"] for cell in build_notebook.build(language)["cells"] if "STOP_NOW = False" in cell["source"])
    stop = stop_cell("EN")
    assert stop.startswith("#@title 10. Finish\n")
    compile(stop, "notebook-cell", "exec")  # A quotation mark in a message stays inside its string.
    assert 'print("Check \\"STOP_NOW\\" to finish.")' in stop
    assert stop_cell("JP").startswith("#@title 10. 保存して停止\n")
    changed["notebook"]["stop"]["unmount"]["EN"] = "Two\nlines"  # Would end the form description early.
    copy.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="stop.unmount must be a single line"):
        build_notebook.build("EN")


def test_runtime_messages_come_from_the_translation_file():
    import ast
    import build_notebook
    import copy
    source = (ROOT / "colab_runtime.py").read_text(encoding="utf-8")
    sections = build_notebook.translations()["runtime"]
    build_notebook.check_runtime(source, sections)
    # The runtime holds no text of its own; run from the source tree it speaks Japanese.
    assert not build_notebook.JAPANESE.search(source)
    assert runtime.MESSAGES == build_notebook.runtime_messages("JP")
    assert len(runtime.MESSAGES) > 100
    assert runtime.t("command.exit_code", stage="apt-get", code=3) == "apt-get が終了コード 3 で失敗しました。直前の出力を確認してください。"
    # An embedded runtime is the source with the messages of its language on that one line.
    placeholder = build_notebook.MESSAGES_LINE.search(source).group(0)
    for language in ("JP", "EN"):
        embedded = build_notebook.runtime_source(language)
        begin = embedded.index("MESSAGES = {\n")
        finish = embedded.index("\n}\n", begin) + 2
        assert ast.literal_eval(embedded[begin + len("MESSAGES = "):finish]) == build_notebook.runtime_messages(language)
        assert embedded[:begin] + placeholder + embedded[finish:] == source

    def refused(change, match, error=ValueError):
        changed = copy.deepcopy(sections)
        change(changed)
        with pytest.raises(error, match=match):
            build_notebook.check_runtime(source, changed)
    # A translation that drops a value, formats it differently or shows one the code does not give.
    refused(lambda s: s["command"]["exit_code"].update(EN="{stage} failed."), "command.exit_code shows different values")
    refused(lambda s: s["command"]["done"].update(EN="Done: {stage} ({seconds:.1f}s)"), "command.done shows different values")
    refused(lambda s: s["command"]["exit_code"].update(JP="{stage} {status}", EN="{stage} {status}"),
            "command.exit_code does not show exactly the values given to it: code, stage")
    # A message the code refers to is missing, in one language or both; a message nothing refers to.
    refused(lambda s: s["stop"].pop("done"), "stop.done", KeyError)
    refused(lambda s: s["stop"]["done"].pop("EN"), "EN", KeyError)
    refused(lambda s: s["stop"].update(leftover={"JP": "未使用", "EN": "Unused"}), "does not use: stop.leftover")
    refused(lambda s: s["stop"]["done"].update(EN="停止しました。"), "stop.done has Japanese in its English text")
    # Text written into the code, or a key the build cannot see.
    with pytest.raises(ValueError, match="holds Japanese text"):
        build_notebook.check_runtime(source + 'print("停止しました。")\n', sections)
    with pytest.raises(ValueError, match="takes a literal key and named values"):
        build_notebook.check_runtime(source + 'print(t("stop." + "done"))\n', sections)


def test_embedded_runtime_needs_no_other_file_and_reports_in_its_language(tmp_path, monkeypatch, capsys):
    import build_notebook
    import importlib.util
    from types import SimpleNamespace

    def embedded(language):
        # Alone in a folder, as on Colab: no translations.json next to it.
        module_path = tmp_path / language / "comfy_colab_runtime.py"
        module_path.parent.mkdir()
        module_path.write_text(build_notebook.runtime_source(language), encoding="utf-8")
        spec = importlib.util.spec_from_file_location("comfy_colab_runtime_" + language, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=3, stdout="", stderr="boom"))
        return module

    english = embedded("EN")
    with pytest.raises(ValueError, match="Specify a safe relative path"):
        english.safe_relative("../secret")
    with pytest.raises(ValueError, match="For Civitai, give the URL"):
        english.parse_model_url("https://civitai.com/models/12345/name")
    with pytest.raises(RuntimeError, match="lines other than pinned versions .* Replace them with regular pip packages"):
        english.checked_lock("-e git+https://example.com/x.git#egg=x\n")
    with pytest.raises(RuntimeError, match="apt-get failed with exit code 3"):
        english.run(["apt-get", "update"])
    assert "Running: apt-get" in capsys.readouterr().out
    assert english.redact("https://quick-tunnel.trycloudflare.com/?access_key=abc") == "https://[REDACTED].trycloudflare.com/?access_key=[REDACTED]"

    japanese = embedded("JP")
    with pytest.raises(ValueError, match="安全な相対パスを指定してください"):
        japanese.safe_relative("../secret")
    with pytest.raises(RuntimeError, match="apt-get が終了コード 3 で失敗しました"):
        japanese.run(["apt-get", "update"])
    assert "実行中: apt-get" in capsys.readouterr().out



def test_proxy_relays_websocket_for_browser_offering_compression():
    from aiohttp import web, ClientSession, CookieJar
    from aiohttp.test_utils import TestServer

    async def scenario():
        handshakes = []
        status = json.dumps({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 0}}, "sid": "sid"}})

        async def backend(request):
            # ComfyUI negotiates compression by default and pushes status as soon as the socket opens.
            handshakes.append(dict(request.headers))
            ws = web.WebSocketResponse(protocols=("comfy",))
            await ws.prepare(request)
            await ws.send_str(status)
            async for msg in ws:
                await ws.send_str(msg.data)
            return ws

        app = web.Application()
        app.router.add_get("/ws", backend)
        async with TestServer(app) as upstream:
            config = {"mode": "tunnel", "cookie_secret": "test-cookie-secret"}
            proxy = runtime.create_proxy(config, str(upstream.make_url("/")).rstrip("/"), secure_cookie=False)
            async with TestServer(proxy) as server:
                async with ClientSession(cookie_jar=CookieJar(unsafe=True)) as client:
                    signed_in(client, server)
                    # Chromium (and so Comfy Desktop) always offers permessage-deflate.
                    async with client.ws_connect(server.make_url("/ws"), compress=15, protocols=("comfy",)) as ws:
                        assert ws.compress and ws.protocol == "comfy"
                        assert (await ws.receive(timeout=3)).data == status
                        await ws.send_str("feature_flags")
                        assert (await ws.receive(timeout=3)).data == "feature_flags"
        assert "Sec-WebSocket-Extensions" not in handshakes[0]
        assert handshakes[0]["Sec-WebSocket-Protocol"] == "comfy"
    asyncio.run(scenario())


def test_proxy_opens_root_page_when_followed_from_another_site():
    from aiohttp import web, ClientSession, CookieJar
    from aiohttp.test_utils import TestServer

    async def scenario():
        async def backend(request):
            # ComfyUI's origin middleware rejects every cross-site request.
            return web.Response(status=403 if request.headers.get("Sec-Fetch-Site") == "cross-site" else 200)

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", backend)
        async with TestServer(app) as upstream:
            config = {"mode": "tunnel", "cookie_secret": "test-cookie-secret"}
            proxy = runtime.create_proxy(config, str(upstream.make_url("/")).rstrip("/"), secure_cookie=False)
            async with TestServer(proxy) as server:
                async with ClientSession(cookie_jar=CookieJar(unsafe=True)) as client:
                    signed_in(client, server)
                    # A link in Colab's output or a bookmark opened from another page.
                    link = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document"}
                    async with client.get(server.make_url("/"), headers=link) as res:
                        assert res.status == 200 and res.url.path_qs == "/"
                    for method, route in (("GET", "/api/manager/reboot"), ("POST", "/")):
                        async with client.request(method, server.make_url(route), headers=link) as res:
                            assert res.status == 403
    asyncio.run(scenario())


@pytest.mark.parametrize("broken,expected", [(None, 0), ("asset", 1), ("api", 1), ("websocket", 1)])
def test_diagnosis_counts_what_blocks_the_frontend_splash(capsys, broken, expected):
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def scenario():
        async def backend(request):
            if request.path == "/ws":
                ws = web.WebSocketResponse()
                await ws.prepare(request)
                await ws.send_str(json.dumps({"type": "crash" if broken == "websocket" else "status"}))
                return ws
            if request.path == "/":
                return web.Response(text='<script type="module" src="./assets/index.js"></script>', content_type="text/html")
            if request.path == "/api/extensions":
                return web.json_response(["/extensions/core/node.js"])
            if request.path == "/api/userdata":
                return web.Response(status=404)  # No saved workflows yet.
            if request.path == {"asset": "/assets/index.js", "api": "/api/settings"}.get(broken):
                return web.Response(status=500)
            return web.json_response({})

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", backend)
        async with TestServer(app) as upstream:
            config = {"mode": "tunnel", "cookie_secret": "test-cookie-secret"}
            proxy = runtime.create_proxy(config, str(upstream.make_url("/")).rstrip("/"), secure_cookie=False)
            async with TestServer(proxy) as server:
                base = str(server.make_url("/")).rstrip("/")
                assert await runtime.probe_frontend(base) == len(runtime.STARTUP_ROUTES) + 1  # No cookie: all 401.
                return await runtime.probe_frontend(base, {"Cookie": "comfy_colab=test-cookie-secret"})
    assert asyncio.run(scenario()) == expected
    report = capsys.readouterr().out
    assert "画面のJS/CSS 2件" in report and "test-cookie-secret" not in report



def test_start_refuses_to_open_a_tunnel_when_the_proxy_is_not_the_one_listening(tmp_path, monkeypatch):
    from types import SimpleNamespace
    spawned = []

    class Process:
        pid = 4242

        def __init__(self, args, **kwargs):
            spawned.append(args[-1])

        def poll(self):
            return 1  # The port was taken: the real proxy exited at once.

    # Whatever took the port answers the way the proxy would.
    monkeypatch.setattr(runtime, "PRIVATE", tmp_path)
    monkeypatch.setattr(runtime, "CONFIG", tmp_path / "session.json")
    monkeypatch.setattr(runtime.subprocess, "Popen", Process)
    monkeypatch.setattr(runtime.os, "killpg", lambda pid, sig: None, raising=False)
    monkeypatch.setattr(runtime.Path, "exists", lambda self: False)
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(
        get=lambda url, timeout: SimpleNamespace(ok=True, status_code=401), RequestException=OSError))
    with pytest.raises(RuntimeError, match="認証プロキシが終了しました"):
        runtime.start({"drive_folder": "ComfyUI-Colab"}, "tunnel")
    assert spawned == ["proxy"]


def test_saved_apt_packages_are_reported_not_installed():
    saved = {"apt_packages": ["libgl1", "openssh-server", "bad name;rm", 7, "aria2"]}
    assert runtime.unlisted_apt_packages(saved, ["aria2"]) == ["libgl1", "openssh-server"]
    assert runtime.unlisted_apt_packages({}, []) == [] and runtime.unlisted_apt_packages(["not", "a", "dict"], []) == []


def test_displayed_logs_hide_keys_tokens_and_the_tunnel_address():
    log = ("INF |  https://quick-test-tunnel.trycloudflare.com  |\n"
           "GET /?access_key=abc_DEF-123 Authorization: Bearer token.value hf_AbCdEf123")
    shown = runtime.redact(log)
    for secret in ("quick-test-tunnel", "abc_DEF-123", "token.value", "hf_AbCdEf123"):
        assert secret not in shown
    assert "trycloudflare.com" in shown and "access_key=[REDACTED]" in shown


def test_proxy_compresses_text_caches_hashed_files_and_rejects_odd_credentials():
    from aiohttp import web, ClientSession, CookieJar
    from aiohttp.test_utils import TestServer

    async def scenario():
        script = ("export const value = 'frontend chunk';\n" * 4000).encode()
        image = bytes(range(256)) * 64

        async def backend(request):
            if request.path.endswith(".png"):
                return web.Response(body=image, content_type="image/png")
            if request.path.startswith("/api/"):
                return web.json_response({"nodes": ["KSampler"] * 2000})
            return web.Response(body=script, content_type="application/javascript", charset="utf-8",
                                headers={"Cache-Control": "no-store"})

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", backend)
        async with TestServer(app) as upstream:
            config = {"mode": "tunnel", "cookie_secret": "test-cookie-secret"}
            proxy = runtime.create_proxy(config, str(upstream.make_url("/")).rstrip("/"), secure_cookie=False)
            async with TestServer(proxy) as server:
                async with ClientSession() as stranger:
                    # Anyone can send these; they must be refused, not crash the comparison.
                    async with stranger.get(server.make_url("/"), headers={"Cookie": "comfy_colab=\u00e3\u0081\u0082"}) as res:
                        assert res.status == 401
                async with ClientSession(cookie_jar=CookieJar(unsafe=True), auto_decompress=False) as client:
                    signed_in(client, server)
                    gzip_ok = {"Accept-Encoding": "gzip, br"}
                    async with client.get(server.make_url("/assets/index-CrctFnrp.js"), headers=gzip_ok) as res:
                        packed = await res.read()
                        assert res.headers["Content-Encoding"] == "gzip" and len(packed) < len(script) / 10
                        assert res.headers["Cache-Control"] == "private, max-age=31536000, immutable"
                        import gzip
                        assert gzip.decompress(packed) == script
                    async with client.get(server.make_url("/extensions/core/node.js"), headers=gzip_ok) as res:
                        assert res.headers["Content-Encoding"] == "gzip" and res.headers["Cache-Control"] == "no-store"
                    async with client.get(server.make_url("/api/object_info"), headers=gzip_ok) as res:
                        assert res.headers["Content-Encoding"] == "gzip" and res.headers["Cache-Control"] == "no-store"
                    # Generated images stay byte-exact and private; a client without gzip gets plain text.
                    async with client.get(server.make_url("/api/view/image-AbCdEf12.png"), headers=gzip_ok) as res:
                        assert "Content-Encoding" not in res.headers and await res.read() == image
                        assert res.headers["Cache-Control"] == "no-store"
                    async with client.get(server.make_url("/assets/index-CrctFnrp.js"), headers={"Accept-Encoding": "identity"}) as res:
                        assert "Content-Encoding" not in res.headers and await res.read() == script
                    async with client.head(server.make_url("/assets/index-CrctFnrp.js"), headers=gzip_ok) as res:
                        assert "Content-Encoding" not in res.headers and int(res.headers["Content-Length"]) == len(script)
    asyncio.run(scenario())


@pytest.mark.parametrize("line", ["--extra-index-url https://attacker.example/simple", "-e git+https://example.com/x.git#egg=x",
                                  "package @ file:///content/local", "-r /content/.comfy-private/session.json",
                                  "package==1.0 --hash=sha256:00", "package>=1.0"])
def test_lock_with_pip_options_or_local_paths_is_refused(line):
    with pytest.raises(RuntimeError, match="固定バージョン以外"):
        runtime.checked_lock("aiohttp==3.14.4\n" + line + "\n")


def test_lock_accepts_what_pip_freeze_writes_for_pinned_packages():
    text = ("aiohttp==3.14.4\ntorch==2.11.0+cu130\nruamel.yaml==0.18.6\n"
            "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git@6fdee8f\n")
    assert runtime.checked_lock(text) == text


def test_cloudflared_is_pinned_and_refused_when_the_download_differs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    requested = []

    class Download:
        ok = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_content(self, size):
            yield b"not the published binary"

    def get(url, **kwargs):
        requested.append(url)
        return Download()

    monkeypatch.setattr(runtime, "PRIVATE", tmp_path)
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=get))
    with pytest.raises(RuntimeError, match="SHA256"):
        runtime.install_cloudflared()
    assert requested == [f"https://github.com/cloudflare/cloudflared/releases/download/{runtime.CLOUDFLARED_VERSION}/cloudflared-linux-amd64"]
    assert not list(tmp_path.iterdir())
    import hashlib
    monkeypatch.setattr(runtime, "CLOUDFLARED_SHA256", hashlib.sha256(b"not the published binary").hexdigest())
    runtime.install_cloudflared()
    assert (tmp_path / "cloudflared").read_bytes() == b"not the published binary"



@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "BASE", tmp_path / "session")
    monkeypatch.setattr(runtime, "PRIVATE", tmp_path / "private")
    monkeypatch.setattr(runtime, "DRIVE_MOUNT", runtime.PRIVATE / "drive")
    (runtime.DRIVE_MOUNT / "MyDrive" / "ComfyUI-Colab" / "models").mkdir(parents=True)
    (runtime.BASE / "models").mkdir(parents=True)
    monkeypatch.setattr(runtime, "run", lambda *a, **k: None)
    return {"drive_folder": "ComfyUI-Colab"}


def test_manual_models_refresh_and_manager_downloads_preserve_drive(storage):
    models = runtime.drive_root(storage) / "models"
    (models / "diffusion_models").mkdir()
    manual = models / "diffusion_models/manual.safetensors"
    manual.write_bytes(b"manually uploaded model")
    (models / "incomplete.part").write_bytes(b"partial")
    assert runtime.sync_models(storage) == 1
    local = runtime.BASE / "models/diffusion_models/manual.safetensors"
    assert local.read_bytes() == manual.read_bytes()
    assert not (runtime.BASE / "models/incomplete.part").exists()
    assert runtime.sync_models(storage) == 0
    # A stale local model must not overwrite a user's newer Drive upload.
    manual.write_bytes(b"new manually uploaded model")
    new = runtime.BASE / "models/diffusion_models/manager.safetensors"
    new.write_bytes(b"manager model")
    runtime.save_new_cache_models(storage, force=True)
    assert manual.read_bytes() == b"new manually uploaded model"
    assert (models / "diffusion_models/manager.safetensors").read_bytes() == b"manager model"
    assert runtime.sync_models(storage) == 1
    assert local.read_bytes() == manual.read_bytes()
    manual.unlink()
    runtime.save_new_cache_models(storage, force=True)
    assert not manual.exists()  # A removed Drive model is not resurrected by the cache.


def test_model_url_builders():
    assert runtime.model_download_url("Hugging Face", hf_repo="author/model", hf_file="split/model.safetensors", hf_revision="refs/pr/2") == "https://huggingface.co/author/model/resolve/refs%2Fpr%2F2/split/model.safetensors"
    assert runtime.model_download_url("Civitai", civitai_version="123456") == "https://civitai.com/api/download/models/123456"
    for value in ("0", "-1", "123?token=secret"):
        with pytest.raises(ValueError):
            runtime.model_download_url("Civitai", civitai_version=value)
    with pytest.raises(ValueError):
        runtime.model_download_url("Hugging Face", hf_repo="author/model", hf_file="../key")


def fake_requests(monkeypatch, responses):
    from types import SimpleNamespace
    class RequestError(Exception):
        pass
    class Response:
        def __init__(self, status=200, headers=None, data=b"model"):
            self.status_code, self.headers, self.data = status, headers or {}, data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def iter_content(self, *args):
            yield self.data
    class Session:
        def __init__(self):
            self.trust_env = True
            self.calls = []
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return Response(**response)
    session = Session()
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(Session=lambda: session, RequestException=RequestError))
    import socket
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.7" if host == "internal.example" else "93.184.216.34", 443))])
    return session, RequestError


def test_model_download_redirect_strips_token_and_saves_to_drive(storage, monkeypatch):
    session, _ = fake_requests(monkeypatch, [
        {"status": 302, "headers": {"Location": "https://cdn.example/model?signature=secret"}},
        {"headers": {"Content-Type": "application/octet-stream"}, "data": b"downloaded model"}])
    import hashlib
    digest = hashlib.sha256(b"downloaded model").hexdigest()
    runtime.download_model(storage, "https://huggingface.co/author/repo/resolve/main/model.safetensors",
                           "diffusion_models/file.safetensors", digest, "HF_SECRET")
    assert session.calls[0][1]["headers"]["Authorization"] == "Bearer HF_SECRET"
    assert "Authorization" not in session.calls[1][1]["headers"]
    assert not session.trust_env
    assert (runtime.drive_root(storage) / "models/diffusion_models/file.safetensors").read_bytes() == b"downloaded model"
    assert (runtime.BASE / "models/diffusion_models/file.safetensors").read_bytes() == b"downloaded model"


def test_model_download_wrong_hash_cleans_partial_and_preserves_existing(storage, monkeypatch):
    fake_requests(monkeypatch, [{"data": b"wrong content"}])
    with pytest.raises(RuntimeError, match="SHA256"):
        runtime.download_model(storage, "https://civitai.com/api/download/models/1",
                               "loras/file.safetensors", "0" * 64)
    models = runtime.drive_root(storage) / "models"
    assert not list(models.rglob("*.part"))
    assert not (models / "loras/file.safetensors").exists()
    (models / "loras/file.safetensors").write_bytes(b"existing model")
    with pytest.raises(FileExistsError):
        runtime.download_model(storage, "https://civitai.com/api/download/models/1", "loras/file.safetensors")
    assert (models / "loras/file.safetensors").read_bytes() == b"existing model"


def test_model_download_failure_hides_signed_url(storage, monkeypatch):
    responses = []
    session, error = fake_requests(monkeypatch, responses)
    responses.append(error("https://cdn.example/?signature=SECRET"))
    with pytest.raises(RuntimeError) as exc:
        runtime.download_model(storage, "https://civitai.com/api/download/models/1", "loras/file.safetensors")
    assert "SECRET" not in str(exc.value) and exc.value.__suppress_context__


@pytest.mark.parametrize("url", ["http://cdn.example/model", "https://127.0.0.1/model", "https://169.254.169.254/key",
                                 "https://u:secret@cdn.example/model", "https://internal.example/model"])
def test_reject_unsafe_redirects(storage, monkeypatch, url):
    session, _ = fake_requests(monkeypatch, [{"status": 302, "headers": {"Location": url}}])
    with pytest.raises(ValueError):
        runtime.download_model(storage, "https://civitai.com/api/download/models/1", "loras/file.safetensors", token_secret="secret")
    assert len(session.calls) == 1
    assert not list((runtime.drive_root(storage) / "models").rglob("*.part"))


def test_root_application_process_preserves_gpu_settings_without_credentials(monkeypatch):
    from types import SimpleNamespace
    captured = {}
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("TBE_EPHEM_CREDS_ADDR", "secret-service")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib64-nvidia:/usr/local/cuda/lib64")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")
    def fake_run(args, **kwargs):
        captured.update(kwargs)
        captured["args"] = args
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    runtime.run(["python", "node_install.py"], app=True)
    assert "HF_TOKEN" not in captured["env"] and "TBE_EPHEM_CREDS_ADDR" not in captured["env"]
    assert captured["args"] == ["python", "node_install.py"]
    assert captured["env"]["LD_LIBRARY_PATH"] == "/usr/lib64-nvidia:/usr/local/cuda/lib64"
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert captured["env"]["NVIDIA_DRIVER_CAPABILITIES"] == "compute,utility"
    assert captured["env"]["HOME"] == str(runtime.BASE / "home")


def test_backend_listens_on_loopback_only():
    command = runtime.backend_command()
    assert command[:2] == [str(runtime.BASE / ".venv/bin/python"), str(runtime.BASE / "environment/ComfyUI/main.py")]
    assert command[command.index("--listen") + 1] == "127.0.0.1"


def as_colab_root(monkeypatch):
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 0, raising=False)


def test_application_commands_drop_root_and_cannot_regain_it(monkeypatch):
    from types import SimpleNamespace
    captured = {}
    as_colab_root(monkeypatch)
    monkeypatch.setattr(runtime.subprocess, "run", lambda args, **kw: captured.update(args=args) or SimpleNamespace(
        returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(runtime, "bounded", lambda args, **kw: captured.update(args=args, **kw) or (0, b"", b""))
    runtime.run(["python", "-m", "pip", "install", "-r", "requirements.txt"], app=True, capture=True)
    assert captured["timeout"] > 0 and captured["limit"] > 0  # Root never waits on it without bound.
    prefix = captured["args"][:captured["args"].index("--") + 1]
    # Looked up while still root, with a PATH whose first directory the application user can write.
    assert prefix[0] == "/usr/bin/setpriv" and "--no-new-privs" in prefix and "--init-groups" in prefix
    assert prefix[prefix.index("--reuid") + 1] == prefix[prefix.index("--regid") + 1] == runtime.APP_USER != "root"
    assert captured["args"][len(prefix):] == ["python", "-m", "pip", "install", "-r", "requirements.txt"]
    runtime.run(["apt-get", "update"], capture=True)
    assert captured["args"] == ["apt-get", "update"]


def test_root_touches_the_application_tree_only_as_the_application_user(monkeypatch, tmp_path):
    from types import SimpleNamespace
    calls = []
    as_colab_root(monkeypatch)
    monkeypatch.setattr(runtime, "BASE", tmp_path / "session")

    def observed(args, **kwargs):
        calls.append((args, kwargs))
        return 0, b"content", b""
    monkeypatch.setattr(runtime, "bounded", observed)
    monkeypatch.setenv("HF_TOKEN", "secret")
    assert runtime.app_action("get", "models/vae/model.safetensors") == b"content"
    args, kwargs = calls[0]
    assert args[0] == "/usr/bin/setpriv" and "--no-new-privs" in args
    assert args[args.index("--") + 1:][2:] == ["app", str(tmp_path / "session"), "get", "models/vae/model.safetensors"]
    assert "HF_TOKEN" not in kwargs["env"] and kwargs["timeout"] > 0 and kwargs["limit"] > 0
    monkeypatch.setattr(runtime, "bounded", lambda args, **kwargs: (_ for _ in ()).throw(RuntimeError("時間内に終わりませんでした")))
    with pytest.raises(RuntimeError, match="archive に失敗しました: 時間内"):
        runtime.app_action("archive")


def test_put_refuses_to_leave_the_application_tree(storage):
    outside = runtime.PRIVATE / "session.json"
    for relative in ("../private/session.json", "/etc/ld.so.preload"):
        with pytest.raises(ValueError):
            runtime.app_action("put", relative, stdin=b"overwritten")
    assert not outside.exists()
    runtime.app_action("put", "models/vae/model.safetensors", 1_700_000_000_000_000_000, stdin=b"weights")
    saved = runtime.BASE / "models/vae/model.safetensors"
    assert saved.read_bytes() == b"weights" and int(saved.stat().st_mtime) == 1_700_000_000
    assert runtime.app_action("get", "models/vae/model.safetensors") == b"weights"


def test_venv_failure_displays_stdout_and_stderr(monkeypatch, capsys):
    from types import SimpleNamespace
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=1, stdout="ensurepip is not available", stderr="venv diagnostic"))
    with pytest.raises(RuntimeError, match="終了コード 1"):
        runtime.run(["python", "-m", "venv", "local-venv"], app=True, capture=True)
    output = capsys.readouterr().out
    assert "ensurepip is not available" in output and "venv diagnostic" in output


def test_existing_ensurepip_needs_no_apt_install(monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    def unexpected_install(*args, **kwargs):
        raise AssertionError("APT should not run when ensurepip is available")
    monkeypatch.setattr(runtime, "run", unexpected_install)
    runtime.ensure_venv_support()


@pytest.mark.parametrize("minor", [12, 13])
def test_missing_ensurepip_installs_matching_python_package(monkeypatch, minor):
    import importlib.util
    from types import SimpleNamespace
    installed = []
    with monkeypatch.context() as scoped:
        scoped.setattr(runtime.sys, "version_info", SimpleNamespace(major=3, minor=minor))
        scoped.setattr(importlib.util, "find_spec", lambda name: object() if installed else None)
        scoped.setattr(runtime, "run", lambda args: installed.append(args))
        scoped.setattr(runtime, "installed_apt_packages", lambda: {"git"})
        scoped.setattr(runtime, "APT_INDEX_REFRESHED", [])
        runtime.ensure_venv_support()
    assert installed == [["apt-get", "update", "-qq"], ["apt-get", "install", "-y", "-qq", f"python3.{minor}-venv"]]


def test_ensurepip_still_missing_after_install_stops_setup(monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(runtime, "run", lambda args: None)
    with pytest.raises(RuntimeError, match="インストール後も"):
        runtime.ensure_venv_support()


def test_venv_package_install_failure_has_actionable_error(monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    def failed_install(args):
        raise RuntimeError("APT failed")
    monkeypatch.setattr(runtime, "run", failed_install)
    with pytest.raises(RuntimeError, match="apt-cache policy"):
        runtime.ensure_venv_support()


def test_uncaptured_pip_failure_keeps_full_diagnostics(monkeypatch, capsys):
    from types import SimpleNamespace
    def failed_run(args, **kwargs):
        assert kwargs["capture_output"] is True
        return SimpleNamespace(returncode=1, stdout="FIRST DIAGNOSTIC\n" + "x" * 5000,
                               stderr="ERROR: incompatible Python version")
    monkeypatch.setattr(runtime.subprocess, "run", failed_run)
    with pytest.raises(RuntimeError, match="python -m pip install"):
        runtime.run(["python", "-m", "pip", "install", "-r", "requirements.txt"], app=True)
    output = capsys.readouterr().out
    assert "FIRST DIAGNOSTIC" in output and "ERROR: incompatible Python version" in output


def test_root_command_failure_also_displays_diagnostics(monkeypatch, capsys):
    from types import SimpleNamespace
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=1, stdout="APT explanation", stderr="package not found"))
    with pytest.raises(RuntimeError, match="apt-get が終了コード 1"):
        runtime.run(["apt-get", "install", "python3.13-venv"], capture=True)
    output = capsys.readouterr().out
    assert "APT explanation" in output and "package not found" in output


def test_successful_captured_command_returns_output_without_printing(monkeypatch, capsys):
    from types import SimpleNamespace
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout="package==1.0\n", stderr=""))
    assert runtime.run(["python", "-m", "pip", "freeze"], app=True, capture=True) == "package==1.0\n"
    assert capsys.readouterr().out == ""


def test_supervisor_preserves_driver_settings_without_notebook_credentials(tmp_path, monkeypatch):
    from types import SimpleNamespace
    captured = {}
    class SpawnObserved(Exception):
        pass
    monkeypatch.setattr(runtime, "PRIVATE", tmp_path)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib64-nvidia")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("TBE_EPHEM_CREDS_ADDR", "secret-service")
    monkeypatch.setattr(runtime.threading, "Thread", lambda **kw: SimpleNamespace(start=lambda: None))
    def observed_spawn(args, **kwargs):
        captured.update(kwargs, args=args)
        raise SpawnObserved()
    monkeypatch.setattr(runtime.subprocess, "Popen", observed_spawn)
    with pytest.raises(SpawnObserved):
        runtime.supervise({"checkpoint_seconds": 120})
    assert captured["env"]["LD_LIBRARY_PATH"] == "/usr/lib64-nvidia"
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert captured["env"]["VIRTUAL_ENV"] == str(runtime.BASE / ".venv")
    assert "HF_TOKEN" not in captured["env"] and "TBE_EPHEM_CREDS_ADDR" not in captured["env"]
    as_colab_root(monkeypatch)
    with pytest.raises(SpawnObserved):
        runtime.supervise({"checkpoint_seconds": 120})
    assert captured["args"][0] == "/usr/bin/setpriv" and captured["args"][-len(runtime.backend_command()):] == runtime.backend_command()


def test_application_library_path_never_includes_the_current_directory(monkeypatch):
    # The loader reads an empty or relative entry as the current directory, which root
    # enters (the application's own tree) before it starts the privilege-dropping program.
    monkeypatch.setenv("LD_LIBRARY_PATH", ":/usr/lib64-nvidia::relative/lib:.:/usr/local/cuda/lib64:")
    assert runtime.application_environment()["LD_LIBRARY_PATH"] == "/usr/lib64-nvidia:/usr/local/cuda/lib64"


def test_apt_runs_only_for_missing_packages_and_refreshes_the_index_once(monkeypatch):
    from types import SimpleNamespace
    calls = []
    # Ubuntu 24.04: libglib2.0-0 exists only as a name libglib2.0-0t64 provides; libgl1 was removed.
    listing = ("ii \tgit\t\n"
               "ii \tffmpeg\t\n"
               "ii \tlibglib2.0-0t64\tlibglib2.0-0 (= 2.80.0-6ubuntu3.4), libglib2.0-0-abi\n"
               "rc \tlibgl1\t\n")
    monkeypatch.setattr(runtime.subprocess, "run", lambda args, **kw: SimpleNamespace(stdout=listing))
    monkeypatch.setattr(runtime, "run", lambda args: calls.append(args))
    monkeypatch.setattr(runtime, "APT_INDEX_REFRESHED", [])
    assert runtime.installed_apt_packages() == {"git", "ffmpeg", "libglib2.0-0t64", "libglib2.0-0", "libglib2.0-0-abi"}
    runtime.apt_install(["git", "ffmpeg", "libglib2.0-0"])
    assert calls == []  # Nothing missing: not even the index refresh.
    runtime.apt_install(["git", "libgl1"])
    runtime.apt_install(["aria2"])
    assert calls == [["apt-get", "update", "-qq"], ["apt-get", "install", "-y", "-qq", "libgl1"],
                     ["apt-get", "install", "-y", "-qq", "aria2"]]


def test_application_pip_ignores_the_global_configuration():
    # Colab's pip.conf logs to a root-only file; unprivileged pip would fail on every record.
    assert runtime.application_environment()["PIP_CONFIG_FILE"] == "/dev/null"


def test_environment_installs_go_through_uv_with_pip_compatible_index_lookup(monkeypatch, capsys):
    from types import SimpleNamespace
    calls = []
    monkeypatch.setattr(runtime.subprocess, "run", lambda args, **kw: calls.append(args) or SimpleNamespace(
        returncode=0, stdout="", stderr=""))
    py = str(runtime.BASE / ".venv/bin/python")
    runtime.install_packages("--index-strategy", "unsafe-best-match", "--extra-index-url",
                             "https://download.pytorch.org/whl/cu130", "-r", "requirements.lock.txt")
    assert calls == [[py, "-m", "uv", "pip", "install", "--python", py, "--index-strategy", "unsafe-best-match",
                      "--extra-index-url", "https://download.pytorch.org/whl/cu130", "-r", "requirements.lock.txt"]]
    assert "実行中: python -m uv pip install" in capsys.readouterr().out


@pytest.mark.parametrize("pasted,download,name", [
    ("https://huggingface.co/author/repo/blob/main/split_files/vae/model.safetensors",
     "https://huggingface.co/author/repo/resolve/main/split_files/vae/model.safetensors", "model.safetensors"),
    ("https://huggingface.co/author/repo/resolve/refs%2Fpr%2F2/model%20v2.safetensors?download=true",
     "https://huggingface.co/author/repo/resolve/refs%2Fpr%2F2/model%20v2.safetensors", "model v2.safetensors"),
    (" https://civitai.com/models/12345/some-model?modelVersionId=67890 ",
     "https://civitai.com/api/download/models/67890", ""),
    ("https://civitai.com/api/download/models/67890?type=Model&format=SafeTensor&token=SECRET",
     "https://civitai.com/api/download/models/67890", ""),
])
def test_model_links_pasted_from_a_browser(pasted, download, name):
    assert runtime.parse_model_url(pasted) == (download, name)


@pytest.mark.parametrize("pasted", [
    "https://huggingface.co/author/repo", "https://huggingface.co/author/repo/tree/main",
    "https://civitai.com/models/12345/some-model", "https://civitai.com/api/download/models/abc",
    "http://huggingface.co/author/repo/blob/main/model.safetensors",
    "https://huggingface.co.attacker.example/author/repo/blob/main/model.safetensors",
    "https://example.com/model.safetensors", ""])
def test_unusable_model_links_are_explained_not_downloaded(pasted):
    with pytest.raises(ValueError):
        runtime.parse_model_url(pasted)


def test_token_secret_follows_the_site_of_the_link():
    assert runtime.model_token_name("https://civitai.com/models/1?modelVersionId=2") == "CIVITAI_TOKEN"
    assert runtime.model_token_name("https://huggingface.co/a/b/blob/main/c.safetensors") == "HF_TOKEN"


def test_pasted_link_downloads_into_the_chosen_folder(storage, monkeypatch):
    fake_requests(monkeypatch, [{"data": b"hf model"}])
    runtime.fetch_model(storage, "https://huggingface.co/author/repo/blob/main/split/model.safetensors", "vae")
    assert (runtime.drive_root(storage) / "models/vae/model.safetensors").read_bytes() == b"hf model"
    assert (runtime.BASE / "models/vae/model.safetensors").read_bytes() == b"hf model"
    session, _ = fake_requests(monkeypatch, [{"data": b"renamed"}])
    runtime.fetch_model(storage, "https://huggingface.co/author/repo/blob/main/split/model.safetensors", " loras/ ", "mine.safetensors")
    assert (runtime.drive_root(storage) / "models/loras/mine.safetensors").read_bytes() == b"renamed"
    with pytest.raises(ValueError):
        runtime.fetch_model(storage, "https://huggingface.co/author/repo/blob/main/model.safetensors", "../outside")
    with pytest.raises(ValueError):
        runtime.fetch_model(storage, "https://huggingface.co/author/repo/blob/main/model.safetensors", "vae", "sub/name.safetensors")


def test_civitai_download_takes_the_served_file_name_but_never_a_path(storage, monkeypatch):
    models = runtime.drive_root(storage) / "models"
    served = {"Content-Type": "application/octet-stream",
              "Content-Disposition": 'attachment; filename="styleLora_v2.safetensors"'}
    fake_requests(monkeypatch, [{"status": 307, "headers": {"Location": "https://cdn.example/signed?key=secret"}},
                                {"headers": served, "data": b"lora"}])
    runtime.fetch_model(storage, "https://civitai.com/models/1/name?modelVersionId=22", "loras")
    assert (models / "loras/styleLora_v2.safetensors").read_bytes() == b"lora"
    fake_requests(monkeypatch, [{"headers": served, "data": b"again"}])
    with pytest.raises(FileExistsError):
        runtime.fetch_model(storage, "https://civitai.com/api/download/models/22", "loras")
    assert (models / "loras/styleLora_v2.safetensors").read_bytes() == b"lora"
    for header in ('attachment; filename="../../environment/latest.json"', "attachment", 'attachment; filename=".hidden"'):
        fake_requests(monkeypatch, [{"headers": {"Content-Disposition": header}, "data": b"x"}])
        try:
            runtime.fetch_model(storage, "https://civitai.com/api/download/models/33", "loras")
        except RuntimeError as failure:
            assert "ファイル名を取得できません" in str(failure)
    assert sorted(p.name for p in models.rglob("*") if p.is_file()) == ["latest.json", "styleLora_v2.safetensors"]
    assert (models / "loras/latest.json").read_bytes() == b"x"  # Only the last path component is ever used.
    assert not list(models.rglob("*.part"))


def test_empty_placeholder_files_are_not_treated_as_models(storage):
    models = runtime.drive_root(storage) / "models"
    (models / "checkpoints").mkdir()
    (models / "checkpoints/put_checkpoints_here").write_bytes(b"")
    (runtime.BASE / "models/vae").mkdir(parents=True)
    (runtime.BASE / "models/vae/put_vae_here").write_bytes(b"")
    assert runtime.sync_models(storage) == 0
    runtime.save_new_cache_models(storage, force=True)
    assert not (models / "vae/put_vae_here").exists() and not (runtime.BASE / "models/checkpoints/put_checkpoints_here").exists()


def test_virtual_environment_needs_nothing_from_apt(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, "run", lambda args, **kw: calls.append(list(map(str, args))))
    monkeypatch.setattr(runtime, "apt_install", lambda packages: calls.append(["apt", *packages]))
    py = str(runtime.BASE / ".venv/bin/python")
    assert str(runtime.create_venv()) == py
    # The notebook's pip installs pip and uv into the new environment; apt would replace
    # the interpreter the running kernel was started from.
    assert calls == [[sys.executable, "-m", "venv", "--without-pip", str(runtime.BASE / ".venv")],
                     [sys.executable, "-m", "pip", "--python", py, "install", "pip", "uv"]]


def notebook_cell(marker, language="JP"):
    import build_notebook
    return next(cell["source"] for cell in build_notebook.build(language)["cells"] if marker in cell["source"])


def test_update_check_offers_the_newest_release_and_setting_it_is_a_choice(monkeypatch, capsys):
    import build_notebook
    import subprocess
    from types import SimpleNamespace
    listing = "".join(f"{index:040x}\trefs/tags/{tag}\n" for index, tag in enumerate(
        ["latest", "v0.9.2", "v0.39.2", "v0.39.10", "v0.40.0-rc1", "v0.4.0", "v0.39.9"]))
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda args, **kw: calls.append(args) or SimpleNamespace(returncode=0, stdout=listing))
    check, settings = notebook_cell("USE_LATEST_COMFYUI"), notebook_cell("DRIVE_FOLDER_NAME")
    assert check.startswith("#@title 0. ") and settings.startswith("#@title 1. ")
    cells = [cell["source"] for cell in build_notebook.build("JP")["cells"]]
    assert cells.index(check) + 1 == cells.index(settings)  # Above the settings, and before the runtime exists.
    assert "runtime." not in check
    names = {}
    exec(check, names)
    exec(settings, names)
    assert calls[0][:2] == ["git", "ls-remote"] and "https://github.com/Comfy-Org/ComfyUI.git" in calls[0]
    # Numbers are compared as numbers; names that are not plain releases are ignored.
    assert names["LATEST_COMFY_REF"] == "v0.39.10" and names["COMFY_REF"] == "v0.39.0"
    exec(check.replace("USE_LATEST_COMFYUI = False", "USE_LATEST_COMFYUI = True"), names)
    assert names["COMFY_REF"] == "v0.39.10"
    exec(settings, names)  # The form of cell 1 does not take the choice back.
    assert names["COMFY_REF"] == "v0.39.10" and "COMFY_REF: v0.39.10" in capsys.readouterr().out
    exec(check, names)
    exec(settings, names)
    assert names["COMFY_REF"] == "v0.39.0"
    monkeypatch.setattr(subprocess, "run", lambda args, **kw: SimpleNamespace(returncode=128, stdout=""))
    with pytest.raises(RuntimeError, match="最新リリースを取得できませんでした"):
        exec(check, names)
    assert names["COMFY_REF"] == "v0.39.0"
    update = notebook_cell("UPDATE_COMFYUI_NOW")
    assert update.startswith("#@title 11. ") and "runtime.update_comfyui(config, COMFY_REF)" in update


class FakeEnvironment:
    """ComfyUI's repository and virtual environment as update_comfyui sees them through run()."""
    OLD, NEW = "a" * 40, "b" * 40
    BASE_PACKAGES = "comfyui-frontend-package==1.0\ncustom-node-dependency==2.0\ntorch==2.9.0+cu130\n"

    def __init__(self, fail=()):
        self.head, self.fail, self.calls, self.saves = self.OLD, set(fail), [], 0
        self.packages = self.BASE_PACKAGES
        self.files = {"environment/runtime.json": json.dumps({"torch_index_url": "https://download.pytorch.org/whl/cu130"})}

    def run(self, args, **kwargs):
        args = list(map(str, args))
        assert kwargs.get("app"), args  # Nothing here runs as root.
        if args[0] == "git":
            self.calls.append(" ".join(args[:2]))
            if args[1] == "fetch":
                return ""
            if args[1] == "describe":
                return {self.OLD: "v0.39.0", self.NEW: "v0.39.2"}[self.head] + "\n"
            if args[1] == "rev-parse":
                known = {"HEAD^{commit}": self.head, "refs/tags/v0.39.2^{commit}": self.NEW, "refs/tags/v0.39.0^{commit}": self.OLD}
                if args[-1] not in known:
                    raise RuntimeError("unknown revision")
                return known[args[-1]] + "\n"
            if args[1] == "checkout":
                assert args[2:4] == ["--force", "--detach"]
                self.head = args[4]
                return ""
        if "freeze" in args:
            return self.packages
        if "check" in args:
            self.calls.append("pip check")
            if "check" in self.fail:
                raise RuntimeError("custom-node-dependency 2.0 has requirement numpy<2")
            return None
        if "uninstall" in args:
            self.calls.append("uninstall " + " ".join(args[args.index("--python") + 2:]))
            self.packages = self.BASE_PACKAGES
            return None
        if "install" in args:
            from_lock = any(arg.endswith("requirements.lock.txt") for arg in args)
            self.calls.append("install lock" if from_lock else "install requirements")
            if ("undo" if from_lock else "install") in self.fail:
                raise RuntimeError("network unreachable")
            if from_lock:
                self.packages = self.packages.replace("==1.5", "==1.0")
            else:
                assert self.head == self.NEW  # The requirements read are the new version's.
                self.packages = self.BASE_PACKAGES.replace("==1.0", "==1.5") + "new-dependency==0.1\n"
            return None
        raise AssertionError(args)

    def app_action(self, action, *args, stdin=None, **kwargs):
        if action == "prepare":
            self.calls.append("prepare")
        elif action == "put":
            self.files[args[0]] = stdin.decode()
        elif action == "get":
            return self.files[args[0]].encode()
        return b""

    def checkpoint(self, config, force=False):
        runtime.refuse_unfinished_update()
        self.saves += 1


@pytest.fixture
def updatable(storage, monkeypatch):
    def environment(fail=()):
        fake = FakeEnvironment(fail)
        monkeypatch.setattr(runtime, "run", fake.run)
        monkeypatch.setattr(runtime, "app_action", fake.app_action)
        monkeypatch.setattr(runtime, "checkpoint", fake.checkpoint)
        return fake
    (runtime.BASE / "environment/ComfyUI").mkdir(parents=True)
    (runtime.BASE / "environment/ComfyUI/requirements.txt").write_text("")
    return storage, environment


def test_update_moves_comfyui_to_the_requested_version_and_saves_it(updatable, capsys):
    config, environment = updatable
    fake = environment()
    assert runtime.update_comfyui(config, "v0.39.2") is True
    assert fake.head == fake.NEW and fake.saves == 1
    assert fake.calls == ["git fetch", "git rev-parse", "git describe", "git rev-parse", "git checkout", "prepare",
                          "install requirements", "pip check", "git describe"]
    # What was installed before is on record before anything changes, for the way back.
    assert fake.files["environment/requirements.lock.txt"] == fake.BASE_PACKAGES
    assert not (runtime.PRIVATE / "update-incomplete.json").exists()
    assert runtime.t("update.done", previous="v0.39.0", version="v0.39.2") in capsys.readouterr().out
    fake.calls.clear()
    assert runtime.update_comfyui(config, "v0.39.2") is False  # Nothing to do, nothing touched.
    assert fake.saves == 1 and "git checkout" not in fake.calls and runtime.t("update.already", version="v0.39.2") in capsys.readouterr().out


@pytest.mark.parametrize("ref,error", [("--upload-pack=x", "refが不正"), ("a b", "refが不正"),
                                        ("v9.9.9", "v9.9.9 というバージョンが見つかりません")])
def test_update_refuses_a_version_that_does_not_exist(updatable, ref, error):
    config, environment = updatable
    fake = environment()
    with pytest.raises(ValueError, match=error):
        runtime.update_comfyui(config, ref)
    assert fake.head == fake.OLD and fake.saves == 0 and "git checkout" not in fake.calls


def test_update_needs_comfyui_stopped_and_drive_connected(updatable, monkeypatch):
    config, environment = updatable
    fake = environment()
    (runtime.PRIVATE / "processes.json").write_text("{}")
    with pytest.raises(RuntimeError, match="ComfyUIが起動中です"):
        runtime.update_comfyui(config, "v0.39.2")
    (runtime.PRIVATE / "processes.json").unlink()
    monkeypatch.setattr(runtime, "DRIVE_MOUNT", runtime.PRIVATE / "unmounted")
    with pytest.raises(RuntimeError, match=re.escape(runtime.t("drive.not_mounted"))):
        runtime.update_comfyui(config, "v0.39.2")
    assert fake.calls == []


@pytest.mark.parametrize("failure", ["install", "check"])
def test_failed_update_is_undone_and_never_reaches_drive(updatable, capsys, failure):
    config, environment = updatable
    fake = environment(fail={failure})
    with pytest.raises(RuntimeError, match="network unreachable|numpy<2"):
        runtime.update_comfyui(config, "v0.39.2")
    assert fake.head == fake.OLD and fake.saves == 0
    undo = fake.calls[len(fake.calls) - fake.calls[::-1].index("git checkout") - 1:]
    assert undo[:3] == ["git checkout", "prepare", "install lock"]
    # Packages only the new version brought are removed again.
    assert undo[3:] == (["uninstall new-dependency"] if failure == "check" else [])
    assert fake.packages == fake.BASE_PACKAGES
    assert not (runtime.PRIVATE / "update-incomplete.json").exists()
    assert "元のバージョン（v0.39.0）へ戻しました" in capsys.readouterr().out
    runtime.checkpoint(config, force=True)  # Saving works again.
    assert fake.saves == 1


def test_update_that_cannot_be_undone_blocks_saving_and_starting_until_it_is(updatable, monkeypatch, capsys):
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace())
    config, environment = updatable
    fake = environment(fail={"check", "undo"})
    with pytest.raises(RuntimeError, match="network unreachable"):
        runtime.update_comfyui(config, "v0.39.2")
    assert (runtime.PRIVATE / "update-incomplete.json").exists() and fake.saves == 0
    assert "元のバージョンへも戻せませんでした" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="更新が途中で止まっています"):
        runtime.checkpoint(config, force=True)
    with pytest.raises(RuntimeError, match="更新が途中で止まっています"):
        runtime.start(dict(config), "tunnel")
    assert not (runtime.PRIVATE / "processes.json").exists()
    # Running the update again first finishes undoing the broken one, then updates.
    fake.fail.clear()
    fake.calls.clear()
    assert runtime.update_comfyui(config, "v0.39.2") is True
    assert fake.calls[:4] == ["git checkout", "prepare", "install lock", "uninstall new-dependency"]
    assert fake.head == fake.NEW and fake.saves == 1 and not (runtime.PRIVATE / "update-incomplete.json").exists()
    assert "途中で止まった更新を元に戻します" in capsys.readouterr().out


def signed_in(client, server):
    """A browser that has already paired: it holds the session cookie."""
    client.cookie_jar.update_cookies({"comfy_colab": "test-cookie-secret"}, server.make_url("/"))


@pytest.fixture
def private(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "PRIVATE", tmp_path / "private")
    runtime.PRIVATE.mkdir()
    return runtime.PRIVATE


def test_proxy_auth_http_range_upload_and_websocket(private):
    from aiohttp import web, ClientSession, CookieJar
    from aiohttp.test_utils import TestServer

    async def scenario():
        seen = []
        binary = bytes(range(256)) * 8192  # Multi-chunk image/video streaming.

        async def backend(request):
            seen.append(dict(request.cookies))
            if request.path == "/ws":
                ws = web.WebSocketResponse()
                await ws.prepare(request)
                async for msg in ws:
                    if isinstance(msg.data, bytes):
                        await ws.send_bytes(msg.data)
                    else:
                        await ws.send_str(msg.data)
                return ws
            if request.method == "POST":
                return web.Response(body=await request.read())
            if request.headers.get("Range"):
                return web.Response(body=binary[:10], status=206, headers={"Content-Range": f"bytes 0-9/{len(binary)}"})
            return web.Response(body=binary, content_type="video/mp4")

        app = web.Application(client_max_size=8 * 1024 * 1024)
        app.router.add_route("*", "/{path:.*}", backend)
        async with TestServer(app) as upstream:
            config = {"mode": "tunnel", "cookie_secret": "test-cookie-secret"}
            proxy = runtime.create_proxy(config, str(upstream.make_url("/")).rstrip("/"), secure_cookie=False)
            async with TestServer(proxy) as server:
                async with ClientSession(cookie_jar=CookieJar(unsafe=True)) as client:
                    for route in ("/", "/api/view?filename=video.mp4", "/ws", "/object_info"):
                        async with client.get(server.make_url(route)) as res:
                            assert res.status == 401
                    async with client.post(server.make_url("/prompt"), data=b"{}") as res:
                        assert res.status == 401
                    # A key in the address is no longer a way in.
                    async with client.get(server.make_url("/?access_key=test-cookie-secret")) as res:
                        assert res.status == 401
                    assert not seen  # Nothing unauthenticated reaches ComfyUI.
                    code = runtime.new_pairing_code()
                    async with client.post(server.make_url(runtime.PAIRING_PATH), data={"code": code}, allow_redirects=False) as res:
                        assert res.status == 303 and res.headers["Location"] == "/"
                        assert res.cookies["comfy_colab"]["httponly"]
                        assert res.cookies["comfy_colab"]["samesite"] == "Lax"
                    async with client.get(server.make_url("/api/view?filename=video.mp4")) as res:
                        assert res.status == 200 and await res.read() == binary
                        assert int(res.headers["Content-Length"]) == len(binary)
                    async with client.get(server.make_url("/api/view"), headers={"Range": "bytes=0-9"}) as res:
                        assert res.status == 206 and await res.read() == binary[:10]
                    async with client.head(server.make_url("/api/view")) as res:
                        assert res.status == 200 and int(res.headers["Content-Length"]) == len(binary)
                    async with client.post(server.make_url("/upload/image"), data=binary) as res:
                        assert res.status == 200 and await res.read() == binary
                    async with client.post(server.make_url("/prompt"), headers={"Origin": "https://evil.example"}) as res:
                        assert res.status == 403
                    async with client.ws_connect(server.make_url("/ws")) as ws:
                        event = json.dumps({"type": "executed", "data": {"output": {"video": [{"filename": "video.mp4", "type": "output"}]}}})
                        await ws.send_str(event)
                        assert (await ws.receive(timeout=3)).data == event
                        await ws.send_bytes(b"preview")
                        assert (await ws.receive(timeout=3)).data == b"preview"
                    # Desktop's asset downloader uses the same session cookie.
                    async with client.get(server.make_url("/api/view?filename=image.png")) as res:
                        assert res.status == 200
                    # Signed in, the pairing page has nothing more to offer.
                    async with client.get(server.make_url(runtime.PAIRING_PATH), allow_redirects=False) as res:
                        assert res.status == 303 and res.headers["Location"] == "/"
                assert seen and all("comfy_colab" not in cookies for cookies in seen)
                async with ClientSession() as unauthenticated:
                    async with unauthenticated.get(server.make_url("/api/view?filename=image.png")) as res:
                        assert res.status == 401
    asyncio.run(scenario())


def test_pairing_code_lets_one_browser_in_once_and_runs_out(private, monkeypatch, capsys):
    from aiohttp import web, ClientSession, CookieJar
    from aiohttp.test_utils import TestServer

    async def scenario():
        reached = []

        async def backend(request):
            reached.append(request.path)
            return web.Response(text="ComfyUI")

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", backend)
        browser = {"Accept": "text/html,application/xhtml+xml,*/*;q=0.8"}
        pair = runtime.PAIRING_PATH

        def fresh():
            return ClientSession(cookie_jar=CookieJar(unsafe=True))

        async def enter(client, code, **kwargs):
            async with client.post(server.make_url(pair), data={"code": code}, allow_redirects=False, **kwargs) as res:
                return res.status, await res.text()

        async with TestServer(app) as upstream:
            config = {"mode": "tunnel", "cookie_secret": "test-cookie-secret"}
            proxy = runtime.create_proxy(config, str(upstream.make_url("/")).rstrip("/"), secure_cookie=False)
            async with TestServer(proxy) as server:
                code = runtime.new_pairing_code()
                assert re.fullmatch(r"\d{8}", code)
                async with fresh() as first, fresh() as second:
                    # Opening the address in a browser leads to the page that asks for the code.
                    async with first.get(server.make_url("/"), headers=browser, allow_redirects=False) as res:
                        assert res.status == 303 and res.headers["Location"] == pair
                    async with first.get(server.make_url("/"), headers=browser) as res:
                        page = await res.text()
                        assert res.status == 200 and res.url.path == pair
                        assert "ペアリングコードを入力してください" in page and 'name="code"' in page
                        assert res.headers["Cache-Control"] == "no-store" and res.headers["X-Frame-Options"] == "DENY"
                        # With no-referrer a browser submits the form as "Origin: null" and would be refused.
                        assert res.headers["Referrer-Policy"] == "same-origin"
                        # (A local filtering proxy may rewrite default-src; these directives pass through.)
                        policy = res.headers["Content-Security-Policy"]
                        assert "form-action 'self'" in policy and "frame-ancestors 'none'" in policy
                        assert code not in page
                    assert (await enter(first, code, headers={"Origin": "null"}))[0] == 403
                    # A wrong code is counted and says how many tries are left.
                    status, page = await enter(first, "00000000" if code != "00000000" else "11111111")
                    assert status == 401 and f"あと{runtime.PAIRING_ATTEMPTS - 1}回" in page
                    # Another site cannot submit a code through this browser; too large a body is not read.
                    assert (await enter(first, code, headers={"Origin": "https://evil.example"}))[0] == 403
                    async with first.post(server.make_url(pair), data=b"code=" + b"1" * 400) as res:
                        assert res.status == 400
                    # The code as shown, with its hyphen, lets this browser in.
                    status, _ = await enter(first, code[:4] + "-" + code[4:])
                    assert status == 303
                    async with first.get(server.make_url("/"), headers=browser) as res:
                        assert res.status == 200 and await res.text() == "ComfyUI"
                    # It worked once: a second browser with the same code stays out.
                    status, page = await enter(second, code)
                    assert status == 401 and "このコードは使えません" in page
                    async with second.get(server.make_url("/api/prompt")) as res:
                        assert res.status == 401
                    # A new code replaces it; wrong guesses use it up.
                    code = runtime.new_pairing_code()
                    for left in range(runtime.PAIRING_ATTEMPTS - 1, 0, -1):
                        status, page = await enter(second, "x")
                        assert status == 401 and f"あと{left}回" in page
                    assert "このコードは使えません" in (await enter(second, "x"))[1]
                    assert "このコードは使えません" in (await enter(second, code))[1]
                    assert not (private / "pairing.json").exists()
                    # A code that was not used in time is gone too.
                    code = runtime.new_pairing_code()
                    clock = runtime.time.time
                    monkeypatch.setattr(runtime.time, "time", lambda: clock() + runtime.PAIRING_SECONDS + 1)
                    assert "このコードは使えません" in (await enter(second, code))[1]
                    monkeypatch.setattr(runtime.time, "time", clock)
                    assert (await enter(second, runtime.new_pairing_code()))[0] == 303
        assert reached == ["/"]  # Only the paired browser's page load got through.
    asyncio.run(scenario())
    assert "別のサイトからの要求を拒否しました" not in capsys.readouterr().out  # Strangers cannot fill the log.


def test_proxy_without_a_session_secret_lets_nobody_in(private):
    from aiohttp import ClientSession
    from aiohttp.test_utils import TestServer

    async def scenario():
        # Stopped, or a settings file from before a start: there is no cookie value that works.
        async with TestServer(runtime.create_proxy({}, secure_cookie=False)) as server:
            async with ClientSession() as client:
                for cookie in ("", "comfy_colab="):
                    async with client.get(server.make_url("/api/prompt"), headers={"Cookie": cookie} if cookie else {}) as res:
                        assert res.status == 401
                async with client.post(server.make_url(runtime.PAIRING_PATH), data={"code": ""}) as res:
                    assert res.status == 401
    asyncio.run(scenario())


def test_production_cookie_security(private):
    from aiohttp import ClientSession
    from aiohttp.test_utils import TestServer

    async def scenario():
        config = {"mode": "tunnel", "cookie_secret": "secret"}
        async with TestServer(runtime.create_proxy(config)) as server:
            async with ClientSession() as client:
                data = {"code": runtime.new_pairing_code()}
                async with client.post(server.make_url(runtime.PAIRING_PATH), data=data, allow_redirects=False) as res:
                    cookie = res.cookies[runtime.COOKIE]
                    assert res.status == 303 and cookie["secure"] and cookie["httponly"]
                    assert cookie["path"] == "/" and not cookie["domain"]
                    assert cookie.value == "secret" and int(cookie["max-age"]) == runtime.COOKIE_SECONDS
    asyncio.run(scenario())


def test_proxy_behind_colab_forwarding_needs_no_code_but_keeps_other_sites_out(capsys):
    from aiohttp import web, ClientSession
    from aiohttp.test_utils import TestServer

    async def scenario():
        seen = []

        async def backend(request):
            seen.append((request.method, request.path, request.headers.get("Host"), dict(request.cookies),
                         request.headers.get("Sec-Fetch-Site")))
            # ComfyUI's own check: a loopback Host with another Origin is refused.
            origin = request.headers.get("Origin")
            if origin and origin.split("//")[1] != request.headers["Host"]:
                return web.Response(status=403)
            return web.Response(text="ComfyUI")

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", backend)
        forwarded = "https://8189-gpu-t4-s-abc123-b.asia-east1-0.prod.colab.dev"
        async with TestServer(app) as upstream:
            config = {"mode": "colab", "origins": [forwarded]}
            proxy = runtime.create_proxy(config, str(upstream.make_url("/")).rstrip("/"))
            async with TestServer(proxy) as server:
                async with ClientSession() as client:
                    # Colab only forwards the owner's browser, so there is no code and no cookie.
                    link = {"Accept": "text/html", "Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate",
                            "Sec-Fetch-Dest": "document", "Cookie": "colab-auth=google-sign-in"}
                    async with client.get(server.make_url("/"), headers=link, allow_redirects=False) as res:
                        assert res.status == 200 and await res.text() == "ComfyUI"
                    # The page's own requests carry the forwarded address as their origin.
                    async with client.post(server.make_url("/prompt"), headers={"Origin": forwarded}) as res:
                        assert res.status == 200
                    # A page of another site open in the same browser cannot use that sign-in.
                    for origin in ("https://evil.example", "https://8189-gpu-t4-s-other-b.asia-east1-0.prod.colab.dev"):
                        async with client.post(server.make_url("/prompt"), headers={"Origin": origin}) as res:
                            assert res.status == 403
                    async with client.get(server.make_url(runtime.PAIRING_PATH), allow_redirects=False) as res:
                        assert res.status == 303 and res.headers["Location"] == "/"
        assert [entry[:2] for entry in seen] == [("GET", "/"), ("POST", "/prompt")]
        # Host follows the origin the browser used, whatever the forwarder put there.
        assert seen[1][2] == forwarded.split("//")[1]
        # Colab's sign-in cookie is not handed to ComfyUI and its custom nodes.
        assert seen[0][3] == {} and seen[0][4] is None
    asyncio.run(scenario())
    assert capsys.readouterr().out.count("別のサイトからの要求を拒否しました") == 2


def tunnel_settings(*services, version=1):
    """The line cloudflared logs when Cloudflare sends a named tunnel its settings."""
    settings = {"ingress": [{"hostname": "comfy.example.com", "originRequest": {}, "service": service} for service in services],
                "warp-routing": {"enabled": False}}
    return f"2026-10-12T00:00:00Z INF Updated to new configuration config={json.dumps(json.dumps(settings))} version={version}\n"


class StartedProcesses:
    """Stand-ins for the processes start() launches."""
    def __init__(self, tmp_path, monkeypatch, proxy_answer, public=None, tunnel_log=None):
        from types import SimpleNamespace
        if tunnel_log is None:
            tunnel_log = ("INF Registered tunnel connection connIndex=0 protocol=quic\n"
                          + tunnel_settings("http://127.0.0.1:8189", "http_status:404"))
        self.spawned = spawned = []
        self.environments = environments = []

        class Process:
            pid = 4242

            def __init__(self, args, stdout=None, env=None, **kwargs):
                spawned.append(args)
                environments.append(env)
                if "tunnel" in args:
                    stdout.write(tunnel_log if "run" in args else "INF |  https://quick-test-tunnel.trycloudflare.com  |\n")
                    stdout.flush()

            def poll(self):
                return None

        def get(url, timeout, **kwargs):
            if public and "127.0.0.1" not in url:
                return public(url)
            return SimpleNamespace(ok=True, status_code=proxy_answer if ":8189" in url else 200)

        monkeypatch.setattr(runtime, "PRIVATE", tmp_path)
        monkeypatch.setattr(runtime, "CONFIG", tmp_path / "session.json")
        monkeypatch.setattr(runtime.subprocess, "Popen", Process)
        # The watch for the stop button is a thread of the notebook; here it is only noted.
        self.followed = []
        monkeypatch.setattr(runtime, "follow_stop_requests", self.followed.append)
        monkeypatch.setattr(runtime.os, "killpg", lambda pid, sig: None, raising=False)
        monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=get, RequestException=OSError))


def test_tunnel_start_shows_a_pairing_code_instead_of_a_key_and_stop_revokes_everything(tmp_path, monkeypatch, capsys):
    started = StartedProcesses(tmp_path, monkeypatch, proxy_answer=401)
    config = {"drive_folder": "ComfyUI-Colab", "cookie_secret": "leaked-cookie", "mode": "colab", "origins": ["https://old"]}
    runtime.start(config, "tunnel")
    output = capsys.readouterr().out
    saved = json.loads((tmp_path / "session.json").read_text())
    assert saved["mode"] == "tunnel" and saved["cookie_secret"] not in ("", "leaked-cookie") and "origins" not in saved
    pairing = json.loads((tmp_path / "pairing.json").read_text())
    code = pairing["code"]
    assert pairing["attempts"] == runtime.PAIRING_ATTEMPTS
    # The address carries no secret, and the secret that signs browsers in is never shown.
    assert "https://quick-test-tunnel.trycloudflare.com/\n" in output and "access_key" not in output
    assert f"ペアリングコード: {code[:4]}-{code[4:]}（10分間・1回だけ有効）" in output
    assert saved["cookie_secret"] not in output
    # The proxy holds its port before any application code runs and could take it.
    assert [args[-1] for args in started.spawned[:-1]] == ["proxy", "supervise"]
    # The tunnel never points at ComfyUI itself, only at the authenticating proxy.
    assert started.spawned[-1][started.spawned[-1].index("--url") + 1] == "http://127.0.0.1:8189"
    assert set(json.loads((tmp_path / "processes.json").read_text())) == {"supervisor", "proxy", "tunnel"}
    with pytest.raises(RuntimeError, match="既に起動済みです"):
        runtime.start(config, "colab")  # One way at a time.
    # Another device: a new code, which replaces the first.
    runtime.issue_pairing_code(config)
    renewed = json.loads((tmp_path / "pairing.json").read_text())["code"]
    assert f"{renewed[:4]}-{renewed[4:]}" in capsys.readouterr().out
    monkeypatch.setattr(runtime.Path, "exists", lambda self: self.name == "processes.json")  # No /proc here.
    runtime.stop(config, save=False)
    monkeypatch.undo()
    assert json.loads((tmp_path / "session.json").read_text()) == {"drive_folder": "ComfyUI-Colab"}
    assert not (tmp_path / "pairing.json").exists()


def test_colab_start_opens_no_tunnel_and_admits_only_the_forwarded_address(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    started = StartedProcesses(tmp_path, monkeypatch, proxy_answer=502)
    asked = []
    address = "https://8189-gpu-t4-s-abc123-b.asia-east1-0.prod.colab.dev"
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(output=SimpleNamespace(
        eval_js=lambda script: asked.append(script) or address + "/")))
    config = {"drive_folder": "ComfyUI-Colab"}
    runtime.start(config, "colab")
    output = capsys.readouterr().out
    assert asked == ["google.colab.kernel.proxyPort(8189)"]
    saved = json.loads((tmp_path / "session.json").read_text())
    assert saved == {"drive_folder": "ComfyUI-Colab", "mode": "colab", "origins": [address]}
    assert [args[-1] for args in started.spawned] == ["proxy", "supervise"]  # Nothing is opened to the outside.
    assert set(json.loads((tmp_path / "processes.json").read_text())) == {"supervisor", "proxy"}
    assert address + "/\n" in output and "ペアリングコード" not in output and not (tmp_path / "pairing.json").exists()
    with pytest.raises(RuntimeError, match="6-2で起動してから"):
        runtime.issue_pairing_code(saved)
    monkeypatch.setattr(runtime.Path, "exists", lambda self: self.name == "processes.json")
    runtime.stop(config, save=False)
    monkeypatch.undo()
    assert json.loads((tmp_path / "session.json").read_text()) == {"drive_folder": "ComfyUI-Colab"}


def test_colab_start_stops_when_the_forwarded_address_is_not_usable(tmp_path, monkeypatch):
    from types import SimpleNamespace
    started = StartedProcesses(tmp_path, monkeypatch, proxy_answer=502)
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(output=SimpleNamespace(eval_js=lambda script: None)))
    with pytest.raises(RuntimeError, match="ポート転送のアドレスを取得できませんでした"):
        runtime.start({"drive_folder": "ComfyUI-Colab"}, "colab")
    assert started.spawned == [] and not (tmp_path / "processes.json").exists()
    with pytest.raises(ValueError):
        runtime.start({"drive_folder": "ComfyUI-Colab"}, "direct")


def test_connection_cells_start_one_way_each_and_only_the_tunnel_issues_codes():
    colab, tunnel = notebook_cell('runtime.start(config, "colab")'), notebook_cell('runtime.start(config, "tunnel")')
    assert colab.startswith("#@title 6-1. ") and tunnel.startswith("#@title 6-2. ")
    assert "#@param" not in colab and "issue_pairing_code" not in colab
    assert 'NEW_PAIRING_CODE = False #@param {type:"boolean"}' in tunnel and "runtime.issue_pairing_code(config)" in tunnel
    import build_notebook
    everything = "\n".join(cell["source"] for cell in build_notebook.build("EN")["cells"] if "write_text(" not in cell["source"])
    assert "access_key" not in everything and "runtime.start(config)" not in everything


ACCESS = {"hostname": "comfy.example.com", "team": "myteam.cloudflareaccess.com", "aud": "a1" * 32}


class AccessIssuer:
    """Cloudflare Access for the tests: signs tokens and publishes the keys to check them with."""
    def __init__(self):
        self.private, self.published, self.fetches = {}, [], 0
        self.add_key("key-1")

    def add_key(self, kid, publish=True):
        import jwt
        from cryptography.hazmat.primitives.asymmetric import rsa
        self.private[kid] = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        if publish:
            entry = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self.private[kid].public_key()))
            self.published.append({**entry, "kid": kid, "alg": "RS256", "use": "sig"})

    def token(self, kid="key-1", **claims):
        import jwt
        import time
        now = int(time.time())
        payload = {"aud": [ACCESS["aud"]], "iss": "https://" + ACCESS["team"], "iat": now, "nbf": now, "exp": now + 600,
                   "email": "me@example.com", "sub": "user-id", "type": "app", **claims}
        return jwt.encode({k: v for k, v in payload.items() if v is not None}, self.private[kid], algorithm="RS256", headers={"kid": kid})

    async def certs(self, request):
        from aiohttp import web
        self.fetches += 1
        return web.json_response({"keys": self.published, "public_certs": []})


def access_scenario(monkeypatch, steps):
    """Run steps(client, server, issuer, seen) against the proxy in Access mode with a recording backend."""
    from aiohttp import web, ClientSession
    from aiohttp.test_utils import TestServer
    issuer = AccessIssuer()

    async def scenario():
        seen = []

        async def backend(request):
            seen.append({"path": request.path, "headers": {k.lower(): v for k, v in request.headers.items()},
                         "cookies": dict(request.cookies)})
            if request.path == "/ws":
                ws = web.WebSocketResponse()
                await ws.prepare(request)
                await ws.send_str("status")
                return ws
            return web.Response(text="ComfyUI")

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", backend)
        cloudflare = web.Application()
        cloudflare.router.add_get("/cdn-cgi/access/certs", issuer.certs)
        async with TestServer(app) as upstream, TestServer(cloudflare) as team:
            monkeypatch.setattr(runtime, "access_certs_url", lambda domain: str(team.make_url("/cdn-cgi/access/certs")))
            config = {"mode": "access", "access": dict(ACCESS)}
            proxy = runtime.create_proxy(config, str(upstream.make_url("/")).rstrip("/"))
            async with TestServer(proxy) as server:
                async with ClientSession() as client:
                    await steps(client, server, issuer, seen)
    asyncio.run(scenario())
    return issuer


def test_access_mode_admits_only_requests_cloudflare_access_signed(monkeypatch):
    import base64
    import hashlib
    import hmac as hmac_module
    import time

    def forged(header, payload, secret=b""):
        encode = lambda data: base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()
        signed = encode(header) + "." + encode(payload)
        signature = hmac_module.new(secret, signed.encode(), hashlib.sha256).digest() if header["alg"] == "HS256" else b""
        return signed + "." + base64.urlsafe_b64encode(signature).rstrip(b"=").decode()

    async def steps(client, server, issuer, seen):
        async def status(token=None, route="/api/prompt", **headers):
            if token is not None:
                headers["Cf-Access-Jwt-Assertion"] = token
            async with client.get(server.make_url(route), headers=headers) as res:
                return res.status

        # Nothing without Access's token, and no other way in: no pairing page, no key, no cookie.
        assert await status() == 401
        assert await status(route="/", Accept="text/html") == 401
        assert await status(route=runtime.PAIRING_PATH) == 401
        assert await status(Cookie=runtime.COOKIE + "=") == 401
        async with client.post(server.make_url(runtime.PAIRING_PATH), data={"code": "12345678"}) as res:
            assert res.status == 401
        now = int(time.time())
        claims = {"aud": [ACCESS["aud"]], "iss": "https://" + ACCESS["team"], "iat": now, "exp": now + 600}
        issuer.add_key("rogue", publish=False)
        public_pem = b"-----BEGIN PUBLIC KEY-----"
        refused = {
            "garbage": "not-a-token",
            "expired": issuer.token(exp=now - 60),
            "another application": issuer.token(aud=["b2" * 32]),
            "another team": issuer.token(iss="https://other.cloudflareaccess.com"),
            "no expiry": issuer.token(exp=None),
            "signed by someone else": issuer.token(kid="rogue"),
            "unsigned": forged({"alg": "none", "kid": "key-1", "typ": "JWT"}, claims),
            "symmetric": forged({"alg": "HS256", "kid": "key-1", "typ": "JWT"}, claims, public_pem),
        }
        for reason, token in refused.items():
            assert await status(token) == 401, reason
        assert await status(issuer.token(padding="x" * 9000)) in (400, 401)  # Too long for a header at all.
        assert seen == []  # None of it reached ComfyUI.
        # The token Access adds to a signed-in visitor's requests.
        token = issuer.token()
        assert await status(token, Cookie="CF_Authorization=" + token + "; comfy-node=kept") == 200
        # ComfyUI and its custom nodes get neither the token nor Cloudflare's cookie.
        assert "cf-access-jwt-assertion" not in seen[0]["headers"] and seen[0]["cookies"] == {"comfy-node": "kept"}
        async with client.ws_connect(server.make_url("/ws"), headers={"Cf-Access-Jwt-Assertion": token}) as ws:
            assert (await ws.receive(timeout=3)).data == "status"
        # Being signed in does not let another site act through the browser.
        async with client.post(server.make_url("/prompt"), headers={"Cf-Access-Jwt-Assertion": token, "Origin": "https://evil.example"}) as res:
            assert res.status == 403
        for _ in range(20):
            assert await status(token) == 200
    issuer = access_scenario(monkeypatch, steps)
    assert issuer.fetches <= 2  # Keys are fetched once, and bad tokens cannot make it a flood.


def test_access_mode_follows_key_rotation(monkeypatch):
    async def steps(client, server, issuer, seen):
        async def status(token):
            async with client.get(server.make_url("/api/prompt"), headers={"Cf-Access-Jwt-Assertion": token}) as res:
                return res.status

        assert await status(issuer.token()) == 200 and issuer.fetches == 1
        # Cloudflare starts signing with a new key: it is looked up when first seen.
        issuer.add_key("key-2")
        assert await status(issuer.token(kid="key-2")) == 200 and issuer.fetches == 2
        assert await status(issuer.token(kid="key-2", sub="another-session")) == 200 and issuer.fetches == 2
        # Who may sign in is the Access policy's decision; a service token has no email at all.
        assert await status(issuer.token(email=None)) == 200
    monkeypatch.setattr(runtime, "ACCESS_KEYS_REFRESH", 0)
    access_scenario(monkeypatch, steps)


def test_access_settings_accept_what_the_dashboard_shows_and_refuse_the_rest():
    settings = runtime.access_settings(" https://Comfy.Example.com/ ", " " + ACCESS["aud"].upper() + "\n")
    assert settings == {"hostname": "comfy.example.com", "aud": ACCESS["aud"]}
    assert runtime.access_certs_url("myteam.cloudflareaccess.com") == "https://myteam.cloudflareaccess.com/cdn-cgi/access/certs"
    for arguments, error in (
            (("", ACCESS["aud"]), "公開ホスト名が不正"),
            (("localhost", ACCESS["aud"]), "公開ホスト名が不正"),
            (("comfy.example.com/path", ACCESS["aud"]), "公開ホスト名が不正"),
            (("comfy.example.com", "short"), "AUDタグが不正"),
            (("comfy.example.com", None), "CF_ACCESS_AUD")):
        with pytest.raises(ValueError, match=error):
            runtime.access_settings(*arguments)


def sign_in_redirect(url):
    from types import SimpleNamespace
    return SimpleNamespace(status_code=302, text="", headers={
        "Location": "https://myteam.cloudflareaccess.com/cdn-cgi/access/login/comfy.example.com?kid=" + ACCESS["aud"]})


def test_access_start_runs_the_named_tunnel_without_exposing_its_token(tmp_path, monkeypatch, capsys):
    started = StartedProcesses(tmp_path, monkeypatch, proxy_answer=401, public=sign_in_redirect)
    token = "eyJhIjoiYWNjb3VudCIsInQiOiJ0dW5uZWwiLCJzIjoic2VjcmV0In0="
    config = {"drive_folder": "ComfyUI-Colab"}
    for bad in ("", None, "cloudflared.exe service install " + token, "short"):
        with pytest.raises(ValueError, match="トンネルのトークンが不正"):
            runtime.start_with_access(config, "comfy.example.com", ACCESS["aud"], bad)
    assert started.spawned == []
    runtime.start_with_access(config, "comfy.example.com", ACCESS["aud"], " " + token + "\n")
    output = capsys.readouterr().out
    # The team the hostname sends visitors to is found and used without being shown.
    assert "cloudflareaccess.com" not in output
    saved = (tmp_path / "session.json").read_text()
    assert json.loads(saved) == {"drive_folder": "ComfyUI-Colab", "mode": "access", "access": ACCESS}
    assert [args[-1] for args in started.spawned] == ["proxy", "supervise", "run"]
    # Only cloudflared holds the token, in its environment: not its command line, a file or the output.
    assert started.environments[-1]["TUNNEL_TOKEN"] == token and started.environments[:-1] == [None, None]
    assert all(token not in " ".join(map(str, args)) for args in started.spawned)
    assert token not in saved and token not in output and token not in (tmp_path / "tunnel.log").read_text()
    assert ACCESS["aud"] not in output  # Kept with the other secrets, and never shown.
    assert "https://comfy.example.com/\n" in output and "ペアリングコード" not in output
    assert not (tmp_path / "pairing.json").exists()
    assert set(json.loads((tmp_path / "processes.json").read_text())) == {"supervisor", "proxy", "tunnel"}
    monkeypatch.setattr(runtime.Path, "exists", lambda self: self.name == "processes.json")
    runtime.stop(config, save=False)
    monkeypatch.undo()
    assert json.loads((tmp_path / "session.json").read_text()) == {"drive_folder": "ComfyUI-Colab"}


@pytest.mark.parametrize("answer", [
    {"status_code": 401, "headers": {}},                                      # Reaches the origin: no Access in front.
    {"status_code": 530, "headers": {}},                                      # No tunnel, and no Access either.
    {"status_code": 200, "headers": {}},                                      # Some other site answers there.
    {"status_code": 302, "headers": {"Location": "https://www.example.com/"}},  # A redirect, but not to a sign-in.
    {"status_code": 302, "headers": {"Location": "https://myteam.cloudflareaccess.com.evil.example/cdn-cgi/access/login/x"}},
    {"status_code": 302, "headers": {"Location": "http://myteam.cloudflareaccess.com/cdn-cgi/access/login/x"}},
    {"status_code": 302, "headers": {"Location": "https://myteam.cloudflareaccess.com/somewhere-else"}},
    None,                                                                     # The name does not resolve yet.
])
def test_access_start_does_nothing_unless_the_hostname_sends_visitors_to_access(tmp_path, monkeypatch, answer):
    from types import SimpleNamespace

    def public(url):
        assert url == "https://comfy.example.com/"
        if answer is None:
            raise OSError("name resolution failed")
        return SimpleNamespace(text="", **answer)

    started = StartedProcesses(tmp_path, monkeypatch, proxy_answer=401, public=public)
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)
    status = "-" if answer is None else str(answer["status_code"])
    with pytest.raises(RuntimeError, match=f"ログイン画面への転送を確認できませんでした（応答: {status}）"):
        runtime.start_with_access({"drive_folder": "ComfyUI-Colab"}, "comfy.example.com", ACCESS["aud"], "eyJ" + "a" * 60)
    # There is no team to check signatures against, so nothing runs and nothing is saved.
    assert started.spawned == [] and not (tmp_path / "session.json").exists() and not (tmp_path / "processes.json").exists()


def test_access_team_comes_from_the_sign_in_the_hostname_redirects_to(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    asked = []

    def public(url):
        asked.append(url)
        if len(asked) < 3:  # The DNS record or the Access application is still settling.
            return SimpleNamespace(status_code=530, text="", headers={})
        return SimpleNamespace(status_code=302, text="", headers={
            "Location": "https://another-team.cloudflareaccess.com/cdn-cgi/access/login/comfy.example.com?kid=x&redirect_url=%2F"})

    started = StartedProcesses(tmp_path, monkeypatch, proxy_answer=401, public=public)
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)
    assert runtime.access_team_domain("comfy.example.com") == "another-team.cloudflareaccess.com"
    assert asked == ["https://comfy.example.com/"] * 3
    runtime.start_with_access({"drive_folder": "ComfyUI-Colab"}, "comfy.example.com", ACCESS["aud"], "eyJ" + "a" * 60)
    saved = json.loads((tmp_path / "session.json").read_text())
    assert saved["access"] == {"hostname": "comfy.example.com", "aud": ACCESS["aud"], "team": "another-team.cloudflareaccess.com"}
    assert "cloudflareaccess.com" not in capsys.readouterr().out
    assert [args[-1] for args in started.spawned] == ["proxy", "supervise", "run"]
    # Running: a second start is refused before anything is asked or changed.
    asked.clear()
    with pytest.raises(RuntimeError, match="既に起動済みです"):
        runtime.start_with_access({"drive_folder": "ComfyUI-Colab"}, "comfy.example.com", ACCESS["aud"], "eyJ" + "a" * 60)
    assert asked == []


def test_access_cell_takes_secrets_from_colab_and_only_the_hostname_from_its_form():
    import build_notebook
    cell = notebook_cell("runtime.start_with_access(")
    assert cell.startswith("#@title 6-3. ")
    assert 'ACCESS_HOSTNAME = "" #@param {type:"string"}' in cell and cell.count("#@param") == 1
    assert "ACCESS_AUD =" not in cell and "ACCESS_EMAILS" not in cell and "ACCESS_TEAM_DOMAIN" not in cell
    assert "runtime.start_with_access(config, ACCESS_HOSTNAME, audience, token)" in cell
    assert 'userdata.get("CF_TUNNEL_TOKEN")' in cell and 'userdata.get("CF_ACCESS_AUD")' in cell
    # Neither secret stays in the notebook's variables.
    assert cell.rstrip().endswith('finally:\n    token = audience = ""')
    assert "http://127.0.0.1:8189" in cell  # Where the dashboard has to point the hostname: the proxy, never ComfyUI.
    unpack = notebook_cell("write_text(")
    assert '"PyJWT[crypto]>=2.8,<3"' in unpack.split("write_text(")[0]
    names = [cell["source"].split("\n")[0] for cell in build_notebook.build("EN")["cells"]]
    assert [name[8:12] for name in names if name.startswith("#@title 6-")] == ["6-1.", "6-2.", "6-3."]


def test_displayed_logs_hide_tunnel_and_access_tokens():
    token = "eyJhIjoiYWNjb3VudCIsInQiOiJ0dW5uZWwiLCJzIjoic2VjcmV0In0="
    signed = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImtleS0xIn0.eyJhdWQiOlsiYTEiXSwiZXhwIjoxfQ.c2lnbmF0dXJl-_abc"
    shown = runtime.redact(f"ERR token={token} Cf-Access-Jwt-Assertion: {signed} done")
    assert token not in shown and "c2lnbmF0dXJl" not in shown and shown.startswith("ERR token=[REDACTED] ") and shown.endswith(" done")


def test_tunnel_settings_are_read_from_what_cloudflared_reports():
    proxy, fallback = "http://127.0.0.1:8189", "http_status:404"
    assert runtime.tunnel_services("INF Starting tunnel\nINF Registered tunnel connection connIndex=0\n") is None
    assert runtime.tunnel_services(tunnel_settings(proxy, fallback)) == [proxy, fallback]
    # The latest settings count: the dashboard can change them while the tunnel runs.
    log = tunnel_settings(proxy, fallback) + "INF Registered tunnel connection\n" + tunnel_settings("http://localhost:8188", fallback, version=2)
    assert runtime.tunnel_services(log) == ["http://localhost:8188", fallback]
    # The same message when cloudflared writes its log as JSON.
    as_json = json.dumps({"level": "info", "config": json.dumps({"ingress": [{"service": proxy}]}), "version": 1,
                          "message": "Updated to new configuration"})
    assert runtime.tunnel_services(as_json + "\n") == [proxy]
    # Settings that cannot be read are unknown, not accepted.
    assert runtime.tunnel_services('INF Updated to new configuration config="{broken" version=3\n') is None
    assert runtime.tunnel_services('INF Updated to new configuration config="{\\"ingress\\":null}" version=3\n') is None
    for services, accepted in (
            ([proxy, fallback], True), (["http://localhost:8189/", "http_status:503"], True), ([proxy, proxy], True),
            (["http://127.0.0.1:8188", fallback], False),      # ComfyUI itself, past every check.
            ([proxy, "http://127.0.0.1:9000", fallback], False),  # Another service of the runtime on a second hostname.
            (["https://127.0.0.1:8189", fallback], False), (["http://127.0.0.1:81890", fallback], False),
            (["ssh://localhost:22"], False), (["hello_world"], False), ([fallback], False), ([], False)):
        assert runtime.tunnel_leads_to_proxy(services) is accepted, services


@pytest.mark.parametrize("services", [("http://localhost:8188", "http_status:404"),
                                      ("http://127.0.0.1:8189", "http://127.0.0.1:9000", "http_status:404")])
def test_access_start_stops_when_the_tunnel_leads_anywhere_but_the_proxy(tmp_path, monkeypatch, capsys, services):
    started = StartedProcesses(tmp_path, monkeypatch, proxy_answer=401, public=sign_in_redirect,
                               tunnel_log="INF Registered tunnel connection connIndex=0\n" + tunnel_settings(*services))
    actual = runtime.Path.exists
    monkeypatch.setattr(runtime.Path, "exists", lambda self: tmp_path in self.parents and actual(self))  # No /proc here.
    with pytest.raises(RuntimeError, match="転送先が http://127.0.0.1:8189 だけになっていません（現在の設定: " + ", ".join(services)):
        runtime.start_with_access({"drive_folder": "ComfyUI-Colab"}, "comfy.example.com", ACCESS["aud"], "eyJ" + "a" * 60)
    monkeypatch.undo()
    # The tunnel was up for a moment: it is taken down again, and no address is shown.
    assert [args[-1] for args in started.spawned] == ["proxy", "supervise", "run"]
    assert "mode" not in json.loads((tmp_path / "session.json").read_text()) and not (tmp_path / "processes.json").exists()
    assert "https://comfy.example.com/" not in capsys.readouterr().out


def test_access_start_says_so_when_the_tunnel_settings_cannot_be_read(tmp_path, monkeypatch, capsys):
    StartedProcesses(tmp_path, monkeypatch, proxy_answer=401, public=sign_in_redirect,
                     tunnel_log="INF Registered tunnel connection connIndex=0\n")
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)
    runtime.start_with_access({"drive_folder": "ComfyUI-Colab"}, "comfy.example.com", ACCESS["aud"], "eyJ" + "a" * 60)
    output = capsys.readouterr().out
    assert "トンネルの転送先の設定を読み取れませんでした" in output and "https://comfy.example.com/" in output


def test_proxy_adds_the_stop_button_to_the_comfyui_page_and_nothing_else(private, monkeypatch):
    monkeypatch.setattr(runtime, "REPOSITORY_URL", runtime.REPOSITORY_URL)  # Put back after the test changes it.
    from aiohttp import web, ClientSession, CookieJar
    from aiohttp.test_utils import TestServer
    import gzip

    async def scenario():
        page = b"<!doctype html><html><HEAD><title>ComfyUI</title></HEAD><body><div id='vue-app'></div></body></html>"
        asked = []

        async def backend(request):
            asked.append((request.path, request.headers.get("Accept-Encoding")))
            if request.path == "/packed":
                return web.Response(body=gzip.compress(page), content_type="text/html", headers={"Content-Encoding": "gzip"})
            if request.path.startswith("/api/"):
                return web.json_response({"html": "</head>"})
            return web.Response(body=page, content_type="text/html", headers={"ETag": '"abc"'})

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", backend)
        tag = b'<script defer src="/_colab/stop.js"></script></HEAD>'
        async with TestServer(app) as upstream:
            config = {"mode": "tunnel", "cookie_secret": "test-cookie-secret"}
            proxy = runtime.create_proxy(config, str(upstream.make_url("/")).rstrip("/"), secure_cookie=False)
            async with TestServer(proxy) as server:
                async with ClientSession(cookie_jar=CookieJar(unsafe=True), auto_decompress=False) as client:
                    signed_in(client, server)
                    for route in ("/", "/index.html"):
                        async with client.get(server.make_url(route), headers={"Accept-Encoding": "identity"}) as res:
                            body = await res.read()
                            # (A local filtering proxy may add a script of its own to HTML, so no exact comparison.)
                            assert res.status == 200 and body.count(tag) == 1 and body.endswith(tag + page.split(b"</HEAD>")[1])
                            assert body.startswith(page.split(b"</HEAD>")[0]) and res.headers["Cache-Control"] == "no-store"
                    # The page still travels compressed, and ComfyUI is asked for it plain so it can be changed.
                    async with client.get(server.make_url("/"), headers={"Accept-Encoding": "gzip, br"}) as res:
                        packed = await res.read()
                        assert (gzip.decompress(packed) if res.headers.get("Content-Encoding") == "gzip" else packed).count(tag) == 1
                    assert [encoding for path, encoding in asked if path in ("/", "/index.html")] == ["identity"] * 3
                    # Only that page: not other HTML, not data that mentions a head, not a page that arrives packed.
                    for route in ("/docs/page.html", "/packed"):
                        async with client.get(server.make_url(route), headers={"Accept-Encoding": "identity"}) as res:
                            assert b"/_colab/stop.js" not in await res.read()
                    async with client.get(server.make_url("/api/x"), headers={"Accept-Encoding": "identity"}) as res:
                        assert await res.json() == {"html": "</head>"}
                    async with client.post(server.make_url("/"), data=b"x") as res:
                        assert b"/_colab/stop.js" not in await res.read()
                    # The script: the button's texts in this language, and only for a browser that is let in.
                    async with client.get(server.make_url(runtime.STOP_SCRIPT_PATH), headers={"Accept-Encoding": "identity"}) as res:
                        script = await res.text()
                        assert res.status == 200 and res.content_type == "application/javascript"
                        for text in ("停止", "接続を解除してランタイムを削除", "接続を解除", "キャンセル"):
                            assert json.dumps(text) in script
                        assert "__TEXTS__" not in script and "__STOP_PATH__" not in script and '"/_colab/stop"' in script
                        # The link to the repository opens outside the page and tells that page nothing.
                        assert f'const repository = "{runtime.REPOSITORY_URL}";' in script and "__REPOSITORY__" not in script
                        assert runtime.REPOSITORY_URL.startswith("https://github.com/") and 'rel = "noopener noreferrer"' in script
                        assert json.dumps(runtime.t("menu.repository")) in script
                async with ClientSession() as stranger:
                    for route in (runtime.STOP_SCRIPT_PATH, runtime.STOP_PATH):
                        async with stranger.get(server.make_url(route)) as res:
                            assert res.status == 401
                    async with stranger.post(server.make_url(runtime.STOP_PATH), json={"delete_runtime": True}) as res:
                        assert res.status == 401
        assert not (private / "stop-request.json").exists()
        assert runtime.with_stop_button(b"no head here") == b"no head here"
        for address in ("", "javascript:alert(1)", "http://github.com/x/y"):  # Nothing but an https address is linked.
            runtime.REPOSITORY_URL = address
            assert 'const repository = "";' in runtime.stop_button_script()
    asyncio.run(scenario())


def test_stop_button_leaves_a_request_only_for_a_signed_in_browser_of_this_site(private):
    from aiohttp import web, ClientSession, CookieJar
    from aiohttp.test_utils import TestServer

    async def scenario():
        reached = []

        async def backend(request):
            reached.append(request.path)
            return web.Response(text="ComfyUI")

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", backend)
        request_file = private / "stop-request.json"
        async with TestServer(app) as upstream:
            config = {"mode": "tunnel", "cookie_secret": "test-cookie-secret"}
            proxy = runtime.create_proxy(config, str(upstream.make_url("/")).rstrip("/"), secure_cookie=False)
            async with TestServer(proxy) as server:
                async with ClientSession(cookie_jar=CookieJar(unsafe=True)) as client:
                    signed_in(client, server)
                    stop = server.make_url(runtime.STOP_PATH)
                    async with client.get(stop) as res:
                        assert res.status == 200 and await res.json() == {"state": "idle"}
                    # Another site open in the same browser cannot stop anything, or delete the runtime.
                    async with client.post(stop, json={"delete_runtime": True}, headers={"Origin": "https://evil.example"}) as res:
                        assert res.status == 403
                    for body in (b"", b"not json", b"[]", b'{"delete_runtime": "yes"}', b'{"delete_runtime": true}' + b" " * 300):
                        async with client.post(stop, data=body) as res:
                            assert res.status == 400, body
                    assert not request_file.exists()
                    async with client.post(stop, json={"delete_runtime": False}) as res:
                        assert res.status == 202 and await res.json() == {"state": "pending"}
                    assert json.loads(request_file.read_text()) == {"delete_runtime": False}
                    async with client.post(stop, json={"delete_runtime": True}) as res:
                        assert res.status == 202
                    assert json.loads(request_file.read_text()) == {"delete_runtime": True}
                    # The notebook could not save: the page is told, and may ask again.
                    request_file.unlink()
                    runtime.private_write(private / "stop-status.json", '{"state": "failed"}')
                    async with client.get(stop) as res:
                        assert await res.json() == {"state": "failed"}
                    async with client.post(stop, json={"delete_runtime": False}) as res:
                        assert await res.json() == {"state": "pending"}
        assert reached == []  # None of this is ComfyUI's business.
    asyncio.run(scenario())


def test_stop_request_is_carried_out_by_the_notebook_and_deletes_the_runtime_only_when_asked(private, monkeypatch, capsys):
    done = []
    monkeypatch.setattr(runtime, "stop", lambda config, save=True: done.append(("stop", save)))
    monkeypatch.setattr(runtime, "delete_colab_runtime", lambda: done.append("delete"))
    config = {"drive_folder": "ComfyUI-Colab", "mode": "tunnel"}
    assert runtime.carry_out_stop_request(config) is False and done == []  # Nothing was asked.
    runtime.request_stop(False)
    assert runtime.stop_request_state() == "pending"
    assert runtime.carry_out_stop_request(config) is True and done == [("stop", True)]
    done.clear()
    runtime.request_stop(True)
    assert runtime.carry_out_stop_request(config) is True and done == [("stop", True), "delete"]  # Saved first.

    def failing_stop(config, save=True):
        raise RuntimeError("Drive is full")
    done.clear()
    monkeypatch.setattr(runtime, "stop", failing_stop)
    runtime.request_stop(True)
    # The save failed: ComfyUI keeps running, the page hears about it, and the runtime is not deleted.
    assert runtime.carry_out_stop_request(config) is False and done == []
    assert runtime.stop_request_state() == "failed" and not (private / "stop-request.json").exists()
    assert "停止できませんでした" in capsys.readouterr().out
    assert runtime.carry_out_stop_request(config) is False  # Not tried again until asked again.
    runtime.request_stop(False)
    assert runtime.stop_request_state() == "pending"


def test_starting_watches_for_the_stop_button_and_stopping_ends_the_watch(tmp_path, monkeypatch):
    import threading
    started = StartedProcesses(tmp_path, monkeypatch, proxy_answer=401)
    runtime.private_write(tmp_path / "stop-request.json", '{"delete_runtime": true}')  # Left over from before.
    config = {"drive_folder": "ComfyUI-Colab"}
    runtime.start(config, "tunnel")
    assert started.followed == [config] and not (tmp_path / "stop-request.json").exists()
    # The watch itself: it keeps looking until the request is carried out or the notebook stops.
    monkeypatch.undo()
    monkeypatch.setattr(runtime, "PRIVATE", tmp_path)
    monkeypatch.setattr(runtime, "CONFIG", tmp_path / "session.json")
    looked = threading.Event()
    monkeypatch.setattr(runtime, "carry_out_stop_request", lambda config: looked.set())
    monkeypatch.setattr(runtime, "STOP_WATCH_SECONDS", 0.05)
    before = threading.active_count()
    runtime.follow_stop_requests(config)
    assert looked.wait(5)
    watch = runtime.STOP_WATCH[0]
    monkeypatch.setattr(runtime.os, "killpg", lambda pid, sig: None, raising=False)
    runtime.private_write(tmp_path / "stop-request.json", '{"delete_runtime": false}')
    runtime.stop(config, save=False)
    assert watch.is_set() and not (tmp_path / "stop-request.json").exists() and not (tmp_path / "stop-status.json").exists()
    for _ in range(50):
        if threading.active_count() <= before:
            break
        threading.Event().wait(0.1)
    assert threading.active_count() <= before


@pytest.mark.parametrize("config", [{"mode": "tunnel", "cookie_secret": "test-cookie-secret"},
                                    {"mode": "colab", "origins": ["https://8189-abc.prod.colab.dev"]}])
def test_proxy_takes_only_the_opening_of_the_page_from_other_sites(config, capsys):
    from aiohttp import web, ClientSession, CookieJar
    from aiohttp.test_utils import TestServer

    async def scenario():
        reached = []

        async def backend(request):
            # A ComfyUI from before it refused cross-site requests itself.
            reached.append((request.method, request.path, request.headers.get("Sec-Fetch-Site")))
            return web.Response(text="ComfyUI", content_type="text/html")

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", backend)
        async with TestServer(app) as upstream:
            proxy = runtime.create_proxy(config, str(upstream.make_url("/")).rstrip("/"), secure_cookie=False)
            async with TestServer(proxy) as server:
                async with ClientSession(cookie_jar=CookieJar(unsafe=True)) as client:
                    signed_in(client, server)

                    async def status(method, route, **fetch):
                        headers = {"Sec-Fetch-" + name.capitalize(): value for name, value in fetch.items()}
                        async with client.request(method, server.make_url(route), headers=headers) as res:
                            return res.status

                    # What a page on another site can make this browser send, with its sign-in attached.
                    for site in ("cross-site", "same-site"):
                        assert await status("GET", "/api/manager/reboot", site=site, mode="navigate", dest="document") == 403  # A link.
                        assert await status("GET", "/api/view?filename=x.png", site=site, mode="no-cors", dest="image") == 403  # An image.
                        assert await status("GET", "/", site=site, mode="navigate", dest="iframe") == 403  # A frame.
                        assert await status("GET", "/", site=site, mode="no-cors", dest="script") == 403
                        assert await status("POST", "/", site=site, mode="navigate", dest="document") == 403  # A form.
                        assert await status("GET", runtime.STOP_PATH, site=site, mode="cors", dest="empty") == 403
                    assert reached == []
                    # Opening the page from a link elsewhere (Colab's output, a sign-in page) is the one thing taken.
                    for route in ("/", "/?workflow=1", "/index.html"):
                        assert await status("GET", route, site="cross-site", mode="navigate", dest="document") == 200
                    assert [entry[2] for entry in reached if entry[1] == "/"] == [None, None]  # ComfyUI is not told to refuse it.
                    # The page's own requests, an address typed in, and programs that are not browsers.
                    assert await status("GET", "/api/queue", site="same-origin", mode="cors", dest="empty") == 200
                    assert await status("POST", "/api/prompt", site="same-origin", mode="cors", dest="empty") == 200
                    assert await status("GET", "/api/view?filename=x.png", site="none", mode="navigate", dest="document") == 200
                    assert await status("GET", "/api/view?filename=x.png") == 200
    asyncio.run(scenario())
    assert "別のサイトからの要求を拒否しました: GET /api/manager/reboot（Sec-Fetch-Site: cross-site）" in capsys.readouterr().out


def test_proxy_keeps_its_sign_in_cookie_and_its_pages_out_of_other_hands(capsys):
    from aiohttp import web, ClientSession, CookieJar
    from aiohttp.test_utils import TestServer

    async def scenario():
        async def backend(request):
            response = web.Response(text="<html><head></head><body>ComfyUI</body></html>", content_type="text/html")
            if request.path == "/node":
                # A custom node answering with cookies of its own choosing.
                for cookie in ("comfy_colab=replaced; Path=/", runtime.COOKIE + "=replaced; Path=/; Secure",
                               "CF_Authorization=replaced; Path=/", "node-setting=kept; Path=/"):
                    response.headers.add("Set-Cookie", cookie)
                response.headers["Content-Security-Policy"] = "img-src 'self'"
            if request.path == "/framed-by-comfyui":
                response.headers["X-Frame-Options"] = "DENY"
            return response

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", backend)
        async with TestServer(app) as upstream:
            config = {"mode": "tunnel", "cookie_secret": "test-cookie-secret"}
            proxy = runtime.create_proxy(config, str(upstream.make_url("/")).rstrip("/"), secure_cookie=False)
            async with TestServer(proxy) as server:
                async with ClientSession(cookie_jar=CookieJar(unsafe=True)) as client:
                    signed_in(client, server)
                    async with client.get(server.make_url("/node")) as res:
                        assert res.headers.getall("Set-Cookie") == ["node-setting=kept; Path=/"]
                        # (A local filtering proxy may merge or rewrite policies, so look for the directive.)
                        policies = " ; ".join(res.headers.getall("Content-Security-Policy"))
                        assert "frame-ancestors 'self'" in policies and "img-src 'self'" in policies
                    # Still signed in: the node did not replace the cookie.
                    for route in ("/", "/api/queue", "/assets/index-CrctFnrp.js"):
                        async with client.get(server.make_url(route)) as res:
                            assert res.status == 200
                            assert "frame-ancestors 'self'" in " ; ".join(res.headers.getall("Content-Security-Policy"))
                            assert res.headers["X-Frame-Options"] == "SAMEORIGIN"
                    async with client.get(server.make_url("/framed-by-comfyui")) as res:
                        assert res.headers.getall("X-Frame-Options") == ["DENY"]  # ComfyUI's stricter word stands.
                    # A signed-in visitor's refused requests are noted, but only so many a minute.
                    for _ in range(50):
                        async with client.post(server.make_url("/prompt"), headers={"Origin": "https://evil.example"}) as res:
                            assert res.status == 403
                async with ClientSession() as stranger:
                    for _ in range(5):
                        async with stranger.post(server.make_url(runtime.PAIRING_PATH), data={"code": "1"},
                                                 headers={"Origin": "https://stranger.example"}) as res:
                            assert res.status == 403
    asyncio.run(scenario())
    log = capsys.readouterr().out
    assert log.count("Origin https://evil.example") == 30 and "stranger.example" not in log


@pytest.fixture
def view_commands(storage, monkeypatch):
    """Record what preparing the read-only view of Drive's models runs, without running it."""
    state = {"mounted": False, "installed": False, "commands": [], "fail": None}

    def run(args, **kwargs):
        args = list(map(str, args))
        state["commands"].append((args, bool(kwargs.get("app"))))
        if state["fail"] and state["fail"] in args:
            raise RuntimeError("fuse: failed to open /dev/fuse: Permission denied")
        if args[:2] == ["apt-get", "install"]:
            state["installed"] = True
        if args[0] == "bindfs":
            state["mounted"] = True

    def unmount(args, **kwargs):
        state["commands"].append((list(map(str, args)), False))
        state["mounted"] = False

    monkeypatch.setattr(runtime, "DRIVE_MODELS", storage_root(storage) / "drive-models")
    monkeypatch.setattr(runtime, "run", run)
    monkeypatch.setattr(runtime.subprocess, "run", unmount)
    monkeypatch.setattr(runtime.shutil, "which", lambda name: "/usr/bin/bindfs" if state["installed"] else None)
    monkeypatch.setattr(runtime.os.path, "ismount", lambda path: state["mounted"])
    monkeypatch.setattr(runtime.os, "geteuid", lambda: runtime.DRIVE_MODELS.stat().st_uid, raising=False)
    monkeypatch.setattr(runtime, "APT_INDEX_REFRESHED", [])
    return state


def storage_root(storage):
    return runtime.PRIVATE.parent


def test_direct_access_shows_the_application_only_the_models_folder_read_only(storage, view_commands):
    runtime.show_drive_models(storage)
    commands = view_commands["commands"]
    source = str(runtime.drive_root(storage) / "models")
    # bindfs is installed without the FUSE tools it merely recommends: Drive is mounted with the ones in place.
    assert commands[:2] == [(["apt-get", "update", "-qq"], False),
                            (["apt-get", "install", "-y", "-qq", "--no-install-recommends", "bindfs"], False)]
    mount, as_root = commands[2]
    assert not as_root and mount[0] == "bindfs" and mount[-2:] == [source, str(runtime.DRIVE_MODELS)]
    assert "-r" in mount and "--force-user=comfy" in mount and "--perms=a=rX" in mount and mount[mount.index("-o") + 1] == "allow_other"
    # Not Drive, not the folder above the models: only that one folder.
    assert Path(source).name == "models" and Path(source).parent.name == "ComfyUI-Colab"
    # The application user is made to look in before anything relies on it.
    assert commands[3] == (["ls", str(runtime.DRIVE_MODELS)], True)
    assert runtime.DRIVE_MODELS.parent != runtime.BASE and runtime.BASE not in runtime.DRIVE_MODELS.parents
    commands.clear()
    runtime.show_drive_models(storage)  # Already there: nothing is run again.
    assert commands == []
    runtime.hide_drive_models()
    assert commands == [(["umount", str(runtime.DRIVE_MODELS)], False)] and not view_commands["mounted"]


@pytest.mark.parametrize("failing", ["bindfs", "ls"])
def test_direct_access_that_cannot_be_prepared_says_how_to_copy_instead(storage, view_commands, failing):
    view_commands["fail"] = failing
    with pytest.raises(RuntimeError) as failure:
        runtime.show_drive_models(storage)
    assert "MODEL_ACCESS" in str(failure.value) and "copy" in str(failure.value) and "Permission denied" in str(failure.value)
    assert not view_commands["mounted"]  # Nothing half-mounted is left.


def test_direct_access_refuses_a_place_someone_else_prepared(storage, view_commands):
    runtime.DRIVE_MODELS.mkdir()
    (runtime.DRIVE_MODELS / "planted").write_text("x")
    with pytest.raises(RuntimeError, match="MODEL_ACCESS"):
        runtime.show_drive_models(storage)
    assert not any(args[0] == "bindfs" for args, _ in view_commands["commands"])


def test_models_are_linked_or_copied_as_setup_chose(storage, view_commands, monkeypatch, capsys):
    asked = []
    monkeypatch.setattr(runtime, "app_action", lambda action, *args, **kwargs: asked.append((action, *map(str, args))) or b"3")
    models = runtime.drive_root(storage) / "models"
    (models / "loras").mkdir()
    (models / "loras/style.safetensors").write_bytes(b"lora")
    assert runtime.sync_models({**storage, "model_access": "direct"}) == 3
    assert asked == [("link-models", str(runtime.DRIVE_MODELS))] and view_commands["mounted"]
    assert runtime.t("models.linked", linked=3) in capsys.readouterr().out
    asked.clear()
    assert runtime.sync_models({**storage, "model_access": "copy"}) == 1  # As before: a copy in the runtime's cache.
    assert [entry[:2] for entry in asked] == [("put", "models/loras/style.safetensors")]
    asked.clear()
    # A runtime set up before the choice existed copies too.
    assert runtime.sync_models(storage) == 1 and asked[0][0] == "put"


def test_links_to_drive_models_follow_drive_and_leave_local_models_alone(storage, tmp_path):
    view, cache = tmp_path / "view", runtime.BASE / "models"
    (view / "loras/sdxl").mkdir(parents=True)
    (view / "loras/sdxl/style.safetensors").write_bytes(b"on drive")
    (view / "loras/placeholder").write_bytes(b"")
    (view / "loras/upload.part").write_bytes(b"half")
    (view / "checkpoints").mkdir()
    (view / "checkpoints/base.safetensors").write_bytes(b"drive copy")
    (cache / "checkpoints").mkdir(parents=True)
    (cache / "checkpoints/base.safetensors").write_bytes(b"downloaded on this runtime")
    try:
        (tmp_path / "probe").symlink_to(view)
    except OSError:
        pytest.skip("this account may not create symbolic links")
    sink = io.BytesIO()
    runtime.app_tree_action("link-models", [str(view)], None, sink)
    link = cache / "loras/sdxl/style.safetensors"
    assert sink.getvalue() == b"1" and link.is_symlink() and link.read_bytes() == b"on drive"
    # Not markers or unfinished uploads, and never in place of a model this runtime has itself.
    assert not (cache / "loras/placeholder").exists() and not (cache / "loras/upload.part").exists()
    assert not (cache / "checkpoints/base.safetensors").is_symlink()
    assert (cache / "checkpoints/base.safetensors").read_bytes() == b"downloaded on this runtime"
    assert [file.name for file in runtime.regular_files(cache)] == ["base.safetensors"]  # Links are not saved to Drive.
    # Run again: nothing new. Removed from Drive: the link goes, other links and files stay.
    (view / "loras/other.safetensors").write_bytes(b"another")
    sink = io.BytesIO()
    runtime.app_tree_action("link-models", [str(view)], None, sink)
    assert sink.getvalue() == b"1"
    (view / "loras/sdxl/style.safetensors").unlink()
    sink = io.BytesIO()
    runtime.app_tree_action("link-models", [str(view)], None, sink)
    assert sink.getvalue() == b"0" and not os.path.lexists(link) and (cache / "loras/other.safetensors").is_symlink()
    assert (cache / "checkpoints/base.safetensors").exists()


def test_notebook_offers_the_choice_and_releases_the_view_before_unmounting_drive(monkeypatch):
    settings, setup, stop = notebook_cell("DRIVE_FOLDER_NAME"), notebook_cell("runtime.setup("), notebook_cell("STOP_NOW = False")
    assert 'MODEL_ACCESS = "direct" #@param ["direct", "copy"]' in settings
    assert "model_access=MODEL_ACCESS" in setup
    assert stop.index("runtime.hide_drive_models()") < stop.index("drive.flush_and_unmount()")
    order = []
    monkeypatch.setattr(runtime, "hide_drive_models", lambda: order.append("hide"))
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(
        drive=SimpleNamespace(flush_and_unmount=lambda: order.append("unmount")),
        runtime=SimpleNamespace(unassign=lambda: order.append("unassign"))))
    runtime.delete_colab_runtime()
    assert order == ["hide", "unmount", "unassign"]


def test_download_progress_is_a_bar_in_the_notebook_and_plain_lines_elsewhere(monkeypatch, capsys):
    from types import SimpleNamespace
    megabyte = 1048576
    monkeypatch.setitem(sys.modules, "tqdm", None)
    monkeypatch.setitem(sys.modules, "tqdm.auto", None)  # Not installed.
    plain = runtime.download_progress(200 * megabyte)
    for _ in range(50):
        plain.update(4 * megabyte)
    plain.close()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 10 and lines[0] == runtime.t("download.progress", done=20, total=200, percent=10)
    assert lines[-1] == runtime.t("download.progress", done=200, total=200, percent=100)
    unknown = runtime.download_progress(0)  # The site did not say how large the file is.
    for _ in range(300):
        unknown.update(4 * megabyte)
    assert capsys.readouterr().out.splitlines() == [runtime.t("download.progress_size", done=512), runtime.t("download.progress_size", done=1024)]
    bars = []

    class Bar:
        def __init__(self, **options):
            self.options, self.updates, self.closed = options, [], False
            bars.append(self)

        def update(self, amount):
            self.updates.append(amount)

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "tqdm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "tqdm.auto", SimpleNamespace(tqdm=Bar))
    bar = runtime.download_progress(200 * megabyte)
    assert bar is bars[0] and bar.options["total"] == 200 * megabyte and bar.options["unit"] == "B" and bar.options["unit_scale"]
    assert runtime.download_progress(0).options["total"] is None


@pytest.mark.parametrize("failing", [False, True])
def test_model_download_reports_its_progress_and_always_ends_the_bar(storage, monkeypatch, failing):
    bars = []

    class Bar:
        def __init__(self, total):
            self.total, self.done, self.closed = total, 0, False
            bars.append(self)

        def update(self, amount):
            self.done += amount

        def close(self):
            self.closed = True

    class Response:
        status_code = 200
        headers = {"Content-Type": "application/octet-stream", "Content-Length": "9"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_content(self, size):
            yield b"abc"
            yield b"def"
            if failing:
                raise OSError("connection reset")
            yield b"ghi"

    monkeypatch.setattr(runtime, "download_progress", Bar)
    monkeypatch.setattr(runtime, "checked_https_url", lambda url: SimpleNamespaceHost())
    session = type("Session", (), {"get": lambda self, url, **kwargs: Response()})()
    target = runtime.PRIVATE / "model.part"
    if failing:
        with pytest.raises(OSError):
            runtime.stream_model(session, "https://huggingface.co/a/b/resolve/main/m.safetensors", target)
    else:
        digest, _ = runtime.stream_model(session, "https://huggingface.co/a/b/resolve/main/m.safetensors", target)
        import hashlib
        assert digest == hashlib.sha256(b"abcdefghi").hexdigest()
    assert bars[0].total == 9 and bars[0].done == (6 if failing else 9) and bars[0].closed


class SimpleNamespaceHost:
    hostname = "huggingface.co"


def test_model_for_this_runtime_only_never_reaches_drive(storage, monkeypatch, capsys):
    drive_models = runtime.drive_root(storage) / "models"
    link = "https://huggingface.co/author/repo/blob/main/split/model.safetensors"
    fake_requests(monkeypatch, [{"data": b"runtime model"}])
    runtime.fetch_model(storage, link, "vae", runtime_only=True)
    local = runtime.BASE / "models/vae/model.safetensors"
    assert local.read_bytes() == b"runtime model"
    assert not (drive_models / "vae").exists() and list(drive_models.rglob("*")) == []
    assert runtime.t("download.saved_runtime") in capsys.readouterr().out
    # Nothing unfinished stays behind where the download was checked.
    assert list((runtime.PRIVATE / "downloads").iterdir()) == []
    # Saving the environment sends the runtime's new models to Drive, but not this one.
    (runtime.BASE / "models/vae/from-manager.safetensors").write_bytes(b"downloaded in ComfyUI")
    runtime.save_new_cache_models(storage, force=True)
    assert sorted(p.name for p in (drive_models / "vae").iterdir()) == ["from-manager.safetensors"]
    # The same name again is refused before and after downloading, and a wrong hash leaves nothing.
    fake_requests(monkeypatch, [{"data": b"again"}])
    with pytest.raises(FileExistsError, match=re.escape(runtime.t("download.exists_runtime"))):
        runtime.fetch_model(storage, link, "vae", runtime_only=True)
    fake_requests(monkeypatch, [{"data": b"served", "headers": {"Content-Disposition": 'attachment; filename="model.safetensors"'}}])
    with pytest.raises(FileExistsError, match=re.escape(runtime.t("download.exists_runtime"))):
        runtime.fetch_model(storage, "https://civitai.com/api/download/models/12345", "vae", runtime_only=True)
    fake_requests(monkeypatch, [{"data": b"tampered"}])
    with pytest.raises(RuntimeError, match="SHA256"):
        runtime.fetch_model(storage, link, "loras", "checked.safetensors", "0" * 64, runtime_only=True)
    assert not (runtime.BASE / "models/loras/checked.safetensors").exists()
    assert list((runtime.PRIVATE / "downloads").iterdir()) == [] and local.read_bytes() == b"runtime model"
    # Without the option the same link goes to Drive as it always did.
    fake_requests(monkeypatch, [{"data": b"kept model"}])
    runtime.fetch_model(storage, link, "loras", "kept.safetensors")
    assert (drive_models / "loras/kept.safetensors").read_bytes() == b"kept model"


def test_notebook_cells_for_setup_download_and_status_are_as_a_user_needs_them():
    import build_notebook
    setup, download, status = notebook_cell("runtime.setup("), notebook_cell("runtime.fetch_model("), notebook_cell("RUN_CONNECTION_DIAGNOSIS")
    # The long setup log scrolls inside a bounded area, set before anything is printed.
    assert setup.index("setIframeHeight(0, true, {maxHeight: 300})") < setup.index("runtime.setup(")
    assert 'RUNTIME_ONLY = False #@param {type:"boolean"}' in download and "runtime_only=RUNTIME_ONLY" in download
    assert "RUN_ISOLATION_CHECK" not in status and "isolation_report" not in status
    # Nothing in the notebook's own text refers to how it used to work.
    for language in ("JP", "EN"):
        text = "\n".join(cell["source"] for cell in build_notebook.build(language)["cells"] if "write_text(" not in cell["source"])
        for phrase in ("従来", "これまで", "as before", "used to"):
            assert phrase not in text, phrase
