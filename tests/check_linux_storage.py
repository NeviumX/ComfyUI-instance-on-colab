"""Run as root on Linux: privilege separation and checkpoint transaction checks.

Uses temporary simulated Drive storage, not a Google account or GPU. Creates the
application user (`comfy`) if it does not exist.
"""
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import colab_runtime as runtime

if sys.platform != "linux" or os.geteuid() != 0:
    raise SystemExit("Run with a Linux root user; only private temporary test files are created.")


def as_app(code, *args):
    """Run Python as the application user, the way ComfyUI and its custom nodes run."""
    return subprocess.run(runtime.app_command([sys.executable, "-c", code, *map(str, args)]),
                          env=runtime.application_environment(), capture_output=True, text=True)


with tempfile.TemporaryDirectory(prefix="comfy-colab-check-") as temporary:
    root = Path(temporary)
    root.chmod(0o755)
    runtime.BASE = root / "session"
    runtime.PRIVATE = root / "private"
    runtime.DRIVE_MOUNT = runtime.PRIVATE / "drive"
    runtime.PRIVATE.mkdir(mode=0o700)
    config = {"drive_folder": "ComfyUI-Colab", "python": "3.12"}
    models = runtime.DRIVE_MOUNT / "MyDrive/ComfyUI-Colab/models"
    models.mkdir(parents=True)
    (models / "manual.safetensors").write_bytes(b"manual model")
    (models / "manual.safetensors").chmod(0o600)
    secret = runtime.PRIVATE / "session.json"
    runtime.private_write(secret, "connection key")
    runtime.ensure_app_user()
    account = pwd.getpwnam(runtime.APP_USER)
    assert account.pw_uid != 0
    runtime.create_app_tree()
    assert all(p.stat().st_uid == account.pw_uid for p in (runtime.BASE, *runtime.BASE.iterdir()))
    # A rerun after a failed setup: that user's code has replaced a directory with a link.
    (runtime.BASE / "temp").rmdir()
    protected = root / "root-owned"
    protected.mkdir()
    (runtime.BASE / "temp").symlink_to(protected, target_is_directory=True)
    runtime.create_app_tree()
    assert protected.stat().st_uid == 0
    (runtime.BASE / "temp").unlink()
    runtime.create_app_tree()
    assert (runtime.BASE / "temp").stat().st_uid == account.pw_uid
    print("PASS: root hands over only the directories it creates and never follows a planted link")
    node = runtime.BASE / "environment/ComfyUI/custom_nodes/example/node.py"
    subprocess.run(runtime.app_command(["mkdir", "-p", str(node.parent), str(runtime.BASE / "environment/ComfyUI/user")]), check=True)
    subprocess.run(runtime.app_command(["sh", "-c", f"printf 'example custom node' > {node}"]), check=True)

    actual_run = runtime.run

    def local_run(args, **kwargs):
        if "freeze" in args:
            assert kwargs.get("app")
            return "example-package==1.0\n"
        raise AssertionError(args)
    runtime.run = local_run

    identity = as_app("import os; print(os.geteuid(), os.getegid(), open('/proc/self/status').read().count('NoNewPrivs:\\t1'))")
    assert identity.stdout.split() == [str(account.pw_uid), str(account.pw_gid), "1"], identity
    for target in (secret, models / "manual.safetensors", runtime.DRIVE_MOUNT):
        probe = as_app("import os, sys; os.stat(sys.argv[1]); open(sys.argv[1], 'rb') if os.path.isfile(sys.argv[1]) else os.listdir(sys.argv[1])", target)
        assert probe.returncode and "PermissionError" in probe.stderr, (target, probe)
    regained = as_app("import subprocess; print(subprocess.run(['su', '-c', 'id -u'], stdin=subprocess.DEVNULL, capture_output=True, text=True).stdout.strip() or 'denied')")
    assert regained.stdout.strip() == "denied", regained
    print("PASS: application user cannot read Drive or the connection key, and cannot regain root")

    # A custom node plants programs where the application's PATH looks first.
    trophy = runtime.PRIVATE / "ran-as-root"
    planted_bin = runtime.BASE / ".venv/bin"
    subprocess.run(runtime.app_command(["mkdir", "-p", str(planted_bin)]), check=True)
    for name in ("setpriv", Path(sys.executable).name):
        subprocess.run(runtime.app_command(["sh", "-c", f"printf '#!/bin/sh\\ntouch {trophy}\\nexit 1\\n' > {planted_bin / name} && chmod +x {planted_bin / name}"]), check=True)
    for attempt in (lambda: runtime.app_action("get", "models/absent"),
                    lambda: actual_run([sys.executable, "-c", "pass"], app=True, capture=True)):
        try:
            attempt()
        except RuntimeError:
            pass
    assert not trophy.exists(), "a program planted by the application user ran as root"
    actual_run(["rm", "-r", runtime.BASE / ".venv"])
    print("PASS: programs planted on the application's PATH are not run by root")

    # A custom node makes every Python started as its user misbehave (user site-packages).
    site = runtime.BASE / f"home/.local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    hostile = (
        "import os, sys, time\n"
        "mode = open(os.path.expanduser('~/mode')).read() if os.path.exists(os.path.expanduser('~/mode')) else ''\n"
        "if mode == 'flood':\n"
        "    while True: os.write(1, b'x' * 1048576)\n"
        "if mode == 'stall':\n"
        "    time.sleep(600)\n"
        "if mode == 'orphan' and os.fork() == 0:\n"
        "    os.setsid(); time.sleep(600); os._exit(0)\n")
    subprocess.run(runtime.app_command(["mkdir", "-p", str(site)]), check=True)
    subprocess.run(runtime.app_command([sys.executable, "-S", "-c", "import sys; open(sys.argv[1], 'w').write(sys.argv[2])",
                                        str(site / "usercustomize.py"), hostile]), check=True)
    import time

    def attempt(mode, **limits):
        subprocess.run(runtime.app_command(["sh", "-c", f"printf {mode} > {runtime.BASE / 'home/mode'}"]), check=True)
        started = time.monotonic()
        try:
            runtime.app_action("prepare", **limits)
            outcome = "completed"
        except RuntimeError as failure:
            outcome = str(failure)
        return outcome, time.monotonic() - started
    outcome, seconds = attempt("flood", limit=4 * 1024 * 1024, timeout=30)
    assert "上限" in outcome and seconds < 10, (outcome, seconds)
    outcome, seconds = attempt("stall", timeout=3)
    assert "時間内" in outcome and seconds < 10, (outcome, seconds)
    outcome, seconds = attempt("orphan", timeout=30)
    assert outcome == "completed" and seconds < 10, (outcome, seconds)
    print("PASS: a flooding, stalling or pipe-holding application command cannot hold root up")
    survivors = subprocess.run(["pgrep", "-u", runtime.APP_USER], capture_output=True, text=True).stdout.split()
    assert survivors, "expected the detached process to be running"
    runtime.stop_app_processes()
    assert not subprocess.run(["pgrep", "-u", runtime.APP_USER], capture_output=True, text=True).stdout.split()
    print("PASS: stopping ends processes a custom node detached from ComfyUI")
    actual_run(["rm", "-r", runtime.BASE / "home/.local", runtime.BASE / "home/mode"])

    assert runtime.sync_models(config) == 1
    cached = runtime.BASE / "models/manual.safetensors"
    assert cached.read_bytes() == b"manual model" and cached.stat().st_uid == account.pw_uid
    assert runtime.sync_models(config) == 0
    print("PASS: Drive models are copied into the cache as the application user")

    # What the Manager downloads into the cache is saved to Drive, byte for byte.
    payload = os.urandom(3 * 1024 * 1024 + 17)
    subprocess.run(runtime.app_command(["tee", str(runtime.BASE / "models/downloaded.safetensors")]),
                   input=payload, stdout=subprocess.DEVNULL, check=True)
    runtime.save_new_cache_models(config, force=True)
    assert (models / "downloaded.safetensors").read_bytes() == payload
    listing = actual_run(["sh", "-c", "pwd; seq 1 200000"], app=True, capture=True, cwd=runtime.BASE / "models")
    assert listing.splitlines()[0] == str(runtime.BASE / "models") and listing.splitlines()[-1] == "200000"
    print("PASS: new cache models reach Drive intact, and long command output is collected completely")

    # A custom node swaps cache entries for links to files only root can read or write.
    (runtime.BASE / "models/stolen.safetensors").symlink_to(secret)
    cached.unlink()
    cached.symlink_to(secret)
    (models / "manual.safetensors").write_bytes(b"changed on Drive")
    planted = runtime.BASE / "models/planted"
    planted.symlink_to(runtime.PRIVATE, target_is_directory=True)
    (models / "planted").mkdir()
    (models / "planted/overwrite.txt").write_bytes(b"attacker controlled")
    try:
        runtime.sync_models(config)
    except (RuntimeError, ValueError):
        pass
    runtime.save_new_cache_models(config, force=True)
    assert secret.read_text() == "connection key"
    assert not (runtime.PRIVATE / "overwrite.txt").exists()
    assert not (models / "stolen.safetensors").exists()
    assert "connection key" not in "".join(p.read_text(errors="replace") for p in models.rglob("*") if p.is_file())
    for link in (planted, runtime.BASE / "models/stolen.safetensors", cached):
        link.unlink()
    (models / "planted/overwrite.txt").unlink()
    (models / "planted").rmdir()
    assert runtime.sync_models(config) == 1 and cached.read_bytes() == b"changed on Drive"
    print("PASS: links planted in the model cache cannot make root leak or overwrite private files")

    lock = runtime.BASE / "environment/requirements.lock.txt"
    victim = runtime.PRIVATE / "victim.txt"
    runtime.private_write(victim, "root-owned file")
    subprocess.run(runtime.app_command(["ln", "-s", str(victim), str(lock)]), check=True)
    try:
        runtime.checkpoint(config, force=True)
    except RuntimeError:
        pass
    assert victim.read_text() == "root-owned file"
    lock.unlink()
    print("PASS: a planted lock-file link cannot make the checkpoint overwrite a root-owned file")

    runtime.checkpoint(config, force=True)
    assert lock.read_text() == "example-package==1.0\n" and lock.stat().st_uid == account.pw_uid
    storage = runtime.drive_root(config) / "environment"
    original = json.loads((storage / "latest.json").read_text())
    original_copy = runtime.atomic_copy
    subprocess.run(runtime.app_command([sys.executable, "-c", "import sys; open(sys.argv[1], 'w').write('updated custom node')", str(node)]), check=True)

    def failed_pointer_copy(source, destination):
        if Path(destination).name == "latest.json":
            raise OSError("simulated Drive pointer write failure")
        original_copy(source, destination)
    runtime.atomic_copy = failed_pointer_copy
    try:
        runtime.checkpoint(config, force=True)
    except OSError:
        pass
    else:
        raise AssertionError("Simulated checkpoint failure did not occur")
    assert json.loads((storage / "latest.json").read_text()) == original
    archive = storage / original["archive"]
    assert runtime.file_hash(archive) == original["sha256"]
    print("PASS: failed checkpoint preserves the preceding environment")

    # A new runtime: the saved environment comes back owned by the application user.
    runtime.atomic_copy = original_copy
    actual_run(["rm", "-rf", runtime.BASE / "environment"])
    (runtime.BASE / "environment").mkdir()
    os.chown(runtime.BASE / "environment", account.pw_uid, account.pw_gid)
    with archive.open("rb") as stream:
        runtime.app_action("restore", stdin=stream)
    assert node.read_text() == "example custom node" and node.stat().st_uid == account.pw_uid
    assert runtime.checked_lock(runtime.app_action("get", "environment/requirements.lock.txt").decode())
    print("PASS: the environment is restored as the application user")

    for version in range(2):
        subprocess.run(runtime.app_command([sys.executable, "-c", "import sys; open(sys.argv[1], 'w').write(sys.argv[2])",
                                            str(node), "new node version " + str(version)]), check=True)
        runtime.checkpoint(config, force=True)
    assert len(list(storage.glob("*.tar.gz"))) == 3
    latest = json.loads((storage / "latest.json").read_text())
    assert runtime.file_hash(storage / latest["archive"]) == latest["sha256"]
    print("PASS: three environment generations retained and latest pointer verified")

    (runtime.BASE / "output/image.png").write_bytes(b"generated")
    (runtime.BASE / "output/leak.txt").symlink_to(secret)
    zipped = runtime.zip_outputs(root / "output.zip")
    import zipfile
    with zipfile.ZipFile(zipped) as bundle:
        assert bundle.namelist() == ["image.png"] and bundle.read("image.png") == b"generated"
    print("PASS: output ZIP contains generated files only")

    # Updating ComfyUI in place: real git as the application user, package installs simulated.
    origin = root / "origin"
    origin.mkdir()
    os.chown(origin, account.pw_uid, account.pw_gid)
    commit = "git add -A && git -c user.name=check -c user.email=check@example.invalid commit -q -m release"
    releases = " && ".join(
        f"echo 'version {number}' > main.py && echo '{requirement}' > requirements.txt && {commit} && git tag v{number}"
        for number, requirement in ((1, "one==1"), (2, "two==2"), (3, "broken==0")))
    subprocess.run(runtime.app_command(["sh", "-c", "set -e; git init -q -b master . && mkdir -p models/checkpoints"
                                        " && : > models/checkpoints/put_checkpoints_here && " + releases]),
                   cwd=origin, env=runtime.application_environment(), check=True)
    repo = runtime.BASE / "environment/ComfyUI"
    settings = repo / "user/default/comfy.settings.json"
    subprocess.run(runtime.app_command(["sh", "-c", f"set -e; git init -q . && git remote add origin {origin}"
                                        " && git fetch -q --tags origin && git checkout -q --force --detach v1"
                                        f" && mkdir -p user/default && printf '{{\"Comfy.Theme\": \"dark\"}}' > {settings}"]),
                   cwd=repo, env=runtime.application_environment(), check=True)
    runtime.app_action("prepare")
    runtime.app_action("put", "environment/runtime.json",
                       stdin=json.dumps({"torch_index_url": "https://download.pytorch.org/whl/cu130"}).encode())
    installed = ["example-package==1.0\n"]

    def update_run(args, **kwargs):
        args = list(map(str, args))
        if args[0] == "git":
            return actual_run(args, **kwargs)
        assert kwargs.get("app"), args
        if "freeze" in args:
            return installed[0]
        if "install" in args and Path(args[-1]).name == "requirements.txt":
            wanted = Path(args[-1]).read_text()
            if "broken" in wanted:
                raise RuntimeError("no such package")
            installed[0] = "example-package==1.0\n" + wanted
    runtime.run = update_run

    def kept():
        return (node.read_text() == "new node version 1" and json.loads(settings.read_text()) == {"Comfy.Theme": "dark"}
                and (repo / "models").is_symlink() and (repo / "models").resolve() == runtime.BASE / "models"
                and cached.read_bytes() == b"changed on Drive" and not (runtime.PRIVATE / "update-incomplete.json").exists())
    assert kept() and (repo / "main.py").read_text() == "version 1\n"
    before = json.loads((storage / "latest.json").read_text())
    assert runtime.update_comfyui(config, "v2") is True
    assert kept() and (repo / "main.py").read_text() == "version 2\n" and (repo / "main.py").stat().st_uid == account.pw_uid
    updated = json.loads((storage / "latest.json").read_text())
    assert updated != before and lock.read_text() == "example-package==1.0\ntwo==2\n"
    print("PASS: ComfyUI is updated in place; custom nodes, settings and models stay, and the result is saved")

    try:
        runtime.update_comfyui(config, "v3")
    except RuntimeError as failure:
        assert "no such package" in str(failure), failure
    else:
        raise AssertionError("the update to a version that cannot be installed did not fail")
    assert kept() and (repo / "main.py").read_text() == "version 2\n"
    assert json.loads((storage / "latest.json").read_text()) == updated
    print("PASS: an update that fails is undone and leaves Drive as it was")

    # The usual case: a new runtime restores the saved environment, then updates it.
    actual_run(["rm", "-rf", runtime.BASE / "environment"])
    (runtime.BASE / "environment").mkdir()
    os.chown(runtime.BASE / "environment", account.pw_uid, account.pw_gid)
    with (storage / updated["archive"]).open("rb") as stream:
        runtime.app_action("restore", stdin=stream)
    runtime.app_action("prepare")
    assert kept() and (repo / "main.py").read_text() == "version 2\n"
    assert runtime.update_comfyui(config, "v1") is True
    assert kept() and (repo / "main.py").read_text() == "version 1\n"
    assert runtime.update_comfyui(config, "v1") is False
    print("PASS: a restored environment can be moved to another version")

    # Reading Drive's models in place: a read-only view of that one folder for the application user.
    import shutil
    if not shutil.which("bindfs"):
        print("SKIP: bindfs is not installed, so reading Drive's models in place was not checked")
    else:
        runtime.run = actual_run
        runtime.DRIVE_MODELS = root / "drive-models"
        try:
            direct = {**config, "model_access": "direct"}
            (models / "loras").mkdir()
            (models / "loras/style.safetensors").write_bytes(b"style lora on Drive")
            (models / "loras/placeholder").write_bytes(b"")
            personal = runtime.DRIVE_MOUNT / "MyDrive/personal.txt"
            personal.write_text("not a model")
            for path in (models / "loras", models / "loras/style.safetensors", personal):
                path.chmod(0o700 if path.is_dir() else 0o600)
            local_copy = cached.read_bytes()
            assert runtime.sync_models(direct) == 1  # The other Drive models are already here as files of this runtime.
            link = runtime.BASE / "models/loras/style.safetensors"
            assert link.is_symlink() and cached.read_bytes() == local_copy and not cached.is_symlink()
            assert not (runtime.BASE / "models/loras/placeholder").exists()
            seen = as_app("import sys; print(open(sys.argv[1], 'rb').read().decode())", link)
            assert seen.stdout.strip() == "style lora on Drive", seen
            for attempt in (f"open({str(link)!r}, 'ab').write(b'x')", f"import os; os.remove({str(runtime.DRIVE_MODELS / 'loras/style.safetensors')!r})",
                            f"open({str(runtime.DRIVE_MODELS / 'new.safetensors')!r}, 'wb')",
                            f"open({str(models / 'loras/style.safetensors')!r}, 'rb')", f"open({str(personal)!r}, 'rb')",
                            f"import os; os.listdir({str(runtime.DRIVE_MODELS / '..' / 'private')!r})"):
                refused = as_app(attempt)
                assert refused.returncode and ("Error" in refused.stderr), (attempt, refused)
            assert (models / "loras/style.safetensors").read_bytes() == b"style lora on Drive"
            print("PASS: Drive's models are read in place through a read-only view of that folder alone")

            # The links are not models of this runtime: saving does not send them back to Drive.
            before_save = sorted(p.name for p in (models / "loras").iterdir())
            runtime.save_new_cache_models(direct, force=True)
            assert sorted(p.name for p in (models / "loras").iterdir()) == before_save
            # Added to Drive: linked on refresh. Removed from Drive: its link goes.
            (models / "loras/second.safetensors").write_bytes(b"second lora")
            assert runtime.sync_models(direct) == 1 and (runtime.BASE / "models/loras/second.safetensors").is_symlink()
            (models / "loras/style.safetensors").unlink()
            time.sleep(1.5)  # The view remembers a file's existence for a second.
            assert runtime.sync_models(direct) == 0 and not os.path.lexists(link)
            # ComfyUI's own files arriving after an update do not write through a link into the view.
            subprocess.run(runtime.app_command(["sh", "-c", f"rm {repo / 'models'} && mkdir -p {repo / 'models/loras'}"
                                                f" && printf shipped > {repo / 'models/loras/second.safetensors'}"]), check=True)
            runtime.app_action("prepare")
            assert (models / "loras/second.safetensors").read_bytes() == b"second lora" and (repo / "models").is_symlink()
            print("PASS: links follow what is on Drive and are never saved or written through")

        finally:
            # Whatever happened above: the temporary folder is deleted next, and not through a mounted view.
            runtime.hide_drive_models()
        assert not runtime.drive_models_shown() and not list(runtime.DRIVE_MODELS.iterdir())
        print("PASS: the view is taken away before Drive is unmounted")
