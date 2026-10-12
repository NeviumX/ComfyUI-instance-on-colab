"""Self-contained runtime embedded in the ComfyUI instance on Colab notebooks."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import threading
import tempfile
import time
from urllib.parse import urlsplit, urljoin, quote, unquote, parse_qs

BASE = Path("/content/comfy-session")
PRIVATE = Path("/content/.comfy-private")
DRIVE_MOUNT = PRIVATE / "drive"
CONFIG = PRIVATE / "session.json"
COOKIE = "__Host-comfy_colab"
# What a session adds to the saved settings, and stopping takes away again.
SESSION_KEYS = ("mode", "cookie_secret", "remote_url", "origins", "access")
PAIRING_PATH = "/_colab/pair"
PAIRING_SECONDS = 600
PAIRING_ATTEMPTS = 5
COOKIE_SECONDS = 24 * 3600
ACCESS_KEYS_REFRESH = 30
EXCLUDED = {"models", "output", "temp", "input"}
APP_USER = "comfy"
SETPRIV = "/usr/bin/setpriv"
# Bounds on what root accepts from, and how long it waits for, commands run as that user.
APP_TIMEOUT = 3600
APP_OUTPUT_LIMIT = 16 * 1024 ** 2
ARCHIVE_LIMIT = 4 * 1024 ** 3
MODEL_FOLDERS = ("checkpoints", "diffusion_models", "unet", "text_encoders", "clip", "clip_vision", "vae",
                 "loras", "controlnet", "upscale_models", "embeddings", "audio_encoders", "model_patches")
CLOUDFLARED_VERSION = "2026.10.0"
CLOUDFLARED_SHA256 = "d33ff2d14475178d2012c2c56beba87389ac5ded27649519f198a7d3134a99db"
MESSAGES = None  # build_notebook.py puts the messages of the notebook's language on this line.
if MESSAGES is None:
    # Run from the source tree: the Japanese messages of translations.json next to this file.
    MESSAGES = {f"{section}.{name}": entry["JP"] for section, entries in json.loads(
        Path(__file__).with_name("translations.json").read_text(encoding="utf-8"))["runtime"].items()
        for name, entry in entries.items()}


def t(key, **values):
    """The message translations.json holds as runtime.<key>, with its values filled in."""
    return MESSAGES[key].format(**values)


def excluded_path(parts):
    return "__pycache__" in parts or ".venv" in parts or (
        len(parts) >= 2 and parts[0] == "ComfyUI" and parts[1] in EXCLUDED)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def private_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(data, encoding="utf-8")
    path.chmod(0o600)


def safe_relative(value):
    if "\\" in value or ":" in value or "\x00" in value:
        raise ValueError(t("path.use_slashes"))
    p = PurePosixPath(value)
    if not value or p.is_absolute() or any(x in {"..", "."} for x in value.split("/")):
        raise ValueError(t("path.unsafe"))
    return str(p)


GPU_ENVIRONMENT_KEYS = (
    "LD_LIBRARY_PATH", "LD_PRELOAD", "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_PATH",
    "CUDA_MODULE_LOADING", "CUDA_DEVICE_ORDER", "CUDA_CACHE_PATH",
    "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "PYTORCH_NVML_BASED_CUDA_CHECK",
)


def application_environment():
    """Preserve GPU loader/device settings without copying notebook credentials."""
    env = {"PATH": f"{BASE / '.venv/bin'}:/usr/local/bin:/usr/bin:/bin",
           "HOME": str(BASE / "home"), "LANG": "C.UTF-8",
           "PYTHONUNBUFFERED": "1", "VIRTUAL_ENV": str(BASE / ".venv"),
           "PIP_DISABLE_PIP_VERSION_CHECK": "1",
           # Colab's global pip.conf logs to /var/log/pip.log, which only root can write:
           # as the application user every log record would fail with a traceback.
           "PIP_CONFIG_FILE": "/dev/null"}
    for name in GPU_ENVIRONMENT_KEYS:
        if name in os.environ:
            env[name] = os.environ[name]
    if "LD_LIBRARY_PATH" in env:
        # An empty or relative entry means the current directory, which the application user controls.
        env["LD_LIBRARY_PATH"] = ":".join(p for p in env["LD_LIBRARY_PATH"].split(":") if p.startswith("/"))
    return env


def drops_privileges():
    return sys.platform == "linux" and os.geteuid() == 0


def app_command(args):
    """Run args as the unprivileged application user; setuid programs cannot give root back."""
    args = list(map(str, args))
    if not drops_privileges():
        return args
    # Absolute: the application's PATH starts in a directory that user can write, and root
    # is still root while this program is looked up.
    return [SETPRIV, "--reuid", APP_USER, "--regid", APP_USER, "--init-groups", "--no-new-privs", "--", *args]


def run(args, *, app=False, capture=False, cwd=None, timeout=None):
    args = list(map(str, args))
    env = application_environment() if app else os.environ.copy()
    # Describe the stage without printing command arguments or credentials.
    stage = Path(args[0]).name
    if "-m" in args and args.index("-m") + 1 < len(args):
        module_index = args.index("-m") + 1
        stage += " -m " + args[module_index]
        words = {"pip": 1, "uv": 2}.get(args[module_index], 0)
        rest = args[module_index + 1:]
        if rest[:1] == ["--python"]:  # pip aimed at another environment: name the subcommand, not the flag.
            rest = rest[2:]
        stage += "".join(" " + word for word in rest[:words])
    if app:
        args = app_command(args)
    if not capture:
        print(t("command.running") + stage, flush=True)
    started = time.monotonic()
    # Child-process file descriptors can bypass Colab's notebook output capture.
    # Always collect output and explicitly display it through notebook Python.
    if app and drops_privileges():
        from types import SimpleNamespace
        try:
            code, out, err = bounded(args, env=env, cwd=cwd, limit=64 * 1024 ** 2, timeout=timeout or 2 * APP_TIMEOUT)
        except RuntimeError as failure:
            raise RuntimeError(t("command.failed", stage=stage, failure=failure)) from None
        result = SimpleNamespace(returncode=code, stdout=out.decode(errors="replace"), stderr=err.decode(errors="replace"))
    else:
        result = subprocess.run(args, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if result.returncode or not capture:
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", flush=True)
    if result.returncode:
        raise RuntimeError(t("command.exit_code", stage=stage, code=result.returncode))
    if not capture:
        print(t("command.done", stage=stage, seconds=time.monotonic() - started), flush=True)
    return result.stdout if capture else None


def bounded(command, *, env, limit, timeout, stdin=None, sink=None, cwd=None):
    """Run a command the application user controls and collect what it prints.

    It cannot make root wait past timeout or take more than limit bytes, whether it stalls,
    floods its output, or leaves a child of its own holding the pipes open.
    Returns (exit code, output unless sink was given, end of the error output)."""
    import select
    with tempfile.TemporaryFile() as spool:
        if isinstance(stdin, bytes):
            spool.write(stdin)
            spool.seek(0)
            stdin = spool
        process = subprocess.Popen(command, env=env, cwd=cwd, stdin=stdin or subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + timeout
    streams = {process.stdout.fileno(): True, process.stderr.fileno(): False}
    output, errors, total, failure = [], b"", 0, None
    try:
        while streams and not failure:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = t("command.timed_out")
                break
            ready, _, _ = select.select(list(streams), [], [], min(remaining, 0.5))
            for fd in ready:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    del streams[fd]
                elif not streams[fd]:
                    errors = (errors + chunk)[-1024 * 1024:]
                elif total + len(chunk) > limit:
                    failure = t("command.output_limit")
                    break
                else:
                    total += len(chunk)
                    sink.write(chunk) if sink else output.append(chunk)
            if not ready and process.poll() is not None:
                break  # It exited; anything still holding the pipes is not waited for.
        if not failure:
            try:
                process.wait(timeout=max(deadline - time.monotonic(), 0))
            except subprocess.TimeoutExpired:
                failure = t("command.timed_out")
    finally:
        if process.poll() is None:
            process.kill()
        process.stdout.close()
        process.stderr.close()
        process.wait()
    if failure:
        raise RuntimeError(failure)
    return process.returncode, b"".join(output), errors


def stop_app_processes():
    """End everything the application user still runs, including what a custom node detached
    from ComfyUI so that it would outlive a stop."""
    if not drops_privileges():
        return
    import pwd
    try:
        uid = pwd.getpwnam(APP_USER).pw_uid
    except KeyError:
        return
    # As that user, signal every process it may signal: all of its own, in one step.
    subprocess.run(app_command(["/usr/bin/kill", "-KILL", "-1"]), env={}, capture_output=True)
    for _ in range(50):
        alive = False
        for entry in Path("/proc").iterdir():
            try:
                if entry.name.isdigit() and entry.stat().st_uid == uid:
                    alive = True
                    os.kill(int(entry.name), signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError):
                pass
        if not alive:
            return
        time.sleep(0.1)
    raise RuntimeError(t("user.processes_remain"))


def ensure_app_user():
    """Create the account ComfyUI, pip and custom nodes run as. It cannot enter PRIVATE (Drive, keys)."""
    if not Path(SETPRIV).is_file():
        raise RuntimeError(t("user.setpriv_missing", setpriv=SETPRIV))
    if subprocess.run(["id", "-u", APP_USER], capture_output=True).returncode:
        run(["useradd", "--system", "--user-group", "--no-create-home", "--home-dir", BASE / "home",
             "--shell", "/usr/sbin/nologin", APP_USER])
    # GPU device nodes are normally open to everyone; join their groups where they are not.
    for device in [*Path("/dev").glob("nvidia*"), *Path("/dev/dri").glob("*")]:
        info = device.stat()
        if stat.S_ISCHR(info.st_mode) and info.st_mode & 0o060 and not info.st_mode & 0o006 and info.st_gid:
            if subprocess.run(["getent", "group", str(info.st_gid)], capture_output=True).returncode:
                run(["groupadd", "--gid", info.st_gid, f"{APP_USER}-device-{info.st_gid}"])
            run(["usermod", "--append", "--groups", info.st_gid, APP_USER])


def create_app_tree():
    """Make the application user's working directories. Root hands over only what it has just
    made and never follows a link: on a rerun after a failed setup, the tree may already hold
    whatever that user's code left there."""
    import pwd
    account = pwd.getpwnam(APP_USER)
    created = []
    for directory in (BASE, *(BASE / name for name in ("environment", "models", "output", "temp", "input", "home", "database"))):
        try:
            os.mkdir(directory)
        except FileExistsError:
            continue
        created.append(directory)
    for directory in reversed(created):  # BASE last, so nothing can be swapped underneath meanwhile.
        os.chown(directory, account.pw_uid, account.pw_gid, follow_symlinks=False)


def app_tree_action(action, args, source, sink):
    """File operations inside BASE. They run as the application user, so a link planted there
    by a custom node cannot redirect root to Drive, the keys or system files."""
    envdir = BASE / "environment"
    repo = envdir / "ComfyUI"
    if action == "put":
        target = contained_path(BASE, args[0])
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".comfy-", suffix=".part", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as file:
                shutil.copyfileobj(source, file, 4 * 1024 * 1024)
            if len(args) > 1:
                os.utime(temporary, ns=(int(args[1]), int(args[1])))
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    elif action == "get":
        with contained_path(BASE, args[0]).open("rb") as file:
            shutil.copyfileobj(file, sink, 4 * 1024 * 1024)
    elif action == "prepare":
        # Keep the core default and Manager's direct models path on the same storage.
        for name in MODEL_FOLDERS:
            (BASE / "models" / name).mkdir(parents=True, exist_ok=True)
        repo_models = repo / "models"
        if repo_models.exists() and not repo_models.is_symlink():
            shutil.copytree(repo_models, BASE / "models", dirs_exist_ok=True)
            shutil.rmtree(repo_models)
        if not repo_models.exists():
            repo_models.symlink_to(BASE / "models", target_is_directory=True)
        model_cfg = {"base_path": str(BASE / "models"), "is_default": True, **{name: name for name in MODEL_FOLDERS}}
        (BASE / "models.yaml").write_text(json.dumps({"colab_drive_cache": model_cfg}))  # JSON is valid YAML.
        for name in ("user", "custom_nodes"):
            (repo / name).mkdir(exist_ok=True)
        if (repo / "user/comfyui.db").exists():
            shutil.copyfile(repo / "user/comfyui.db", BASE / "database/comfyui.db")
    elif action == "restore":
        fd, temporary = tempfile.mkstemp(prefix=".comfy-", suffix=".tar.gz", dir=BASE)
        try:
            with os.fdopen(fd, "wb") as file:
                shutil.copyfileobj(source, file, 4 * 1024 * 1024)
            restore_archive(temporary, envdir)
        finally:
            Path(temporary).unlink(missing_ok=True)
    elif action == "backup-database":
        # SQLite's online backup includes committed WAL changes consistently.
        import sqlite3
        with sqlite3.connect(f"file:{BASE / 'database/comfyui.db'}?mode=ro", uri=True) as database:
            with sqlite3.connect(repo / "user/comfyui.db") as target:
                database.backup(target)
    elif action == "archive":
        with tarfile.open(fileobj=sink, mode="w|gz", compresslevel=1) as tf:
            for child in sorted(envdir.iterdir()):
                tf.add(child, arcname=child.name, filter=archive_filter)
    elif action == "zip-output":
        import zipfile
        root = BASE / "output"
        count = 0
        with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_STORED) as archive:
            for parent, dirs, files in os.walk(root, followlinks=False):
                dirs[:] = [d for d in dirs if not (Path(parent) / d).is_symlink()]
                for name in files:
                    file = Path(parent) / name
                    if file.is_symlink() or not file.is_file():
                        continue
                    archive.write(file, file.relative_to(root))
                    count += 1
        if not count:
            raise RuntimeError(t("app.no_outputs"))
    else:
        raise ValueError(action)


def app_action(action, *args, stdin=None, stdout=None, limit=APP_OUTPUT_LIMIT, timeout=APP_TIMEOUT):
    """Perform app_tree_action as the application user. stdin is bytes or a file; stdout is a file."""
    args = list(map(str, args))
    if not drops_privileges():
        import io
        sink = stdout or io.BytesIO()
        app_tree_action(action, args, io.BytesIO(stdin) if isinstance(stdin, bytes) else stdin, sink)
        return b"" if stdout else sink.getvalue()
    command = app_command([sys.executable, Path(__file__).resolve(), "app", BASE, action, *args])
    try:
        code, output, errors = bounded(command, env=application_environment(), stdin=stdin, sink=stdout,
                                       limit=limit, timeout=timeout)
    except RuntimeError as failure:
        raise RuntimeError(t("app.failed", action=action, failure=failure)) from None
    if code:
        detail = errors.decode(errors="replace").strip().splitlines()[-1:]
        raise RuntimeError(t("app.failed_detail", action=action) + "".join(detail))
    return output


def receive_file(target, relative, times=None, limit=APP_OUTPUT_LIMIT):
    """Copy a file out of the application's tree to a root-only destination, atomically."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".comfy-", suffix=".part", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            app_action("get", relative, stdout=file, limit=limit)
        if times:
            os.utime(temporary, ns=times)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def installed_apt_packages():
    """Names dpkg has installed, including names they provide: libglib2.0-0 is provided by
    libglib2.0-0t64 on current Ubuntu."""
    try:
        listing = subprocess.run(["dpkg-query", "-W", "-f", "${db:Status-Abbrev}\t${Package}\t${Provides}\n"],
                                 capture_output=True, text=True).stdout
    except OSError:
        return set()
    names = set()
    for line in listing.splitlines():
        status, _, rest = line.partition("\t")
        package, _, provides = rest.partition("\t")
        if status.startswith("ii"):
            names.add(package)
            names.update(item.split("(")[0].strip() for item in provides.split(",") if item.strip())
    return names


APT_INDEX_REFRESHED = []


def apt_install(packages):
    """Install the packages that are missing. Colab ships most of them, and refreshing the
    package index alone takes longer than everything else here."""
    missing = sorted(set(packages) - installed_apt_packages())
    if not missing:
        return
    if not APT_INDEX_REFRESHED:
        run(["apt-get", "update", "-qq"])
        APT_INDEX_REFRESHED.append(True)
    run(["apt-get", "install", "-y", "-qq", *missing])


def install_packages(*args):
    """Install into the application's virtual environment with uv. It takes the same indexes
    and pins as pip but downloads and unpacks the multi-gigabyte PyTorch stack in parallel."""
    py = BASE / ".venv/bin/python"
    run([py, "-m", "uv", "pip", "install", "--python", py, *args], app=True)


def create_venv():
    """Create the application's virtual environment with pip and uv in it.

    The notebook's own pip installs them, so nothing is needed from apt: installing
    pythonX.Y-venv there upgrades the interpreter the running kernel was started from."""
    venv = BASE / ".venv"
    py = venv / "bin/python"
    run([sys.executable, "-m", "venv", "--without-pip", venv], app=True, capture=True)
    try:
        run([sys.executable, "-m", "pip", "--python", py, "install", "pip", "uv"], app=True)
    except RuntimeError:
        print(t("venv.pip_fallback"), flush=True)
        ensure_venv_support()
        run([sys.executable, "-m", "venv", venv], app=True, capture=True)
        run([py, "-m", "pip", "install", "uv"], app=True)
    return py


def ensure_venv_support():
    """Install ensurepip for the notebook's Python version when it is absent."""
    import importlib.util
    if importlib.util.find_spec("ensurepip") is not None:
        return
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    package = f"python{version}-venv"
    print(t("venv.ensurepip_missing", version=version, package=package), flush=True)
    try:
        apt_install([package])
    except RuntimeError:
        raise RuntimeError(t("venv.package_failed", package=package)) from None
    importlib.invalidate_caches()
    if importlib.util.find_spec("ensurepip") is None:
        raise RuntimeError(t("venv.ensurepip_unavailable", package=package, python=sys.executable))


def mount_drive():
    """Mount Drive under a root-only parent, out of reach of the application user."""
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError(t("drive.needs_root"))
    if PRIVATE.is_symlink() or DRIVE_MOUNT.is_symlink():
        raise RuntimeError(t("drive.private_symlink"))
    PRIVATE.mkdir(mode=0o700, exist_ok=True)
    PRIVATE.chmod(0o700)
    if PRIVATE.stat().st_uid != 0:
        raise RuntimeError(t("drive.private_owner"))
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        if len(fields) > 2 and fields[2] == "fuse.drive" and fields[1] != str(DRIVE_MOUNT):
            raise RuntimeError(t("drive.mounted_elsewhere"))
    from google.colab import drive
    drive.mount(str(DRIVE_MOUNT))
    PRIVATE.chmod(0o700)
    if not (DRIVE_MOUNT / "MyDrive").is_dir():
        raise RuntimeError(t("drive.no_mydrive"))


def contained_path(root, relative):
    root = Path(root)
    rel = safe_relative(relative)
    target = root / rel
    if root.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(t("path.outside_folder"))
    current = root
    for part in PurePosixPath(rel).parts:
        current /= part
        if current.is_symlink():
            raise ValueError(t("path.symlink"))
    return target


def drive_root(config):
    mydrive = DRIVE_MOUNT / "MyDrive"
    if not mydrive.is_dir():
        raise RuntimeError(t("drive.not_mounted"))
    return contained_path(mydrive, config["drive_folder"])


def atomic_copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".comfy-", suffix=".part", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def regular_files(root):
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(parent) / d).is_symlink()]
        for name in files:
            file = Path(parent) / name
            if file.is_symlink() or not file.is_file() or name.endswith((".part", ".tmp", ".download")):
                continue
            if name.startswith(".comfy-") or not file.stat().st_size:
                continue  # Empty files are ComfyUI's "put_models_here" markers, not models.
            yield file


def sync_models(config):
    """Drive is authoritative; refresh the private runtime's working copy."""
    source = drive_root(config) / "models"
    source.mkdir(exist_ok=True)
    copied = 0
    origin_file = PRIVATE / "model-origins.json"
    origins = set(json.loads(origin_file.read_text())) if origin_file.exists() else set()
    for file in regular_files(source):
        relative = safe_relative(file.relative_to(source).as_posix())
        origins.add(relative)
        target = BASE / "models" / relative
        st = file.stat()
        if target.exists():
            old = target.stat()
            if old.st_size == st.st_size and abs(old.st_mtime - st.st_mtime) < 1:
                continue
        with file.open("rb") as stream:
            app_action("put", "models/" + relative, st.st_mtime_ns, stdin=stream)
        copied += 1
    private_write(origin_file, json.dumps(sorted(origins)))
    print(t("models.synced", copied=copied))
    return copied


def save_new_cache_models(config, force=False):
    # Preserve Manager downloads without overwriting manually updated Drive files.
    destination = drive_root(config) / "models"
    destination.mkdir(exist_ok=True)
    origin_file = PRIVATE / "model-origins.json"
    origins = set(json.loads(origin_file.read_text())) if origin_file.exists() else set()
    for file in regular_files(BASE / "models"):
        relative = file.relative_to(BASE / "models").as_posix()
        if relative in origins:
            continue
        st = file.stat()
        if not force and time.time() - st.st_mtime < 30:
            continue
        target = contained_path(destination, relative)
        if not target.exists():
            # No more than the size root saw: the file is the application user's to change.
            receive_file(target, "models/" + relative, (st.st_atime_ns, st.st_mtime_ns), limit=st.st_size)
        origins.add(relative)
    private_write(origin_file, json.dumps(sorted(origins)))


def archive_filter(member):
    parts = PurePosixPath(member.name).parts
    if excluded_path(parts) or any(p.endswith(".pyc") for p in parts):
        return None
    # Third-party symlinks could capture credentials or files outside the workspace.
    if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
        return None
    return member


def restore_archive(archive, dest):
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            safe_relative(member.name)
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError(t("environment.archive_special_files"))
        if not hasattr(tarfile, "data_filter"):
            raise RuntimeError(t("environment.archive_needs_data_filter"))
        tf.extractall(dest, filter="data")


def fingerprint(root):
    entries = []
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not excluded_path((Path(parent) / d).relative_to(root).parts)
                         and not (Path(parent) / d).is_symlink())
        for name in sorted(files):
            p = Path(parent) / name
            if p.is_symlink() or name.endswith(".pyc"):
                continue
            try:
                st = p.stat()
                entries.append((str(p.relative_to(root)), st.st_size, st.st_mtime_ns))
            except FileNotFoundError:
                continue
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def checkpoint(config, force=False):
    import fcntl
    with (PRIVATE / "checkpoint.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        refuse_unfinished_update()
        envdir = BASE / "environment"
        save_new_cache_models(config, force=force)
        storage = drive_root(config) / "environment"
        storage.mkdir(exist_ok=True)
        database = BASE / "database/comfyui.db"
        if database.exists():
            signature = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in
                         (database, database.with_name(database.name + "-wal")) if p.exists()]
            signature_text = json.dumps(signature)
            db_marker = PRIVATE / "database-signature"
            if force or not db_marker.exists() or db_marker.read_text() != signature_text:
                app_action("backup-database")
                db_marker.write_text(signature_text)
        write_lock()
        marker = PRIVATE / "last-fingerprint"
        before = fingerprint(envdir)
        if not force and marker.exists() and marker.read_text() == before:
            return False
        staging = PRIVATE / "environment.tar.gz"
        with staging.open("wb") as file:
            app_action("archive", stdout=file, limit=ARCHIVE_LIMIT, timeout=1800)
        after = fingerprint(envdir)
        if before != after:
            staging.unlink(missing_ok=True)
            raise RuntimeError(t("environment.busy"))
        # Generation files + last-written pointer keep the preceding good snapshot.
        digest = file_hash(staging)
        name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + digest[:12] + ".tar.gz"
        atomic_copy(staging, contained_path(storage, name))
        pointer = {"archive": name, "sha256": digest, "python": config["python"],
                   "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        private_write(PRIVATE / "latest.json", json.dumps(pointer))
        atomic_copy(PRIVATE / "latest.json", contained_path(storage, "latest.json"))
        marker.write_text(after)
        private_write(PRIVATE / "checkpoint-status.json", json.dumps({"ok": True, **pointer}))
        staging.unlink(missing_ok=True)
        # Retain three completed generations; delete only our strictly named archives.
        archives = [item for item in storage.iterdir()
                    if not item.is_symlink() and re.fullmatch(r"\d{8}T\d{6}Z-[a-f0-9]{12}\.tar\.gz", item.name)]
        archives.sort(key=lambda item: (item.stat().st_mtime_ns, item.name), reverse=True)
        for old in archives[3:]:
            if old.name != name:
                contained_path(storage, old.name).unlink()
        return True


def write_lock():
    """Record the installed packages, to install the same ones on the next runtime."""
    py = BASE / ".venv/bin/python"
    # No editable or local paths: those cannot survive a new Colab VM.
    packages = checked_lock(run([py, "-m", "pip", "freeze"], app=True, capture=True, timeout=300))
    lock = "environment/requirements.lock.txt"
    if not (BASE / lock).exists() or app_action("get", lock).decode() != packages:
        app_action("put", lock, stdin=packages.encode())


LOCK_LINE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(==[A-Za-z0-9._+!-]+| @ (git\+)?https://\S+)")


def checked_lock(text):
    """Accept only pinned packages: a lock is installed again on every new runtime, so pip
    options, editable installs and local paths must not ride along in it."""
    if any(line.strip() and not LOCK_LINE.fullmatch(line) for line in text.splitlines()):
        raise RuntimeError(t("environment.unpinned_lock"))
    return text


def checked_torch_index(url):
    if not isinstance(url, str) or not re.fullmatch(r"https://download\.pytorch\.org/whl/cu\d+", url):
        raise RuntimeError(t("setup.invalid_torch_mirror"))
    return url


def unlisted_apt_packages(saved, requested):
    """Package names recorded with a saved environment that this setup was not asked for."""
    names = saved.get("apt_packages", []) if isinstance(saved, dict) else []
    return sorted({name for name in names if isinstance(name, str)
                   and re.fullmatch(r"[a-z0-9][a-z0-9+.-]*", name)} - set(requested))


def install_cloudflared():
    """Download the pinned release and refuse to run anything else."""
    import requests
    binary = PRIVATE / "cloudflared"
    partial = PRIVATE / "cloudflared.part"
    url = ("https://github.com/cloudflare/cloudflared/releases/download/"
           + CLOUDFLARED_VERSION + "/cloudflared-linux-amd64")
    try:
        with requests.get(url, stream=True, timeout=60) as res:
            if not res.ok:
                raise RuntimeError(t("cloudflared.download_failed"))
            with partial.open("wb") as f:
                for chunk in res.iter_content(1024 * 1024):
                    f.write(chunk)
        if file_hash(partial) != CLOUDFLARED_SHA256:
            raise RuntimeError(t("cloudflared.sha256_mismatch"))
        partial.chmod(0o700)
        os.replace(partial, binary)
    finally:
        partial.unlink(missing_ok=True)


def setup(*, root_name="ComfyUI-Colab", comfy_ref="v0.39.0",
          checkpoint_seconds=120, apt_packages=(), restore=True):
    if sys.platform != "linux" or not Path("/content").exists() or os.geteuid() != 0:
        raise RuntimeError(t("setup.not_colab"))
    if CONFIG.exists():
        raise RuntimeError(t("setup.already_done"))
    safe_relative(root_name)
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", comfy_ref) or comfy_ref.startswith("-"):
        raise ValueError(t("setup.invalid_ref"))
    if checkpoint_seconds < 60:
        raise ValueError(t("setup.interval_too_short"))
    if any(not re.fullmatch(r"[a-z0-9][a-z0-9+.-]*", p) for p in apt_packages):
        raise ValueError(t("setup.invalid_apt_package"))
    if subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL).returncode:
        raise RuntimeError(t("setup.no_gpu"))
    if PRIVATE.is_symlink():
        raise RuntimeError(t("drive.private_symlink"))
    PRIVATE.mkdir(mode=0o700, exist_ok=True)
    PRIVATE.chmod(0o700)
    apt_install(["git", "ffmpeg", "libgl1", "libglib2.0-0", *apt_packages])
    mount_drive()
    config = {"drive_folder": root_name,
              "python": f"{sys.version_info.major}.{sys.version_info.minor}",
              "checkpoint_seconds": checkpoint_seconds}
    storage = drive_root(config)
    for directory in (storage, storage / "environment", *(storage / "models" / name for name in MODEL_FOLDERS)):
        directory.mkdir(parents=True, exist_ok=True)
    # Test the selected folder, not any other location in Drive.
    probe = storage / "environment/write-probe.txt"
    probe.write_text("ComfyUI-Colab write test")
    # The application user owns its working tree from here on. Root reads and writes inside
    # it only through app_action, because code running there can plant links.
    envdir = BASE / "environment"
    ensure_app_user()
    create_app_tree()
    pointer_path = storage / "environment/latest.json"
    if pointer_path.exists() and not restore:
        print(t("setup.fresh_environment"))
    elif pointer_path.exists():
        pointer = json.loads(pointer_path.read_text())
        archive_name = safe_relative(pointer["archive"])
        if "/" in archive_name or not archive_name.endswith(".tar.gz"):
            raise RuntimeError(t("setup.invalid_archive_name"))
        if pointer["python"] != config["python"]:
            raise RuntimeError(t("setup.python_changed"))
        archive = PRIVATE / "restore.tar.gz"
        atomic_copy(contained_path(storage / "environment", archive_name), archive)
        if file_hash(archive) != pointer["sha256"]:
            raise RuntimeError(t("setup.archive_sha256_mismatch"))
        with archive.open("rb") as stream:
            app_action("restore", stdin=stream, timeout=2 * APP_TIMEOUT)
        archive.unlink()
        print(t("setup.restored"), pointer["saved_at"])
        nodes = sorted(item.name for item in (envdir / "ComfyUI/custom_nodes").glob("*")
                       if item.is_dir() and item.name != "__pycache__")
        # Restored nodes run again on this runtime; show them so an unwanted one is noticed.
        print(t("setup.restored_nodes"), ", ".join(nodes) if nodes else t("common.none"))
    repo = envdir / "ComfyUI"
    if not repo.exists():
        run(["git", "clone", "https://github.com/Comfy-Org/ComfyUI.git", repo], app=True)
        run(["git", "checkout", comfy_ref], app=True, cwd=repo)
    if not (repo / "main.py").exists():
        raise RuntimeError(t("setup.source_incomplete"))
    app_action("prepare")
    previous_meta = (json.loads(app_action("get", "environment/runtime.json"))
                     if (envdir / "runtime.json").exists() else {})
    # The saved list is a file the application user could rewrite, and apt runs as root:
    # name what was used before and leave the decision to install it to the notebook's setting.
    wanted = unlisted_apt_packages(previous_meta, apt_packages)
    if wanted:
        print(t("setup.saved_apt_packages"), " ".join(wanted))
        print(t("setup.saved_apt_packages_hint"))
    # Match the CUDA build already supported by the current Colab runtime.
    # The mirror is persisted because pip freeze can contain +cuXXX versions.
    lock = envdir / "requirements.lock.txt"
    locked = lock.exists()
    if locked:
        checked_lock(app_action("get", "environment/requirements.lock.txt").decode())
        torch_index = previous_meta.get("torch_index_url", "")
        torch_packages = []
    else:
        import importlib.metadata
        import torch
        cuda = torch.version.cuda
        if not cuda or not re.fullmatch(r"\d+\.\d+", cuda):
            raise RuntimeError(t("setup.no_torch"))
        torch_index = "https://download.pytorch.org/whl/cu" + cuda.replace(".", "")
        torch_packages = ["torch==" + importlib.metadata.version("torch"),
                          "torchvision==" + importlib.metadata.version("torchvision")]
    checked_torch_index(torch_index)
    app_action("put", "environment/runtime.json", stdin=json.dumps(
        {"apt_packages": sorted(apt_packages), "python": config["python"], "torch_index_url": torch_index}).encode())
    py = create_venv()
    if locked:
        # pip looks for each package on every index; uv does only when told to.
        install_packages("--index-strategy", "unsafe-best-match", "--extra-index-url", torch_index, "-r", lock)
    else:
        install_packages("--index-url", torch_index, *torch_packages)
        install_packages("-r", repo / "requirements.txt", "-r", repo / "manager_requirements.txt")
        # Explicit requirements are useful for custom nodes added outside Manager.
        for req in sorted((repo / "custom_nodes").glob("*/requirements.txt")):
            install_packages("-r", req)
    run([py, "-m", "pip", "check"], app=True)
    run([py, "-c", "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); torch.cuda.init(); print('GPU:', torch.cuda.get_device_name(0))"], app=True)
    print(t("setup.copying_models"))
    sync_models(config)
    install_cloudflared()
    private_write(CONFIG, json.dumps(config))
    checkpoint(config, force=True)
    print(t("setup.done") + root_name + "/models")
    return config


def refuse_unfinished_update():
    """A ComfyUI source and packages of two different versions must not be saved or started."""
    if (PRIVATE / "update-incomplete.json").exists():
        raise RuntimeError(t("update.incomplete"))


def update_comfyui(config, ref):
    """Move ComfyUI to another version and install what that version requires.

    Custom nodes, settings, workflows and the database are not files of the ComfyUI
    repository, so they stay as they are. Drive receives the environment only once the new
    version is installed and consistent; an update that fails is undone."""
    if (PRIVATE / "processes.json").exists():
        raise RuntimeError(t("update.running"))
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", ref) or ref.startswith("-"):
        raise ValueError(t("setup.invalid_ref"))
    drive_root(config)  # The result is saved at the end: find out now if Drive is gone.
    repo = BASE / "environment/ComfyUI"
    py = BASE / ".venv/bin/python"
    lock = "environment/requirements.lock.txt"
    marker = PRIVATE / "update-incomplete.json"

    def git(*args):
        return run(["git", *args], app=True, capture=True, cwd=repo, timeout=600).strip()

    def commit(name):
        # The repository belongs to the application user: take nothing but a commit id from it.
        try:
            found = git("rev-parse", "--verify", "--quiet", name + "^{commit}")
        except RuntimeError:
            return ""
        return found if re.fullmatch(r"[0-9a-f]{40,64}", found) else ""

    def switch(target):
        git("checkout", "--force", "--detach", target)
        app_action("prepare")  # The checkout puts back the models folder ComfyUI ships.

    def package_names(listing):
        return {re.split(r"==| @ ", line)[0] for line in listing.splitlines() if LOCK_LINE.fullmatch(line)}

    def undo(state):
        """Back to the version and the packages recorded before the update began."""
        packages = checked_lock(app_action("get", lock).decode())
        index = checked_torch_index(json.loads(app_action("get", "environment/runtime.json")).get("torch_index_url"))
        switch(state["commit"])
        install_packages("--index-strategy", "unsafe-best-match", "--extra-index-url", index, "-r", BASE / lock)
        added = sorted(package_names(run([py, "-m", "pip", "freeze"], app=True, capture=True, timeout=300))
                       - package_names(packages))
        if added:
            run([py, "-m", "uv", "pip", "uninstall", "--python", py, *added], app=True)
        marker.unlink()

    if marker.exists():
        # An earlier update failed and could not be undone then.
        print(t("update.resuming"))
        undo(json.loads(marker.read_text()))
    git("fetch", "--tags", "--force", "origin")
    state = {"commit": commit("HEAD"), "version": git("describe", "--tags", "--always")[:80]}
    target = next(filter(None, map(commit, ("refs/tags/" + ref, "refs/remotes/origin/" + ref, ref))), "")
    if not state["commit"] or not target:
        raise ValueError(t("update.unknown_ref", ref=ref))
    if target == state["commit"]:
        print(t("update.already", version=state["version"]))
        return False
    print(t("update.starting", previous=state["version"], ref=ref))
    write_lock()
    private_write(marker, json.dumps(state))
    try:
        switch(target)
        install_packages(*(argument for name in ("requirements.txt", "manager_requirements.txt")
                           if (repo / name).exists() for argument in ("-r", repo / name)))
        run([py, "-m", "pip", "check"], app=True)
    except Exception:
        try:
            undo(state)
        except Exception:
            print(t("update.undo_failed"))
            raise
        print(t("update.undone", version=state["version"]))
        raise
    marker.unlink()
    version = git("describe", "--tags", "--always")[:80]
    checkpoint(config, force=True)
    print(t("update.done", previous=state["version"], version=version))
    return True


def model_download_url(source, *, hf_repo="", hf_file="", hf_revision="main", civitai_version=""):
    if source == "Hugging Face":
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", hf_repo):
            raise ValueError(t("download.hf_repository"))
        file = safe_relative(hf_file)
        if not hf_revision or "\x00" in hf_revision:
            raise ValueError(t("download.hf_revision"))
        return f"https://huggingface.co/{hf_repo}/resolve/{quote(hf_revision, safe='')}/{quote(file, safe='/')}"
    if source == "Civitai":
        if not re.fullmatch(r"[1-9][0-9]*", str(civitai_version)):
            raise ValueError(t("download.civitai_version"))
        return "https://civitai.com/api/download/models/" + str(civitai_version)
    raise ValueError(t("download.unknown_source"))


def parse_model_url(url):
    """Download URL and file name for a Hugging Face or Civitai link copied from a browser."""
    parsed = urlsplit(url.strip())
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    host = (parsed.hostname or "").removeprefix("www.")
    if parsed.scheme == "https" and host == "huggingface.co":
        if len(parts) < 5 or parts[2] not in {"resolve", "blob"}:
            raise ValueError(t("download.hf_url"))
        return model_download_url("Hugging Face", hf_repo="/".join(parts[:2]), hf_file="/".join(parts[4:]),
                                  hf_revision=parts[3]), parts[-1]
    if parsed.scheme == "https" and host == "civitai.com":
        version = parse_qs(parsed.query).get("modelVersionId", [""])[0]
        if parts[:3] == ["api", "download", "models"] and len(parts) == 4:
            version = parts[3]
        elif parts[:1] != ["models"] or not version:
            raise ValueError(t("download.civitai_url"))
        return model_download_url("Civitai", civitai_version=version), ""
    raise ValueError(t("download.unsupported_url"))


def model_token_name(url):
    """The Colab secret holding the access token for the site a model link points at."""
    return "CIVITAI_TOKEN" if "civitai.com" in (urlsplit(url.strip()).hostname or "") else "HF_TOKEN"


def served_filename(headers):
    """The file name a download response declares, or "" unless it is a plain name."""
    from email.message import EmailMessage
    message = EmailMessage()
    try:
        message["Content-Disposition"] = headers.get("Content-Disposition") or "attachment"
        name = message.get_filename() or ""
    except (ValueError, TypeError, LookupError):
        return ""
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    return name if re.fullmatch(r"[^\x00-\x1f:]{1,200}", name) and not name.startswith(".") else ""


def fetch_model(config, url, folder, filename="", sha256="", token_secret=""):
    """Download the model a pasted link points at into a folder of Drive's models directory."""
    download, suggested = parse_model_url(url)
    folder = safe_relative(folder.strip().strip("/"))
    filename = filename.strip() or suggested
    if "/" in filename:
        raise ValueError(t("download.filename_has_folder"))
    if filename:
        return download_model(config, download, folder + "/" + filename, sha256.strip(), token_secret)
    return download_model(config, download, folder, sha256.strip(), token_secret, name_from_server=True)


def checked_https_url(url):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.port not in {None, 443}:
        raise ValueError(t("download.not_https"))
    import ipaddress
    import socket
    if parsed.hostname in {"localhost", "metadata.google.internal"}:
        raise ValueError(t("download.local_redirect"))
    try:
        addresses = {ipaddress.ip_address(parsed.hostname)}
    except ValueError:
        # A public-looking name can still point at the VM's internal services.
        try:
            found = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
        except OSError:
            raise ValueError(t("download.unresolved_host")) from None
        addresses = {ipaddress.ip_address(item[4][0].split("%")[0]) for item in found}
    if any(not ip.is_global for ip in addresses):
        raise ValueError(t("download.private_redirect"))
    return parsed


def stream_model(session, url, target, token_secret=""):
    """Follow HTTPS redirects, forwarding the token only to the original host."""
    initial_host = checked_https_url(url).hostname
    current = url
    for _ in range(11):
        parsed = checked_https_url(current)
        headers = {"User-Agent": "ComfyUI-Colab-Downloader/1"}
        if token_secret and parsed.hostname == initial_host:
            headers["Authorization"] = "Bearer " + token_secret
        with session.get(current, headers=headers, stream=True, timeout=(30, 300), allow_redirects=False) as response:
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                if not location:
                    raise RuntimeError(t("download.redirect_without_target"))
                current = urljoin(current, location)
                continue
            if response.status_code != 200:
                raise RuntimeError(t("download.http_error", status=response.status_code))
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type or "application/json" in content_type:
                raise RuntimeError(t("download.not_a_model"))
            digest = hashlib.sha256()
            with Path(target).open("wb") as file:
                for chunk in response.iter_content(4 * 1024 * 1024):
                    file.write(chunk)
                    digest.update(chunk)
            if not Path(target).stat().st_size:
                raise RuntimeError(t("download.empty"))
            return digest.hexdigest(), served_filename(response.headers)
    raise RuntimeError(t("download.too_many_redirects"))


def download_model(config, url, relative_path, sha256="", token_secret="", name_from_server=False):
    """Save a model under relative_path in Drive's models folder. With name_from_server,
    relative_path is the folder and the file takes the name the site serves it under."""
    import requests
    rel = safe_relative(relative_path)
    parsed = checked_https_url(url)
    if parsed.hostname not in {"huggingface.co", "civitai.com"}:
        raise ValueError(t("download.host_not_allowed"))
    if parsed.query or parsed.fragment:
        raise ValueError(t("download.url_has_query"))
    if sha256 and not re.fullmatch(r"[a-fA-F0-9]{64}", sha256):
        raise ValueError(t("download.invalid_sha256"))
    models = drive_root(config) / "models"
    models.mkdir(exist_ok=True)
    target = contained_path(models, rel)
    if not name_from_server and target.exists():
        raise FileExistsError(t("download.exists"))
    folder = target if name_from_server else target.parent
    folder.mkdir(parents=True, exist_ok=True)
    partial = folder / (".comfy-" + secrets.token_hex(8) + ".part")
    try:
        with requests.Session() as session:
            session.trust_env = False  # No implicit .netrc credentials or proxy settings.
            digest, served = stream_model(session, url, partial, token_secret)
        if sha256 and not hmac.compare_digest(digest, sha256.lower()):
            raise RuntimeError(t("download.sha256_mismatch"))
        if name_from_server:
            if not served:
                raise RuntimeError(t("download.no_filename"))
            rel = rel + "/" + served
            target = contained_path(models, rel)
        if target.exists():
            raise FileExistsError(t("download.exists_not_overwritten"))
        os.replace(partial, target)
        sync_models(config)
        print(t("download.saved"), rel, "SHA256:", digest)
    except requests.RequestException:
        raise RuntimeError(t("download.connection_failed")) from None
    finally:
        partial.unlink(missing_ok=True)


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
    private_write(PRIVATE / "pairing.json", json.dumps(
        {"code": code, "expires": time.time() + PAIRING_SECONDS, "attempts": PAIRING_ATTEMPTS}))
    return code


def redeem_pairing_code(supplied):
    """(accepted, tries left). A code is good for one browser, a few minutes and a few wrong
    guesses; after any of these there is no code until the notebook issues another."""
    path = PRIVATE / "pairing.json"
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
    private_write(path, json.dumps(state))
    return False, state["attempts"]


def create_proxy(config, upstream="http://127.0.0.1:8188", secure_cookie=True):
    """Streaming proxy in front of ComfyUI.

    Behind the tunnel it lets in only browsers that hold the session cookie, which a pairing
    code buys once; the cookie also rides Desktop's download session. Behind Colab's port
    forwarding Google has already let in the runtime's owner only, so it just keeps other
    sites from sending requests through that browser. Behind Cloudflare Access it lets in
    only requests that carry Access's signature for this application."""
    from aiohttp import web, ClientError, ClientSession, ClientTimeout, WSMsgType, DummyCookieJar
    import html
    forwarded = config.get("mode") == "colab"
    access = config.get("access") if config.get("mode") == "access" else None
    if access:
        import jwt
    signing_keys = {}
    keys_fetched = [float("-inf")]
    verified = {}  # token -> expiry: one page load sends the same token a few hundred times.
    cookie_secret = config.get("cookie_secret", "")
    cookie_name = COOKIE if secure_cookie else "comfy_colab"
    origins = set(config.get("origins", ()))
    hop = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
           "te", "trailer", "transfer-encoding", "upgrade", "content-length"}
    client_key = web.AppKey("client", ClientSession)
    # Vite names frontend files after their content, so they can be cached for good.
    hashed_asset = re.compile(r"/assets/[\w.-]+-[A-Za-z0-9_-]{8}\.(?:js|css|json|svg|ttf|woff2?)")
    compressible = ("text/", "application/javascript", "application/json", "image/svg+xml")

    def matches(supplied, secret):
        # Bytes: comparing str raises on the non-ASCII input anyone can send.
        return bool(secret) and hmac.compare_digest(supplied.encode(), secret.encode())

    async def lifecycle(app):
        app[client_key] = ClientSession(timeout=ClientTimeout(total=None, sock_connect=30),
                                      auto_decompress=False, cookie_jar=DummyCookieJar())
        yield
        await app[client_key].close()

    async def signing_key(kid):
        if kid not in signing_keys and time.monotonic() - keys_fetched[0] >= ACCESS_KEYS_REFRESH:
            # Cloudflare rotates its keys, so an unknown one is looked up, but not for every bad request.
            keys_fetched[0] = time.monotonic()
            async with ClientSession(timeout=ClientTimeout(total=15)) as session:
                async with session.get(access_certs_url(access["team"])) as res:
                    published = (await res.json(content_type=None))["keys"]
            signing_keys.update({key["kid"]: jwt.PyJWK.from_dict(key).key for key in published})
        return signing_keys[kid]

    async def signed_by_access(request):
        """Cloudflare adds this token to what it lets through. Checking it here means a request
        that reaches the tunnel some other way, or for another application, is still refused."""
        token = request.headers.get("Cf-Access-Jwt-Assertion", "")
        if not token or len(token) > 8192:
            return False
        if verified.get(token, 0) > time.time():
            return True
        try:
            key = await signing_key(jwt.get_unverified_header(token)["kid"])
            claims = jwt.decode(token, key, algorithms=["RS256"], audience=access["aud"],
                                issuer="https://" + access["team"], options={"require": ["exp", "iat", "aud", "iss"]})
        except (jwt.PyJWTError, ClientError, KeyError, TypeError, ValueError, OSError, asyncio.TimeoutError):
            return False
        if len(verified) >= 256:
            verified.clear()
        verified[token] = claims["exp"]
        return True

    def refuse_other_sites(request):
        origin = request.headers.get("Origin")
        if origin and origin not in {"https://" + request.host, "http://" + request.host} | origins:
            print(t("proxy.origin_rejected", origin=origin[:200], host=request.host[:200]), flush=True)
            raise web.HTTPForbidden(text=t("proxy.cross_origin"))
        return origin

    def pairing_page(notice="", status=200):
        page = PAIRING_PAGE.format(path=PAIRING_PATH, notice=html.escape(notice), title=html.escape(t("pairing.title")),
                                   prompt=html.escape(t("pairing.prompt")), button=html.escape(t("pairing.button")))
        # Not no-referrer: a browser then submits the form with "Origin: null", which is refused below.
        return web.Response(text=page, status=status, content_type="text/html", headers={
            "Cache-Control": "no-store", "Referrer-Policy": "same-origin", "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; "
                                       "frame-ancestors 'none'; base-uri 'none'"})

    async def pair(request):
        """The one page anyone can reach: it trades the code shown in the notebook for the cookie."""
        if request.method != "POST":
            return pairing_page()
        refuse_other_sites(request)
        length = request.content_length
        if length is None or length > 256:  # Nothing unauthenticated gets to send more.
            raise web.HTTPBadRequest()
        try:
            form = parse_qs((await request.content.readexactly(length)).decode(errors="replace"))
        except asyncio.IncompleteReadError:
            raise web.HTTPBadRequest() from None
        accepted, remaining = redeem_pairing_code(form.get("code", [""])[0])
        if not accepted:
            return pairing_page(t("pairing.wrong", remaining=remaining) if remaining else t("pairing.closed"), 401)
        response = web.HTTPSeeOther(location="/")
        response.set_cookie(cookie_name, cookie_secret, max_age=COOKIE_SECONDS,
                            secure=secure_cookie, httponly=True, samesite="Lax", path="/")
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        raise response

    async def handler(request):
        if access:
            if not await signed_by_access(request):
                raise web.HTTPUnauthorized(text=t("proxy.access_required"))
        elif not forwarded and not matches(request.cookies.get(cookie_name, ""), cookie_secret):
            if request.path == PAIRING_PATH:
                return await pair(request)
            if request.method == "GET" and "text/html" in request.headers.get("Accept", ""):
                # A browser opening the address: take it to where the code goes.
                raise web.HTTPSeeOther(location=PAIRING_PATH, headers={"Cache-Control": "no-store"})
            raise web.HTTPUnauthorized(text=t("proxy.unauthorized"))
        if request.path == PAIRING_PATH:
            raise web.HTTPSeeOther(location="/")
        origin = refuse_other_sites(request)
        # What proves the visitor's identity stops here: custom nodes get neither the cookie nor Access's token.
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in hop | {"cookie", "authorization"} and not k.lower().startswith("cf-access-")}
        if origin:
            # ComfyUI refuses a request whose Host and Origin differ, and a forwarder may rewrite Host.
            headers = {k: v for k, v in headers.items() if k.lower() != "host"}
            headers["Host"] = urlsplit(origin).netloc
        # ComfyUI answers 403 to cross-site requests, which includes following the
        # connection URL from a link (e.g. Colab output). Loading the root page is safe.
        if request.method == "GET" and request.path == "/":
            headers = {k: v for k, v in headers.items() if k.lower() != "sec-fetch-site"}
        # Preserve other backend cookies, but never disclose the proxy credential to custom nodes.
        # Behind Colab's forwarding its own sign-in cookies cannot be told apart, so none are passed on.
        backend_cookies = {} if forwarded else {
            k: v for k, v in request.cookies.items() if k not in {COOKIE, "comfy_colab"} and not k.startswith("CF_")}
        if backend_cookies:
            from http.cookies import SimpleCookie
            cookies = SimpleCookie()
            for k, v in backend_cookies.items():
                cookies[k] = v
            headers["Cookie"] = cookies.output(header="", sep=";").strip()
        url = upstream + request.rel_url.raw_path_qs
        if request.headers.get("Upgrade", "").lower() == "websocket":
            # Each hop negotiates its own handshake. Forwarding the browser's
            # permessage-deflate offer makes ComfyUI compress frames this client cannot read.
            ws_headers = {k: v for k, v in headers.items() if not k.lower().startswith("sec-websocket-")}
            protocols = [p.strip() for p in request.headers.get("Sec-WebSocket-Protocol", "").split(",") if p.strip()]
            async with request.app[client_key].ws_connect(url, headers=ws_headers, protocols=protocols,
                                                        max_msg_size=0) as remote:
                local = web.WebSocketResponse(max_msg_size=0, protocols=[remote.protocol] if remote.protocol else ())
                await local.prepare(request)

                async def relay(source, dest):
                    async for msg in source:
                        if msg.type == WSMsgType.TEXT:
                            await dest.send_str(msg.data)
                        elif msg.type == WSMsgType.BINARY:
                            await dest.send_bytes(msg.data)
                        elif msg.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
                            if msg.type == WSMsgType.ERROR:
                                print(t("proxy.websocket_error"), repr(source.exception()), flush=True)
                            break
                tasks = [asyncio.create_task(relay(local, remote)), asyncio.create_task(relay(remote, local))]
                try:
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    await local.close()
                return local
        try:
            async with request.app[client_key].request(request.method, url, headers=headers,
                    data=request.content.iter_chunked(1024 * 1024), allow_redirects=False) as res:
                response = web.StreamResponse(status=res.status,
                    headers=[(k, v) for k, v in res.headers.items() if k.lower() not in hop - {"content-length"}])
                response.headers["Referrer-Policy"] = "no-referrer"
                complete = request.method == "GET" and res.status == 200
                response.headers["Cache-Control"] = ("private, max-age=31536000, immutable"
                    if complete and hashed_asset.fullmatch(request.path) else "no-store")
                # ComfyUI sends its 26 MB frontend uncompressed and the tunnel is the slow hop.
                if (complete and "Content-Encoding" not in res.headers
                        and res.headers.get("Content-Type", "").lower().startswith(compressible)
                        and "gzip" in request.headers.get("Accept-Encoding", "").lower()):
                    response.headers["Vary"] = "Accept-Encoding"
                    response.enable_compression(web.ContentCoding.gzip)
                await response.prepare(request)
                async for chunk in res.content.iter_chunked(1024 * 1024):
                    await response.write(chunk)
                await response.write_eof()
                return response
        except (ConnectionError, OSError):
            raise web.HTTPBadGateway(text=t("proxy.upstream_down")) from None

    app = web.Application(client_max_size=1024 ** 3)
    app.cleanup_ctx.append(lifecycle)
    app.router.add_route("*", "/{path:.*}", handler)
    return app


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
    auth = {"Cookie": COOKIE + "=" + config["cookie_secret"]} if config.get("cookie_secret") else None
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
    result = subprocess.run(app_command([sys.executable, "-c", probe]), env=application_environment(),
                            input=json.dumps([sorted(services), sockets]), capture_output=True, text=True, timeout=600)
    owners = subprocess.run(["ss", "-H", "-ltnp"], capture_output=True, text=True).stdout if shutil.which("ss") else ""
    print(t("isolation.heading", user=APP_USER))
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


def backend_command():
    repo = BASE / "environment/ComfyUI"
    return [str(BASE / ".venv/bin/python"), str(repo / "main.py"),
            "--listen", "127.0.0.1", "--port", "8188", "--enable-manager",
            "--base-directory", str(repo), "--user-directory", str(repo / "user"),
            "--input-directory", str(BASE / "input"), "--output-directory", str(BASE / "output"),
            "--temp-directory", str(BASE / "temp"), "--extra-model-paths-config", str(BASE / "models.yaml"),
            "--database-url", "sqlite:///" + str(BASE / "database/comfyui.db")]


def supervise(config):
    # Background process persists independently of the notebook cell.
    def save_loop():
        while True:
            time.sleep(config["checkpoint_seconds"])
            try:
                checkpoint(config)
            except Exception as exc:
                private_write(PRIVATE / "checkpoint-status.json", json.dumps({"ok": False, "error": str(exc)}))
                print(t("supervisor.autosave_failed"), flush=True)
    threading.Thread(target=save_loop, daemon=True).start()
    env = application_environment()
    for _ in range(5):
        with (PRIVATE / "comfy.log").open("a") as log:
            child = subprocess.Popen(app_command(backend_command()), cwd=BASE / "environment/ComfyUI", env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
            child.wait()
        try:
            checkpoint(config, force=True)
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


def access_certs_url(team_domain):
    return f"https://{team_domain}/cdn-cgi/access/certs"


def access_settings(hostname, audience):
    """What the notebook was told about the Cloudflare side, checked before anything starts."""
    hostname = re.sub(r"^https://|/+$", "", hostname.strip().lower())
    if not re.fullmatch(r"(?=.{4,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9-]{2,63}", hostname):
        raise ValueError(t("access.invalid_hostname"))
    audience = (audience or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", audience):
        raise ValueError(t("access.invalid_audience"))
    return {"hostname": hostname, "aud": audience}


def access_team_domain(hostname):
    """The team whose sign-in Cloudflare Access puts in front of the hostname.

    Asked the way a stranger would, over verified HTTPS: only whoever controls the hostname can
    make it send visitors elsewhere, and then no setting here would help. Without that redirect
    there is nothing to check signatures against, so nothing is started."""
    import requests
    status = "-"
    for _ in range(4):
        try:
            answer = requests.get("https://" + hostname + "/", timeout=10, allow_redirects=False)
        except requests.RequestException:
            time.sleep(3)
            continue
        sign_in = urlsplit(answer.headers.get("Location", ""))
        if (answer.status_code in (301, 302, 303, 307, 308) and sign_in.scheme == "https"
                and re.fullmatch(r"[a-z0-9][a-z0-9-]*\.cloudflareaccess\.com", sign_in.netloc)
                and sign_in.path.startswith("/cdn-cgi/access/login")):
            return sign_in.netloc
        status = answer.status_code
        time.sleep(3)
    raise RuntimeError(t("access.unprotected", hostname=hostname, status=status))


def tunnel_services(log):
    """Where the named tunnel sends its hostnames: the services in the latest settings cloudflared
    reports having received from Cloudflare. None while it has reported none that can be read."""
    services = None
    for line in log.splitlines():
        if "Updated to new configuration" not in line:
            continue
        found = re.search(r'config"?[=:]\s*"((?:[^"\\]|\\.)*)"', line)
        try:
            rules = json.loads(json.loads('"' + found.group(1) + '"'))["ingress"]
            services = [str(rule["service"]) for rule in rules]
        except (AttributeError, ValueError, KeyError, TypeError):
            services = None
    return services


def tunnel_leads_to_proxy(services):
    """Every hostname of the tunnel has to end at the proxy. Any other port of this runtime it
    named would be served to whoever gets past Cloudflare, without the checks made here."""
    proxy = [service for service in services if service.rstrip("/") in ("http://127.0.0.1:8189", "http://localhost:8189")]
    others = [service for service in services if service not in proxy and not re.fullmatch(r"http_status:\d{3}", service)]
    return bool(proxy) and not others


def start_with_access(config, hostname, audience, tunnel_token):
    """Start behind a named Cloudflare tunnel whose hostname Cloudflare Access protects."""
    access = access_settings(hostname, audience)
    token = (tunnel_token or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9+/=_-]{40,4096}", token):
        raise ValueError(t("access.invalid_token"))
    if (PRIVATE / "processes.json").exists():
        raise RuntimeError(t("start.already_running"))
    access["team"] = access_team_domain(access["hostname"])
    print(t("access.team", team=access["team"]))
    start(config, "access", access=access, tunnel_token=token)


def start(config, mode, access=None, tunnel_token=""):
    """Start ComfyUI and make it reachable: mode "colab" through Colab's port forwarding,
    "tunnel" through a Quick Tunnel that asks each browser for a pairing code, "access"
    through a named tunnel on a hostname that Cloudflare Access protects."""
    import requests
    if mode not in ("colab", "tunnel", "access") or (mode == "access") != bool(access and tunnel_token):
        raise ValueError(mode)
    if (PRIVATE / "processes.json").exists():
        raise RuntimeError(t("start.already_running"))
    refuse_unfinished_update()
    for name in SESSION_KEYS:
        config.pop(name, None)
    config["mode"] = mode
    if mode == "colab":
        address = colab_forwarded_address(8189)
        config["origins"] = [address.rstrip("/")]
    elif mode == "tunnel":
        # A cookie from an earlier start (browser, Desktop) stops working.
        config["cookie_secret"] = secrets.token_urlsafe(32)
    else:
        config["access"] = access  # Not the tunnel's token: only cloudflared is given that.
    private_write(CONFIG, json.dumps(config))
    module = Path(__file__).resolve()
    processes = {}

    def launch(name):
        with (PRIVATE / (name + ".log")).open("a") as log:
            process = subprocess.Popen([sys.executable, str(module), name.replace("supervisor", "supervise")],
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes[name] = process.pid
        private_write(PRIVATE / "processes.json", json.dumps(processes))
        return process
    try:
        # A process left behind by a custom node could take the proxy's port and receive the tunnel.
        stop_app_processes()
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
            return
        logfile = PRIVATE / "tunnel.log"
        logfile.write_text("")
        with logfile.open("a") as log:
            if mode == "access":
                # The token goes in the environment: a command line can be read by every user here.
                tunnel = subprocess.Popen([str(PRIVATE / "cloudflared"), "tunnel", "--no-autoupdate", "run"],
                                          env={**os.environ, "TUNNEL_TOKEN": tunnel_token},
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            else:
                tunnel = subprocess.Popen([str(PRIVATE / "cloudflared"), "tunnel", "--no-autoupdate",
                        "--url", "http://127.0.0.1:8189"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes["tunnel"] = tunnel.pid
        private_write(PRIVATE / "processes.json", json.dumps(processes))
        if mode == "access":
            services = connected = None
            for second in range(60):
                log = logfile.read_text()
                services = tunnel_services(log)
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
            elif not tunnel_leads_to_proxy(services):
                raise RuntimeError(t("access.wrong_target", services=", ".join(services)[:300]))
            print(t("access.heading"))
            print("https://" + access["hostname"] + "/")
            print(t("access.note"))
            print(t("start.url_auto_download"))
            return
        for _ in range(90):
            match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", logfile.read_text())
            if match:
                url = match.group(0)
                config["remote_url"] = url
                private_write(CONFIG, json.dumps(config))
                show_pairing(config)
                print(t("start.url_auto_download"))
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
        checkpoint(config, force=True)  # A failed save must not look like successful shutdown.
        print(t("stop.saved"))
    file = PRIVATE / "processes.json"
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
    stop_app_processes()
    (PRIVATE / "pairing.json").unlink(missing_ok=True)
    for name in SESSION_KEYS:
        config.pop(name, None)
    private_write(CONFIG, json.dumps(config))
    print(t("stop.done"))


def zip_outputs(destination="/content/comfy-output.zip"):
    try:
        with open(destination, "wb") as file:
            app_action("zip-output", stdout=file, limit=1024 ** 4)
    except Exception:
        Path(destination).unlink(missing_ok=True)
        raise
    return str(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["proxy", "supervise", "diagnose", "app"])
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == "app":
        if drops_privileges():
            raise SystemExit("app actions must run as the application user")
        BASE = Path(args.rest[0])
        app_tree_action(args.rest[1], args.rest[2:], sys.stdin.buffer, sys.stdout.buffer)
    elif args.action == "proxy":
        from aiohttp import web
        web.run_app(create_proxy(json.loads(CONFIG.read_text())), host="127.0.0.1", port=8189, access_log=None)
    elif args.action == "diagnose":
        asyncio.run(diagnose(json.loads(CONFIG.read_text())))
    else:
        supervise(json.loads(CONFIG.read_text()))
