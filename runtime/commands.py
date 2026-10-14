"""Running commands, and running them as the unprivileged application user."""

import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time

from . import paths
from .messages import t


APP_USER = "comfy"
SETPRIV = "/usr/bin/setpriv"
# Bounds on what root accepts from, and how long it waits for, commands run as that user.
APP_TIMEOUT = 3600
APP_OUTPUT_LIMIT = 16 * 1024 ** 2


GPU_ENVIRONMENT_KEYS = (
    "LD_LIBRARY_PATH", "LD_PRELOAD", "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_PATH",
    "CUDA_MODULE_LOADING", "CUDA_DEVICE_ORDER", "CUDA_CACHE_PATH",
    "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "PYTORCH_NVML_BASED_CUDA_CHECK",
)


def application_environment():
    """Preserve GPU loader/device settings without copying notebook credentials."""
    env = {"PATH": f"{paths.BASE / '.venv/bin'}:/usr/local/bin:/usr/bin:/bin",
           "HOME": str(paths.BASE / "home"), "LANG": "C.UTF-8",
           "PYTHONUNBUFFERED": "1", "VIRTUAL_ENV": str(paths.BASE / ".venv"),
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


# The folder this package is in. python -m finds the package in its working directory and puts
# that directory first on the import path: the processes below start there, never in BASE.
OWN_DIRECTORY = Path(__file__).resolve().parent.parent


def own_command(action, *args):
    """This package run as a program (__main__.py). Start it with cwd=OWN_DIRECTORY."""
    return [sys.executable, "-m", __package__, action, *map(str, args)]


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
        run(["useradd", "--system", "--user-group", "--no-create-home", "--home-dir", paths.BASE / "home",
             "--shell", "/usr/sbin/nologin", APP_USER])
    # GPU device nodes are normally open to everyone; join their groups where they are not.
    for device in [*Path("/dev").glob("nvidia*"), *Path("/dev/dri").glob("*")]:
        info = device.stat()
        if stat.S_ISCHR(info.st_mode) and info.st_mode & 0o060 and not info.st_mode & 0o006 and info.st_gid:
            if subprocess.run(["getent", "group", str(info.st_gid)], capture_output=True).returncode:
                run(["groupadd", "--gid", info.st_gid, f"{APP_USER}-device-{info.st_gid}"])
            run(["usermod", "--append", "--groups", info.st_gid, APP_USER])
