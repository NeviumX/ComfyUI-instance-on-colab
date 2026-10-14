"""OS packages, the application's virtual environment and the Python packages in it."""

import re
import subprocess
import sys

from . import app_tree, commands, paths
from .messages import t


# The release made of Python only: Triton, which PyTorch installs on Linux, compiles its kernels.
SAGE_ATTENTION_PACKAGE = "sageattention==1.0.6"
# The call ComfyUI makes, on a small input. The first call compiles the kernels for this GPU.
SAGE_ATTENTION_PROBE = """
import torch
from sageattention import sageattn
q, k, v = (torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16) for _ in range(3))
out = sageattn(q, k, v, is_causal=False, tensor_layout="HND", sm_scale=None, smooth_k=False)
assert out.shape == q.shape and bool(torch.isfinite(out).all()), "SageAttention returned an unusable result"
"""


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
        commands.run(["apt-get", "update", "-qq"])
        APT_INDEX_REFRESHED.append(True)
    commands.run(["apt-get", "install", "-y", "-qq", *missing])


def install_packages(*args):
    """Install into the application's virtual environment with uv. It takes the same indexes
    and pins as pip but downloads and unpacks the multi-gigabyte PyTorch stack in parallel."""
    py = paths.BASE / ".venv/bin/python"
    commands.run([py, "-m", "uv", "pip", "install", "--python", py, *args], app=True)


def create_venv():
    """Create the application's virtual environment with pip and uv in it.

    The notebook's own pip installs them, so nothing is needed from apt: installing
    pythonX.Y-venv there upgrades the interpreter the running kernel was started from."""
    venv = paths.BASE / ".venv"
    py = venv / "bin/python"
    commands.run([sys.executable, "-m", "venv", "--without-pip", venv], app=True, capture=True)
    try:
        commands.run([sys.executable, "-m", "pip", "--python", py, "install", "pip", "uv"], app=True)
    except RuntimeError:
        print(t("venv.pip_fallback"), flush=True)
        ensure_venv_support()
        commands.run([sys.executable, "-m", "venv", venv], app=True, capture=True)
        commands.run([py, "-m", "pip", "install", "uv"], app=True)
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


def write_lock():
    """Record the installed packages, to install the same ones on the next runtime."""
    py = paths.BASE / ".venv/bin/python"
    # No editable or local paths: those cannot survive a new Colab VM.
    packages = checked_lock(commands.run([py, "-m", "pip", "freeze"], app=True, capture=True, timeout=300))
    lock = "environment/requirements.lock.txt"
    if not (paths.BASE / lock).exists() or app_tree.app_action("get", lock).decode() != packages:
        app_tree.app_action("put", lock, stdin=packages.encode())


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


def prepare_sage_attention():
    """Install SageAttention unless the environment has one already, and try it on this GPU.

    Told to use it, ComfyUI exits at start when the package is absent, and logs an error at
    every attention call when its kernels cannot be compiled. Both are found out here, and
    ComfyUI is then started without it."""
    py = paths.BASE / ".venv/bin/python"
    try:
        found = commands.run([py, "-c", "import importlib.util; print(importlib.util.find_spec('sageattention') is not None)"],
                             app=True, capture=True, timeout=60)
        if found.strip() != "True":  # A restored environment keeps the version it was saved with.
            install_packages(SAGE_ATTENTION_PACKAGE)
        commands.run([py, "-c", SAGE_ATTENTION_PROBE], app=True, timeout=600)
    except RuntimeError:
        print(t("setup.sage_unavailable"))
        return False
    print(t("setup.sage_enabled"))
    return True
