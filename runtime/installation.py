"""Setting up a runtime (cell 3) and updating ComfyUI (cell 11)."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

from . import app_tree, cloudflare, commands, dependencies, drive, environment, filesystem, paths
from .messages import t


def setup(*, root_name="ComfyUI-Colab", comfy_ref="v0.39.0",
          checkpoint_seconds=120, apt_packages=(), restore=True, model_access="direct",
          sage_attention=True, fp16_accumulation=False):
    if sys.platform != "linux" or not Path("/content").exists() or os.geteuid() != 0:
        raise RuntimeError(t("setup.not_colab"))
    if paths.CONFIG.exists():
        raise RuntimeError(t("setup.already_done"))
    filesystem.safe_relative(root_name)
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", comfy_ref) or comfy_ref.startswith("-"):
        raise ValueError(t("setup.invalid_ref"))
    if checkpoint_seconds < 60:
        raise ValueError(t("setup.interval_too_short"))
    if any(not re.fullmatch(r"[a-z0-9][a-z0-9+.-]*", p) for p in apt_packages):
        raise ValueError(t("setup.invalid_apt_package"))
    if subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL).returncode:
        raise RuntimeError(t("setup.no_gpu"))
    if paths.PRIVATE.is_symlink():
        raise RuntimeError(t("drive.private_symlink"))
    paths.PRIVATE.mkdir(mode=0o700, exist_ok=True)
    paths.PRIVATE.chmod(0o700)
    dependencies.apt_install(["git", "ffmpeg", "libgl1", "libglib2.0-0", *apt_packages])
    drive.mount_drive()
    if model_access not in ("direct", "copy"):
        raise ValueError(t("setup.invalid_model_access"))
    config = {"drive_folder": root_name,
              "python": f"{sys.version_info.major}.{sys.version_info.minor}",
              "checkpoint_seconds": checkpoint_seconds, "model_access": model_access,
              "fp16_accumulation": bool(fp16_accumulation)}
    storage = drive.drive_root(config)
    for directory in (storage, storage / "environment", *(storage / "models" / name for name in paths.MODEL_FOLDERS)):
        directory.mkdir(parents=True, exist_ok=True)
    # Test the selected folder, not any other location in Drive.
    probe = storage / "environment/write-probe.txt"
    probe.write_text("ComfyUI-Colab write test")
    # The application user owns its working tree from here on. Root reads and writes inside
    # it only through app_action, because code running there can plant links.
    envdir = paths.BASE / "environment"
    commands.ensure_app_user()
    app_tree.create_app_tree()
    if model_access == "direct":
        drive.show_drive_models(config)  # Before the long steps: if it cannot be done, that is known at once.
    pointer_path = storage / "environment/latest.json"
    if pointer_path.exists() and not restore:
        print(t("setup.fresh_environment"))
    elif pointer_path.exists():
        pointer = json.loads(pointer_path.read_text())
        if pointer["python"] != config["python"]:
            raise RuntimeError(t("setup.python_changed"))
        filesystem.private_write(paths.PRIVATE / "saved-parts.json", json.dumps(environment.restore_environment(storage / "environment", pointer)))
        print(t("setup.restored"), pointer["saved_at"])
        nodes = sorted(item.name for item in (envdir / "ComfyUI/custom_nodes").glob("*")
                       if item.is_dir() and item.name != "__pycache__")
        # Restored nodes run again on this runtime; show them so an unwanted one is noticed.
        print(t("setup.restored_nodes"), ", ".join(nodes) if nodes else t("common.none"))
    repo = envdir / "ComfyUI"
    if not repo.exists():
        commands.run(["git", "clone", "https://github.com/Comfy-Org/ComfyUI.git", repo], app=True)
        commands.run(["git", "checkout", comfy_ref], app=True, cwd=repo)
    if not (repo / "main.py").exists():
        raise RuntimeError(t("setup.source_incomplete"))
    app_tree.app_action("prepare")
    previous_meta = (json.loads(app_tree.app_action("get", "environment/runtime.json"))
                     if (envdir / "runtime.json").exists() else {})
    # The saved list is a file the application user could rewrite, and apt runs as root:
    # name what was used before and leave the decision to install it to the notebook's setting.
    wanted = dependencies.unlisted_apt_packages(previous_meta, apt_packages)
    if wanted:
        print(t("setup.saved_apt_packages"), " ".join(wanted))
        print(t("setup.saved_apt_packages_hint"))
    # Match the CUDA build already supported by the current Colab runtime.
    # The mirror is persisted because pip freeze can contain +cuXXX versions.
    lock = envdir / "requirements.lock.txt"
    locked = lock.exists()
    if locked:
        dependencies.checked_lock(app_tree.app_action("get", "environment/requirements.lock.txt").decode())
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
    dependencies.checked_torch_index(torch_index)
    app_tree.app_action("put", "environment/runtime.json", stdin=json.dumps(
        {"apt_packages": sorted(apt_packages), "python": config["python"], "torch_index_url": torch_index}).encode())
    py = dependencies.create_venv()
    if locked:
        # pip looks for each package on every index; uv does only when told to.
        dependencies.install_packages("--index-strategy", "unsafe-best-match", "--extra-index-url", torch_index, "-r", lock)
    else:
        dependencies.install_packages("--index-url", torch_index, *torch_packages)
        dependencies.install_packages("-r", repo / "requirements.txt", "-r", repo / "manager_requirements.txt")
        # Explicit requirements are useful for custom nodes added outside Manager.
        for req in sorted((repo / "custom_nodes").glob("*/requirements.txt")):
            dependencies.install_packages("-r", req)
    commands.run([py, "-m", "pip", "check"], app=True)
    commands.run([py, "-c", "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); torch.cuda.init(); print('GPU:', torch.cuda.get_device_name(0))"], app=True)
    config["sage_attention"] = bool(sage_attention) and dependencies.prepare_sage_attention()
    if model_access == "copy":
        print(t("setup.copying_models"))
    drive.sync_models(config)
    cloudflare.install_cloudflared()
    filesystem.private_write(paths.CONFIG, json.dumps(config))
    environment.checkpoint(config, force=True)
    print(t("setup.done") + root_name + "/models")
    return config


def update_comfyui(config, ref):
    """Move ComfyUI to another version and install what that version requires.

    Custom nodes, settings, workflows and the database are not files of the ComfyUI
    repository, so they stay as they are. Drive receives the environment only once the new
    version is installed and consistent; an update that fails is undone."""
    if (paths.PRIVATE / "processes.json").exists():
        raise RuntimeError(t("update.running"))
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", ref) or ref.startswith("-"):
        raise ValueError(t("setup.invalid_ref"))
    drive.drive_root(config)  # The result is saved at the end: find out now if Drive is gone.
    repo = paths.BASE / "environment/ComfyUI"
    py = paths.BASE / ".venv/bin/python"
    lock = "environment/requirements.lock.txt"
    marker = paths.PRIVATE / "update-incomplete.json"

    def git(*args):
        return commands.run(["git", *args], app=True, capture=True, cwd=repo, timeout=600).strip()

    def commit(name):
        # The repository belongs to the application user: take nothing but a commit id from it.
        try:
            found = git("rev-parse", "--verify", "--quiet", name + "^{commit}")
        except RuntimeError:
            return ""
        return found if re.fullmatch(r"[0-9a-f]{40,64}", found) else ""

    def switch(target):
        git("checkout", "--force", "--detach", target)
        app_tree.app_action("prepare")  # The checkout puts back the models folder ComfyUI ships.

    def package_names(listing):
        return {re.split(r"==| @ ", line)[0] for line in listing.splitlines() if dependencies.LOCK_LINE.fullmatch(line)}

    def undo(state):
        """Back to the version and the packages recorded before the update began."""
        packages = dependencies.checked_lock(app_tree.app_action("get", lock).decode())
        index = dependencies.checked_torch_index(json.loads(app_tree.app_action("get", "environment/runtime.json")).get("torch_index_url"))
        switch(state["commit"])
        dependencies.install_packages("--index-strategy", "unsafe-best-match", "--extra-index-url", index, "-r", paths.BASE / lock)
        added = sorted(package_names(commands.run([py, "-m", "pip", "freeze"], app=True, capture=True, timeout=300))
                       - package_names(packages))
        if added:
            commands.run([py, "-m", "uv", "pip", "uninstall", "--python", py, *added], app=True)
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
    dependencies.write_lock()
    filesystem.private_write(marker, json.dumps(state))
    try:
        switch(target)
        dependencies.install_packages(*(argument for name in ("requirements.txt", "manager_requirements.txt")
                                        if (repo / name).exists() for argument in ("-r", repo / name)))
        commands.run([py, "-m", "pip", "check"], app=True)
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
    environment.checkpoint(config, force=True)
    print(t("update.done", previous=state["version"], version=version))
    return True
