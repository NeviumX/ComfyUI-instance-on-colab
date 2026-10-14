"""Google Drive: mounting it, its models folder and how the runtime's models follow it."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from . import app_tree, commands, dependencies, filesystem, paths
from .messages import t


def mount_drive():
    """Mount Drive under a root-only parent, out of reach of the application user."""
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError(t("drive.needs_root"))
    if paths.PRIVATE.is_symlink() or paths.DRIVE_MOUNT.is_symlink():
        raise RuntimeError(t("drive.private_symlink"))
    paths.PRIVATE.mkdir(mode=0o700, exist_ok=True)
    paths.PRIVATE.chmod(0o700)
    if paths.PRIVATE.stat().st_uid != 0:
        raise RuntimeError(t("drive.private_owner"))
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        if len(fields) > 2 and fields[2] == "fuse.drive" and fields[1] != str(paths.DRIVE_MOUNT):
            raise RuntimeError(t("drive.mounted_elsewhere"))
    from google.colab import drive
    drive.mount(str(paths.DRIVE_MOUNT))
    paths.PRIVATE.chmod(0o700)
    if not (paths.DRIVE_MOUNT / "MyDrive").is_dir():
        raise RuntimeError(t("drive.no_mydrive"))


def drive_root(config):
    mydrive = paths.DRIVE_MOUNT / "MyDrive"
    if not mydrive.is_dir():
        raise RuntimeError(t("drive.not_mounted"))
    return filesystem.contained_path(mydrive, config["drive_folder"])


def drive_models_shown():
    return os.path.ismount(paths.DRIVE_MODELS)


def show_drive_models(config):
    """Give the application user a read-only view of Drive's models folder, and of nothing else.

    bindfs presents that one folder as the application user's own files; the Drive mount it
    reads from stays where only root can go. The view lies outside the application's tree,
    so that user cannot move the place root mounts on."""
    source = drive_root(config) / "models"
    source.mkdir(exist_ok=True)
    if drive_models_shown():
        return
    try:
        if paths.DRIVE_MODELS.is_symlink():
            raise RuntimeError(t("models.view_place", path=paths.DRIVE_MODELS))
        paths.DRIVE_MODELS.mkdir(mode=0o755, exist_ok=True)
        if paths.DRIVE_MODELS.stat().st_uid != os.geteuid() or any(paths.DRIVE_MODELS.iterdir()):
            raise RuntimeError(t("models.view_place", path=paths.DRIVE_MODELS))
        if not shutil.which("bindfs"):
            if not dependencies.APT_INDEX_REFRESHED:
                commands.run(["apt-get", "update", "-qq"])
                dependencies.APT_INDEX_REFRESHED.append(True)
            # Without what it only recommends: that would swap the FUSE tools Drive is mounted with.
            commands.run(["apt-get", "install", "-y", "-qq", "--no-install-recommends", "bindfs"])
        commands.run(["bindfs", "-r", "--force-user=" + commands.APP_USER, "--force-group=" + commands.APP_USER, "--perms=a=rX",
                      "-o", "allow_other", source, paths.DRIVE_MODELS])
        commands.run(["ls", paths.DRIVE_MODELS], app=True, capture=True)  # The application user can really look in.
    except (RuntimeError, OSError) as failure:
        hide_drive_models()
        raise RuntimeError(t("models.view_failed", reason=failure)) from None


def hide_drive_models():
    """Take the view away again: Drive cannot be unmounted while something still reads from it."""
    for command in (["umount", paths.DRIVE_MODELS], ["fusermount", "-u", paths.DRIVE_MODELS], ["umount", "-l", paths.DRIVE_MODELS]):
        if not drive_models_shown():
            return
        try:
            subprocess.run(list(map(str, command)), capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            pass


def link_models(config):
    """With the view in place nothing is copied: ComfyUI's models folder gets a link to each
    of Drive's files, next to the models downloaded on this runtime."""
    show_drive_models(config)
    linked = int(app_tree.app_action("link-models", paths.DRIVE_MODELS) or 0)
    print(t("models.linked", linked=linked))
    return linked


def sync_models(config):
    """Drive is authoritative; refresh the private runtime's working copy."""
    if config.get("model_access") == "direct":
        return link_models(config)
    source = drive_root(config) / "models"
    source.mkdir(exist_ok=True)
    copied = 0
    origin_file = paths.PRIVATE / "model-origins.json"
    origins = set(json.loads(origin_file.read_text())) if origin_file.exists() else set()
    for file in filesystem.regular_files(source):
        relative = filesystem.safe_relative(file.relative_to(source).as_posix())
        origins.add(relative)
        target = paths.BASE / "models" / relative
        st = file.stat()
        if target.exists():
            old = target.stat()
            if old.st_size == st.st_size and abs(old.st_mtime - st.st_mtime) < 1:
                continue
        with file.open("rb") as stream:
            app_tree.app_action("put", "models/" + relative, st.st_mtime_ns, stdin=stream)
        copied += 1
    filesystem.private_write(origin_file, json.dumps(sorted(origins)))
    print(t("models.synced", copied=copied))
    return copied


def save_new_cache_models(config, force=False):
    # Preserve Manager downloads without overwriting manually updated Drive files.
    destination = drive_root(config) / "models"
    destination.mkdir(exist_ok=True)
    origin_file = paths.PRIVATE / "model-origins.json"
    origins = set(json.loads(origin_file.read_text())) if origin_file.exists() else set()
    for file in filesystem.regular_files(paths.BASE / "models"):
        relative = file.relative_to(paths.BASE / "models").as_posix()
        if relative in origins:
            continue
        st = file.stat()
        if not force and time.time() - st.st_mtime < 30:
            continue
        target = filesystem.contained_path(destination, relative)
        if not target.exists():
            # No more than the size root saw: the file is the application user's to change.
            app_tree.receive_file(target, "models/" + relative, (st.st_atime_ns, st.st_mtime_ns), limit=st.st_size)
        origins.add(relative)
    filesystem.private_write(origin_file, json.dumps(sorted(origins)))
