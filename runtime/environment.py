"""Saving the environment to Drive in parts and restoring it."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import time

from . import app_tree, commands, dependencies, drive, filesystem, paths
from .messages import t


EXCLUDED = {"models", "output", "temp", "input"}


ARCHIVE_LIMIT = 4 * 1024 ** 3


def excluded_path(parts):
    return "__pycache__" in parts or ".venv" in parts or (
        len(parts) >= 2 and parts[0] == "ComfyUI" and parts[1] in EXCLUDED)


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
            filesystem.safe_relative(member.name)
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError(t("environment.archive_special_files"))
        if not hasattr(tarfile, "data_filter"):
            raise RuntimeError(t("environment.archive_needs_data_filter"))
        tf.extractall(dest, filter="data")


GENERATIONS = 3
PART_NAME = re.compile(r"[a-f0-9]{64}\.tar\.gz")
GENERATION_NAME = re.compile(r"\d{8}T\d{6}Z-[a-f0-9]{12}\.json")
WHOLE_ARCHIVE_NAME = re.compile(r"\d{8}T\d{6}Z-[a-f0-9]{12}\.tar\.gz")


def component_of(parts):
    """The separately saved part of the environment a file belongs to: ComfyUI itself, each
    custom node, the user's data, and the few files kept beside ComfyUI. Saving a workflow
    then sends the user's data, not the custom nodes with it."""
    if parts[0] != "ComfyUI":
        return "state"
    if len(parts) >= 3 and parts[1] == "user":
        return "user"
    if len(parts) >= 4 and parts[1] == "custom_nodes":
        return "node/" + parts[2]
    return "core"


def component_fingerprints(root):
    """{component: fingerprint of its files}, from their names, sizes and times."""
    entries = {}
    for parent, dirs, files in os.walk(root, followlinks=False):
        relative = Path(parent).relative_to(root)
        dirs[:] = sorted(d for d in dirs if not excluded_path((relative / d).parts)
                         and not (Path(parent) / d).is_symlink())
        for name in sorted(files):
            p = Path(parent) / name
            if p.is_symlink() or name.endswith(".pyc"):
                continue
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            path = relative / name
            entries.setdefault(component_of(path.parts), []).append((path.as_posix(), st.st_size, st.st_mtime_ns))
    return {name: hashlib.sha256(json.dumps(found).encode()).hexdigest() for name, found in entries.items()}


def discard_old_saves(storage):
    """Keep the newest saves, and of the parts only what one of them needs. Nothing is deleted
    but files with the names this notebook gives its own."""
    def named(folder, pattern):
        return sorted((item for item in folder.iterdir() if not item.is_symlink() and pattern.fullmatch(item.name)),
                      key=lambda item: item.name, reverse=True)
    try:
        current = (storage / "latest.json").read_text()
        saves = {save: save.read_text() for save in named(storage / "generations", GENERATION_NAME)}
        # The save Drive points to is kept whatever its name says of its age, then the newest.
        kept = sorted(saves, key=lambda save: saves[save] != current)[:GENERATIONS]
        lists = [json.loads(current), *(json.loads(saves[save]) for save in kept)]
        needed = {part["file"] for save in lists for part in save.get("parts", {}).values()}
        needed |= {save["archive"] for save in lists if "archive" in save}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return  # A save that cannot be read: delete nothing it might need.
    for old in set(saves) - set(kept):
        old.unlink()
    for part in named(storage / "parts", PART_NAME):
        if part.name not in needed:
            part.unlink()
    # Whole-environment archives, from before the parts were saved separately, count as older saves.
    for old in named(storage, WHOLE_ARCHIVE_NAME)[max(0, GENERATIONS - len(kept)):]:
        if old.name not in needed:
            old.unlink()


def checkpoint(config, force=False):
    """Save what changed. Each part of the environment is its own archive on Drive, named after
    its content; a save is a short list of the parts it consists of. Only parts that changed
    since the last save are packed and sent. force: also make sure Drive still has the rest."""
    import fcntl
    with (paths.PRIVATE / "checkpoint.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        refuse_unfinished_update()
        envdir = paths.BASE / "environment"
        drive.save_new_cache_models(config, force=force)
        storage = drive.drive_root(config) / "environment"
        for folder in (storage, storage / "parts", storage / "generations"):
            folder.mkdir(exist_ok=True)
        database = paths.BASE / "database/comfyui.db"
        if database.exists():
            signature = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in
                         (database, database.with_name(database.name + "-wal")) if p.exists()]
            signature_text = json.dumps(signature)
            db_marker = paths.PRIVATE / "database-signature"
            if force or not db_marker.exists() or db_marker.read_text() != signature_text:
                app_tree.app_action("backup-database")
                db_marker.write_text(signature_text)
        dependencies.write_lock()
        record = paths.PRIVATE / "saved-parts.json"
        saved = json.loads(record.read_text()) if record.exists() else {}
        before = component_fingerprints(envdir)
        parts, sent = {}, []
        staging = paths.PRIVATE / "part.tar.gz"
        for name in sorted(before):
            known = saved.get(name)
            if known and known["fingerprint"] == before[name] and (
                    not force or filesystem.contained_path(storage / "parts", known["file"]).exists()):
                parts[name] = known
                continue
            try:
                with staging.open("wb") as file:
                    app_tree.app_action("archive", name, stdout=file, limit=ARCHIVE_LIMIT, timeout=1800)
                digest = filesystem.file_hash(staging)
                filesystem.atomic_copy(staging, filesystem.contained_path(storage / "parts", digest + ".tar.gz"))
            finally:
                staging.unlink(missing_ok=True)
            parts[name] = {"fingerprint": before[name], "file": digest + ".tar.gz", "sha256": digest}
            sent.append(name)
        if not sent and set(parts) == set(saved):
            return False
        after = component_fingerprints(envdir)
        if any(after.get(name) != before[name] for name in sent):
            raise RuntimeError(t("environment.busy"))
        # The list is written last: until then the preceding save is the one Drive points to.
        save = {"format": 2, "python": config["python"],
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "parts": {name: {"file": part["file"], "sha256": part["sha256"]} for name, part in parts.items()}}
        text = json.dumps(save, sort_keys=True)
        name = (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-"
                + hashlib.sha256(text.encode()).hexdigest()[:12] + ".json")
        filesystem.private_write(paths.PRIVATE / "latest.json", text)
        filesystem.atomic_copy(paths.PRIVATE / "latest.json", filesystem.contained_path(storage / "generations", name))
        filesystem.atomic_copy(paths.PRIVATE / "latest.json", filesystem.contained_path(storage, "latest.json"))
        filesystem.private_write(record, json.dumps(parts))
        filesystem.private_write(paths.PRIVATE / "checkpoint-status.json", json.dumps(
            {"ok": True, "saved_at": save["saved_at"], "python": save["python"], "parts": len(parts), "sent": sent}))
        discard_old_saves(storage)
        return True


def restore_environment(storage, save):
    """Bring a saved environment back, part by part (or from the single archive of earlier
    saves). Returns what this runtime then holds that Drive already has, for the next save."""
    archive = paths.PRIVATE / "restore.tar.gz"

    def bring(folder, name, digest):
        filesystem.atomic_copy(filesystem.contained_path(folder, name), archive)
        if filesystem.file_hash(archive) != digest:
            raise RuntimeError(t("setup.archive_sha256_mismatch"))
        with archive.open("rb") as stream:
            app_tree.app_action("restore", stdin=stream, timeout=2 * commands.APP_TIMEOUT)
        archive.unlink()

    if "parts" not in save:
        name = filesystem.safe_relative(save["archive"])
        if "/" in name or not name.endswith(".tar.gz"):
            raise RuntimeError(t("setup.invalid_archive_name"))
        bring(storage, name, save["sha256"])
        return {}
    parts = save["parts"]
    for part in parts.values():
        if not PART_NAME.fullmatch(str(part["file"])) or part["file"] != str(part["sha256"]) + ".tar.gz":
            raise RuntimeError(t("setup.invalid_archive_name"))
    for name in sorted(parts, key=lambda name: (name != "core", name)):  # ComfyUI first, then what goes into it.
        bring(storage / "parts", parts[name]["file"], parts[name]["sha256"])
    restored = component_fingerprints(paths.BASE / "environment")
    return {name: {"fingerprint": restored[name], "file": parts[name]["file"], "sha256": parts[name]["sha256"]}
            for name in parts if name in restored}


def refuse_unfinished_update():
    """A ComfyUI source and packages of two different versions must not be saved or started."""
    if (paths.PRIVATE / "update-incomplete.json").exists():
        raise RuntimeError(t("update.incomplete"))
