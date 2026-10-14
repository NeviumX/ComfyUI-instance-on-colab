"""Small file helpers: hashing, root-only files and paths that must stay inside a folder."""

import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from .messages import t


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
