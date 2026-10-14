"""The application user's working tree (BASE) and the file operations root has done in it."""

import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

from . import commands, environment, filesystem, paths
from .messages import t


def create_app_tree():
    """Make the application user's working directories. Root hands over only what it has just
    made and never follows a link: on a rerun after a failed setup, the tree may already hold
    whatever that user's code left there."""
    import pwd
    account = pwd.getpwnam(commands.APP_USER)
    created = []
    for directory in (paths.BASE, *(paths.BASE / name for name in ("environment", "models", "output", "temp", "input", "home", "database"))):
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
    envdir = paths.BASE / "environment"
    repo = envdir / "ComfyUI"
    if action == "put":
        target = filesystem.contained_path(paths.BASE, args[0])
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
        with filesystem.contained_path(paths.BASE, args[0]).open("rb") as file:
            shutil.copyfileobj(file, sink, 4 * 1024 * 1024)
    elif action == "prepare":
        # Keep the core default and Manager's direct models path on the same storage.
        for name in paths.MODEL_FOLDERS:
            (paths.BASE / "models" / name).mkdir(parents=True, exist_ok=True)
        repo_models = repo / "models"
        if repo_models.exists() and not repo_models.is_symlink():
            # What is already there stays: it may be a link into the read-only view of Drive.
            shutil.copytree(repo_models, paths.BASE / "models", dirs_exist_ok=True,
                            copy_function=lambda source, target: os.path.lexists(target) or shutil.copy2(source, target))
            shutil.rmtree(repo_models)
        if not repo_models.exists():
            repo_models.symlink_to(paths.BASE / "models", target_is_directory=True)
        model_cfg = {"base_path": str(paths.BASE / "models"), "is_default": True, **{name: name for name in paths.MODEL_FOLDERS}}
        (paths.BASE / "models.yaml").write_text(json.dumps({"colab_drive_cache": model_cfg}))  # JSON is valid YAML.
        for name in ("user", "custom_nodes"):
            (repo / name).mkdir(exist_ok=True)
        if (repo / "user/comfyui.db").exists():
            shutil.copyfile(repo / "user/comfyui.db", paths.BASE / "database/comfyui.db")
    elif action == "restore":
        fd, temporary = tempfile.mkstemp(prefix=".comfy-", suffix=".tar.gz", dir=paths.BASE)
        try:
            with os.fdopen(fd, "wb") as file:
                shutil.copyfileobj(source, file, 4 * 1024 * 1024)
            environment.restore_archive(temporary, envdir)
        finally:
            Path(temporary).unlink(missing_ok=True)
    elif action == "backup-database":
        # SQLite's online backup includes committed WAL changes consistently.
        import sqlite3
        with sqlite3.connect(f"file:{paths.BASE / 'database/comfyui.db'}?mode=ro", uri=True) as database:
            with sqlite3.connect(repo / "user/comfyui.db") as target:
                database.backup(target)
    elif action == "archive":
        # One part of the environment, as component_of() divides it.
        component = args[0]

        def without_other_parts(member):
            parts = PurePosixPath(member.name).parts
            if parts[:2] == ("ComfyUI", "user") or (member.isdir() and len(parts) == 3 and parts[1] == "custom_nodes"):
                return None
            return environment.archive_filter(member)
        import gzip
        # No time of packing in the archive: the same files give the same bytes, and so the same name on Drive.
        with gzip.GzipFile(fileobj=sink, mode="wb", compresslevel=1, mtime=0) as packed, tarfile.open(
                fileobj=packed, mode="w|") as tf:
            if component == "core":
                tf.add(repo, arcname="ComfyUI", filter=without_other_parts)
            elif component == "user":
                tf.add(repo / "user", arcname="ComfyUI/user", filter=environment.archive_filter)
            elif component == "state":
                for child in sorted(envdir.iterdir()):
                    if child.name != "ComfyUI":
                        tf.add(child, arcname=child.name, filter=environment.archive_filter)
            elif component.startswith("node/") and component[5:] not in ("", ".", "..") and "/" not in component[5:]:
                tf.add(repo / "custom_nodes" / component[5:], arcname="ComfyUI/custom_nodes/" + component[5:],
                       filter=environment.archive_filter)
            else:
                raise ValueError(component)
    elif action == "link-models":
        view, cache = Path(args[0]), paths.BASE / "models"
        for parent, _, files in os.walk(cache, followlinks=False):
            for name in files:
                link = Path(parent) / name
                # No longer on Drive: the link into the view leads nowhere.
                if link.is_symlink() and view in Path(os.readlink(link)).parents and not link.exists():
                    link.unlink()
        linked = 0
        for file in filesystem.regular_files(view):
            link = cache / file.relative_to(view)
            if link.parent.is_symlink() or (link.parent.exists() and not link.parent.is_dir()):
                continue
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() and os.readlink(link) != str(file):
                link.unlink()
            if not os.path.lexists(link):  # A model of this runtime with the same name stays.
                link.symlink_to(file)
                linked += 1
        sink.write(str(linked).encode())
    elif action == "zip-output":
        import zipfile
        root = paths.BASE / "output"
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


def app_action(action, *args, stdin=None, stdout=None, limit=commands.APP_OUTPUT_LIMIT, timeout=commands.APP_TIMEOUT):
    """Perform app_tree_action as the application user. stdin is bytes or a file; stdout is a file."""
    args = list(map(str, args))
    if not commands.drops_privileges():
        import io
        sink = stdout or io.BytesIO()
        app_tree_action(action, args, io.BytesIO(stdin) if isinstance(stdin, bytes) else stdin, sink)
        return b"" if stdout else sink.getvalue()
    command = commands.app_command(commands.own_command("app", paths.BASE, action, *args))
    try:
        code, output, errors = commands.bounded(command, env=commands.application_environment(), stdin=stdin, sink=stdout,
                                                limit=limit, timeout=timeout, cwd=commands.OWN_DIRECTORY)
    except RuntimeError as failure:
        raise RuntimeError(t("app.failed", action=action, failure=failure)) from None
    if code:
        detail = errors.decode(errors="replace").strip().splitlines()[-1:]
        raise RuntimeError(t("app.failed_detail", action=action) + "".join(detail))
    return output


def receive_file(target, relative, times=None, limit=commands.APP_OUTPUT_LIMIT):
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
