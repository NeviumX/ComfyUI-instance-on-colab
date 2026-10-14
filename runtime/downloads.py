"""Downloading models from Hugging Face and Civitai (cell 4)."""

import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
from urllib.parse import urlsplit, urljoin, quote, unquote, parse_qs

from . import app_tree, drive, filesystem, paths
from .messages import t


def model_download_url(source, *, hf_repo="", hf_file="", hf_revision="main", civitai_version=""):
    if source == "Hugging Face":
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", hf_repo):
            raise ValueError(t("download.hf_repository"))
        file = filesystem.safe_relative(hf_file)
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


def fetch_model(config, url, folder, filename="", sha256="", token_secret="", runtime_only=False):
    """Download the model a pasted link points at into a folder of Drive's models directory,
    or with runtime_only into the same folder of this runtime's models, leaving Drive alone."""
    download, suggested = parse_model_url(url)
    folder = filesystem.safe_relative(folder.strip().strip("/"))
    filename = filename.strip() or suggested
    if "/" in filename:
        raise ValueError(t("download.filename_has_folder"))
    if filename:
        return download_model(config, download, folder + "/" + filename, sha256.strip(), token_secret,
                              runtime_only=runtime_only)
    return download_model(config, download, folder, sha256.strip(), token_secret, name_from_server=True,
                          runtime_only=runtime_only)


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


class PlainProgress:
    """Download progress as lines of text, where the notebook's own bar is not available."""
    def __init__(self, total):
        self.total, self.done, self.shown = total, 0, 0

    def update(self, amount):
        self.done += amount
        megabytes = self.done / 1048576
        if self.total:
            step = self.done * 10 // self.total
            line = t("download.progress", done=megabytes, total=self.total / 1048576, percent=self.done * 100 / self.total)
        else:
            step = self.done // (512 * 1048576)
            line = t("download.progress_size", done=megabytes)
        if step != self.shown:
            self.shown = step
            print(line, flush=True)

    def close(self):
        pass


def download_progress(total):
    """A progress bar in the notebook's output for a download of total bytes (0: unknown)."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return PlainProgress(total)
    return tqdm(total=total or None, desc=t("download.label"), unit="B", unit_scale=True, unit_divisor=1024)


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
            length = str(response.headers.get("Content-Length", ""))
            progress = download_progress(int(length) if length.isdigit() else 0)
            try:
                with Path(target).open("wb") as file:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        file.write(chunk)
                        digest.update(chunk)
                        progress.update(len(chunk))
            finally:
                progress.close()
            if not Path(target).stat().st_size:
                raise RuntimeError(t("download.empty"))
            return digest.hexdigest(), served_filename(response.headers)
    raise RuntimeError(t("download.too_many_redirects"))


def keep_on_runtime(relative):
    """Mark a model of this runtime as not to be saved: saving sends its new models to Drive."""
    origin_file = paths.PRIVATE / "model-origins.json"
    origins = set(json.loads(origin_file.read_text())) if origin_file.exists() else set()
    filesystem.private_write(origin_file, json.dumps(sorted(origins | {relative})))


def download_model(config, url, relative_path, sha256="", token_secret="", name_from_server=False, runtime_only=False):
    """Save a model under relative_path in Drive's models folder. With name_from_server,
    relative_path is the folder and the file takes the name the site serves it under.
    With runtime_only it goes into this runtime's models folder instead and never to Drive."""
    import requests
    rel = filesystem.safe_relative(relative_path)
    parsed = checked_https_url(url)
    if parsed.hostname not in {"huggingface.co", "civitai.com"}:
        raise ValueError(t("download.host_not_allowed"))
    if parsed.query or parsed.fragment:
        raise ValueError(t("download.url_has_query"))
    if sha256 and not re.fullmatch(r"[a-fA-F0-9]{64}", sha256):
        raise ValueError(t("download.invalid_sha256"))
    if runtime_only:
        # Checked and completed where only root can go, then handed to the application user.
        folder = paths.PRIVATE / "downloads"
        if not name_from_server and os.path.lexists(paths.BASE / "models" / rel):
            raise FileExistsError(t("download.exists_runtime"))
    else:
        models = drive.drive_root(config) / "models"
        models.mkdir(exist_ok=True)
        target = filesystem.contained_path(models, rel)
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
            rel = filesystem.safe_relative(rel + "/" + served)
        if runtime_only:
            if os.path.lexists(paths.BASE / "models" / rel):
                raise FileExistsError(t("download.exists_runtime"))
            with partial.open("rb") as stream:
                app_tree.app_action("put", "models/" + rel, stdin=stream)
            keep_on_runtime(rel)
            print(t("download.saved_runtime"), rel, "SHA256:", digest)
            return
        target = filesystem.contained_path(models, rel)
        if target.exists():
            raise FileExistsError(t("download.exists_not_overwritten"))
        os.replace(partial, target)
        drive.sync_models(config)
        print(t("download.saved"), rel, "SHA256:", digest)
    except requests.RequestException:
        raise RuntimeError(t("download.connection_failed")) from None
    finally:
        partial.unlink(missing_ok=True)
