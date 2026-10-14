"""cloudflared, and the named tunnel and Cloudflare Access of the custom domain URL (6-3)."""

import json
import os
import re
import time
from urllib.parse import urlsplit

from . import filesystem, paths
from .messages import t


CLOUDFLARED_VERSION = "2026.10.0"
# cloudflared's link to Cloudflare. Its default, QUIC, cannot get the UDP buffers it wants in a
# Colab container ("failed to sufficiently increase receive buffer size"); HTTP/2 runs over TCP.
TUNNEL_PROTOCOL = "http2"
CLOUDFLARED_SHA256 = "d33ff2d14475178d2012c2c56beba87389ac5ded27649519f198a7d3134a99db"


def install_cloudflared():
    """Download the pinned release and refuse to run anything else."""
    import requests
    binary = paths.PRIVATE / "cloudflared"
    partial = paths.PRIVATE / "cloudflared.part"
    url = ("https://github.com/cloudflare/cloudflared/releases/download/"
           + CLOUDFLARED_VERSION + "/cloudflared-linux-amd64")
    try:
        with requests.get(url, stream=True, timeout=60) as res:
            if not res.ok:
                raise RuntimeError(t("cloudflared.download_failed"))
            with partial.open("wb") as f:
                for chunk in res.iter_content(1024 * 1024):
                    f.write(chunk)
        if filesystem.file_hash(partial) != CLOUDFLARED_SHA256:
            raise RuntimeError(t("cloudflared.sha256_mismatch"))
        partial.chmod(0o700)
        os.replace(partial, binary)
    finally:
        partial.unlink(missing_ok=True)


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
