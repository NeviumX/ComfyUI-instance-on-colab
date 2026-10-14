"""The proxy in front of ComfyUI: the only way in from outside the machine."""

import asyncio
import hmac
import json
import re
import time
from urllib.parse import urlsplit, parse_qs

from . import cloudflare, pairing, stop_button
from .messages import t


COOKIE = "__Host-comfy_colab"


COOKIE_SECONDS = 24 * 3600
ACCESS_KEYS_REFRESH = 30


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
                async with session.get(cloudflare.access_certs_url(access["team"])) as res:
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

    noted = [0.0, 0]  # When the current minute of the log began, and the lines written in it.

    def note(message):
        """A line in proxy.log, but no more of them than a visitor's requests may fill it with."""
        if time.monotonic() - noted[0] >= 60:
            noted[:] = [time.monotonic(), 0]
        noted[1] += 1
        if noted[1] <= 30:
            print(message, flush=True)

    def refuse_other_sites(request, quiet=False):
        """Stop what another site sends through a signed-in browser. quiet: the sender has not
        signed in, so it could be anyone, and gets no line in the log."""
        origin = request.headers.get("Origin")
        if origin and origin not in {"https://" + request.host, "http://" + request.host} | origins:
            if not quiet:
                note(t("proxy.origin_rejected", origin=origin[:200], host=request.host[:200]))
            raise web.HTTPForbidden(text=t("proxy.cross_origin"))
        # A link, an image or a frame on another site carries no Origin, but the browser says where
        # it comes from. From elsewhere only opening the page itself is taken, not ComfyUI's addresses:
        # ComfyUI refuses those itself only in its recent versions.
        site = request.headers.get("Sec-Fetch-Site", "").lower()
        if site in ("cross-site", "same-site"):
            opening = (request.method == "GET" and request.path in ("/", "/index.html")
                       and request.headers.get("Sec-Fetch-Mode", "").lower() == "navigate"
                       and request.headers.get("Sec-Fetch-Dest", "").lower() == "document")
            if not opening:
                if not quiet:
                    note(t("proxy.site_rejected", method=request.method[:16], path=request.path[:200], site=site))
                raise web.HTTPForbidden(text=t("proxy.cross_origin"))
        return origin

    def passed_on(res, keep_length):
        """ComfyUI's response headers for the browser. A custom node must not replace the cookie
        that signs the browser in, and no other site may show these pages in a frame."""
        dropped = hop - {"content-length"} if keep_length else hop
        headers = [(k, v) for k, v in res.headers.items() if k.lower() not in dropped and not (
            k.lower() == "set-cookie" and (v.split("=", 1)[0].strip() in {COOKIE, "comfy_colab"}
                                           or v.lstrip().startswith("CF_")))]
        headers.append(("Content-Security-Policy", "frame-ancestors 'self'"))
        if "X-Frame-Options" not in res.headers:
            headers.append(("X-Frame-Options", "SAMEORIGIN"))
        return headers

    def pairing_page(notice="", status=200):
        page = pairing.PAIRING_PAGE.format(path=pairing.PAIRING_PATH, notice=html.escape(notice), title=html.escape(t("pairing.title")),
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
        refuse_other_sites(request, quiet=True)
        length = request.content_length
        if length is None or length > 256:  # Nothing unauthenticated gets to send more.
            raise web.HTTPBadRequest()
        try:
            form = parse_qs((await request.content.readexactly(length)).decode(errors="replace"))
        except asyncio.IncompleteReadError:
            raise web.HTTPBadRequest() from None
        accepted, remaining = pairing.redeem_pairing_code(form.get("code", [""])[0])
        if not accepted:
            return pairing_page(t("pairing.wrong", remaining=remaining) if remaining else t("pairing.closed"), 401)
        response = web.HTTPSeeOther(location="/")
        response.set_cookie(cookie_name, cookie_secret, max_age=COOKIE_SECONDS,
                            secure=secure_cookie, httponly=True, samesite="Lax", path="/")
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        raise response

    async def stop_endpoint(request):
        """The stop button's request, and afterwards its questions about how that is going."""
        if request.method == "POST":
            length = request.content_length
            if length is None or length > 256:
                raise web.HTTPBadRequest()
            try:
                delete_runtime = json.loads(await request.content.readexactly(length))["delete_runtime"]
            except (asyncio.IncompleteReadError, ValueError, KeyError, TypeError):
                raise web.HTTPBadRequest() from None
            if not isinstance(delete_runtime, bool):
                raise web.HTTPBadRequest()
            stop_button.request_stop(delete_runtime)
        return web.json_response({"state": stop_button.stop_request_state()}, status=202 if request.method == "POST" else 200,
                                 headers={"Cache-Control": "no-store"})

    async def handler(request):
        if access:
            if not await signed_by_access(request):
                raise web.HTTPUnauthorized(text=t("proxy.access_required"))
        elif not forwarded and not matches(request.cookies.get(cookie_name, ""), cookie_secret):
            if request.path == pairing.PAIRING_PATH:
                return await pair(request)
            if request.method == "GET" and "text/html" in request.headers.get("Accept", ""):
                # A browser opening the address: take it to where the code goes.
                raise web.HTTPSeeOther(location=pairing.PAIRING_PATH, headers={"Cache-Control": "no-store"})
            raise web.HTTPUnauthorized(text=t("proxy.unauthorized"))
        if request.path == pairing.PAIRING_PATH:
            raise web.HTTPSeeOther(location="/")
        origin = refuse_other_sites(request)
        if request.path == stop_button.STOP_SCRIPT_PATH:
            return web.Response(text=stop_button.stop_button_script(), content_type="application/javascript", charset="utf-8",
                                headers={"Cache-Control": "no-store"})
        if request.path == stop_button.STOP_PATH:
            return await stop_endpoint(request)
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
        page = request.method == "GET" and request.path in ("/", "/index.html")
        if page:
            # Asked for uncompressed: the stop button goes into this page before it is passed on.
            headers = {k: v for k, v in headers.items() if k.lower() != "accept-encoding"}
            headers["Accept-Encoding"] = "identity"
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
                                note(t("proxy.websocket_error") + " " + repr(source.exception()))
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
                page = (page and res.status == 200 and "Content-Encoding" not in res.headers
                        and res.headers.get("Content-Type", "").lower().startswith("text/html"))
                if page:
                    response = web.Response(body=stop_button.with_stop_button(await res.read()), headers=passed_on(res, False))
                else:
                    response = web.StreamResponse(status=res.status, headers=passed_on(res, True))
                response.headers["Referrer-Policy"] = "no-referrer"
                complete = request.method == "GET" and res.status == 200
                # ComfyUI marks the pictures it ships (template thumbnails, icons) as good for a day.
                # The browser may keep those, so they are not fetched again on every look; nothing in
                # between may. Generated files come from /api and /view addresses and are never kept.
                kept = re.search(r"max-age=\d+", res.headers.get("Cache-Control", ""))
                if complete and hashed_asset.fullmatch(request.path):
                    response.headers["Cache-Control"] = "private, max-age=31536000, immutable"
                elif complete and kept and "." in request.path.rsplit("/", 1)[-1] and not request.path.startswith("/api/"):
                    response.headers["Cache-Control"] = "private, " + kept.group(0)
                else:
                    response.headers["Cache-Control"] = "no-store"
                # ComfyUI sends its 26 MB frontend uncompressed and the tunnel is the slow hop.
                if (complete and "Content-Encoding" not in res.headers
                        and res.headers.get("Content-Type", "").lower().startswith(compressible)
                        and "gzip" in request.headers.get("Accept-Encoding", "").lower()):
                    response.headers["Vary"] = "Accept-Encoding"
                    response.enable_compression(web.ContentCoding.gzip)
                if page:
                    return response
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
