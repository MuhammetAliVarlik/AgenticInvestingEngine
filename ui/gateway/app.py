"""Web gateway: serves the React app and relays its calls to the internal API.

The browser only ever talks to this gateway. It signs the user in, checks
the allowlist and forwards a fixed set of API routes, adding the shared
internal token and the user's identity server-side, so the token never
reaches the browser. The API re-checks both (defence in depth).

CSRF: state-changing requests must carry the ``X-Requested-With`` header,
which a cross-site form or simple request cannot set, and the session
cookie is ``SameSite=Lax``.

With ``AUTH_PROVIDER=accesscode`` the gateway also enforces the anonymous
access-code rules: one device per code, a quota per code, a daily cap for
the whole deployment, no document uploads, and owner alerts on anomalies.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from gateway.alerts import Alerter, SmtpSettings, digest_hour_from_env
from gateway.auth import (
    AuthSettings,
    ForbiddenError,
    NotSignedInError,
    resolve_identity,
)
from gateway.codes import AccessCode, CodeSigner, CodeStore, InvalidCodeError, store_from_env

CSRF_HEADER = "x-requested-with"
DEFAULT_MAX_BODY_BYTES = 12 * 1024 * 1024
STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=900.0, write=120.0, pool=10.0)

# (method, path pattern) pairs the browser may reach; everything else is 404.
ALLOWED_ROUTES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (method, re.compile(pattern))
    for method, pattern in (
        ("GET", r"instruments"),
        ("GET", r"sources"),
        ("GET", r"usage"),
        ("GET", r"history/[A-Za-z0-9.]{1,16}"),
        ("GET", r"analyses/[A-Za-z0-9_\-]{1,64}/report\.pdf"),
        ("POST", r"datasets"),
        ("POST", r"documents"),
        ("POST", r"analyses/stream"),
    )
)
CODE_COOKIE = "ie_code"
DEVICE_COOKIE = "ie_device"
# Groq free tier: about 13 complete analyses a day (docs/OPERATIONS.md).
DEFAULT_GLOBAL_DAILY_ANALYSES = 12
BRUTE_FORCE_THRESHOLD = 20  # failed code entries per hour before an alert
BURST_SHARE = 0.8  # share of a code's quota used within one hour before an alert

FORWARDED_REQUEST_HEADERS = ("content-type", "accept")
FORWARDED_RESPONSE_HEADERS = (
    "content-type",
    "content-disposition",
    "cache-control",
    "retry-after",
)

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; "
        "form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


class SecurityHeadersMiddleware:
    """Pure ASGI middleware (streaming-safe) adding hardening headers."""

    def __init__(self, app: ASGIApp, *, hsts: bool) -> None:
        self.app = app
        self.hsts = hsts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers: list[Any] = list(message.get("headers", []))
                existing = {k.lower() for k, _ in headers}
                for name, value in SECURITY_HEADERS.items():
                    if name.lower().encode() not in existing:
                        headers.append((name.lower().encode(), value.encode()))
                if self.hsts:
                    headers.append(
                        (b"strict-transport-security", b"max-age=31536000; includeSubDomains")
                    )
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


def _route_allowed(method: str, path: str) -> bool:
    return any(method == m and pattern.fullmatch(path) for m, pattern in ALLOWED_ROUTES)


def _error(status: int, detail: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"detail": detail, **extra}, status_code=status)


class AccessCodes:
    """State and policy for the ``accesscode`` sign-in mode."""

    def __init__(
        self,
        signer: CodeSigner,
        store: CodeStore,
        alerter: Alerter,
        *,
        global_daily_analyses: int,
        secure_cookies: bool,
    ) -> None:
        self.signer = signer
        self.store = store
        self.alerter = alerter
        self.global_daily_analyses = global_daily_analyses
        self.secure_cookies = secure_cookies

    @classmethod
    def from_env(cls, *, production: bool) -> AccessCodes:
        store = store_from_env()
        return cls(
            CodeSigner(os.getenv("ACCESS_CODE_SECRET", "")),
            store,
            Alerter(store, SmtpSettings.from_env()),
            global_daily_analyses=int(
                os.getenv("GLOBAL_DAILY_ANALYSES", DEFAULT_GLOBAL_DAILY_ANALYSES)
            ),
            secure_cookies=production,
        )

    def identify(self, request: Request) -> AccessCode:
        """The valid code of this browser. Raises NotSignedIn/Forbidden errors."""
        raw = request.cookies.get(CODE_COOKIE, "")
        device = request.cookies.get(DEVICE_COOKIE, "")
        if not raw or not device:
            raise NotSignedInError
        try:
            code = self.signer.parse(raw)
        except InvalidCodeError:
            raise NotSignedInError from None
        if self.store.is_revoked(code.code_id) or not self.store.bind(code.code_id, device):
            raise ForbiddenError
        return code

    def describe(self, code: AccessCode) -> dict[str, Any]:
        return {
            "id": code.short_id,
            "expires_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(code.expires_at)),
            "quota": code.quota,
            "used": self.store.used(code.code_id),
        }


def create_app(
    auth: AuthSettings | None = None,
    *,
    api_base_url: str | None = None,
    internal_token: str | None = None,
    static_dir: Path | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    codes: AccessCodes | None = None,
) -> Starlette:
    auth = auth or AuthSettings.from_env()
    api_base_url = (api_base_url or os.getenv("API_BASE_URL", "http://localhost:8080")).rstrip("/")
    internal_token = (
        internal_token if internal_token is not None else os.getenv("INTERNAL_API_TOKEN", "")
    )
    static_dir = static_dir or Path(os.getenv("STATIC_DIR", Path(__file__).parents[1] / "dist"))
    max_body = int(os.getenv("MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES))
    production = os.getenv("ENVIRONMENT", "development") == "production"
    oauth = _oauth_client() if auth.provider == "oidc" else None
    if auth.provider == "accesscode" and codes is None:
        codes = AccessCodes.from_env(production=production)
    # Document text can hold personal data, so anonymous deployments do not accept it.
    documents_enabled = auth.provider != "accesscode"

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        digest: asyncio.Task[None] | None = None
        hour = digest_hour_from_env() if codes is not None else None
        if codes is not None and codes.alerter.enabled and hour is not None:
            digest = asyncio.create_task(codes.alerter.run_digest(hour))
        async with httpx.AsyncClient(
            base_url=api_base_url, timeout=STREAM_TIMEOUT, transport=transport
        ) as client:
            app.state.client = client
            try:
                yield
            finally:
                if digest is not None:
                    digest.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await digest

    def identify(request: Request) -> tuple[str, AccessCode | None]:
        if codes is not None:
            code = codes.identify(request)
            return code.identity, code
        return resolve_identity(request, auth), None

    def session_payload(identity: str, code: AccessCode | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "user": identity if code is None else f"Access code {code.short_id}",
            "provider": auth.provider,
            "logout_url": _logout_url(auth),
            "features": {"documents": documents_enabled},
        }
        if code is not None and codes is not None:
            payload["code"] = codes.describe(code)
        return payload

    # --- Session endpoints ------------------------------------------------------------

    async def me(request: Request) -> Response:
        try:
            identity, code = identify(request)
        except NotSignedInError:
            return _error(
                401, "Sign-in required", login_url=_login_url(auth), method=_sign_in_method(auth)
            )
        except ForbiddenError:
            detail = (
                "This access code is revoked or in use on another device"
                if codes is not None
                else "This account does not have access"
            )
            return _error(403, detail, logout_url=_logout_url(auth))
        return JSONResponse(session_payload(identity, code))

    async def redeem(request: Request) -> Response:
        """Accept an access code and bind it to this browser."""
        if codes is None:
            return _error(404, "Not found")
        if not request.headers.get(CSRF_HEADER):
            return _error(403, "Missing CSRF header")
        try:
            body = json.loads(await request.body() or b"{}")
            raw = str(body.get("code", ""))[:100]
        except (ValueError, AttributeError):
            raw = ""
        try:
            code = codes.signer.parse(raw)
            if codes.store.is_revoked(code.code_id):
                raise InvalidCodeError
        except InvalidCodeError:
            failures = codes.store.record_failure()
            if failures >= BRUTE_FORCE_THRESHOLD:
                await codes.alerter.alert(
                    "many invalid codes",
                    f"{failures} invalid or expired access codes were entered in the last "
                    "hour. This can be a guessing attempt. The codes are 128-bit random "
                    "values, so guessing is not practical, but look at the trend.",
                    key="bruteforce",
                )
            return _error(400, "This access code is not valid or has expired")

        device = request.cookies.get(DEVICE_COOKIE) or secrets.token_urlsafe(32)
        if not codes.store.bind(code.code_id, device):
            await codes.alerter.alert(
                "code used on a second device",
                f"Access code {code.short_id} was entered on a second device and was refused. "
                f"The code can be shared or leaked. To stop it, run: "
                f"python -m gateway.codes revoke {code.short_id}",
                key=f"device:{code.code_id}",
            )
            return _error(403, "This access code is already in use on another device")

        response = JSONResponse(session_payload(code.identity, code))
        response.set_cookie(
            CODE_COOKIE,
            raw.strip(),
            max_age=max(60, int(code.expires_at - time.time())),
            httponly=True,
            secure=codes.secure_cookies,
            samesite="strict",
        )
        response.set_cookie(
            DEVICE_COOKIE,
            device,
            max_age=365 * 86400,
            httponly=True,
            secure=codes.secure_cookies,
            samesite="strict",
        )
        return response

    async def login(request: Request) -> Response:
        if oauth is None:
            return RedirectResponse("/")
        redirect_uri = os.environ["OIDC_REDIRECT_URI"]
        return await oauth.authorize_redirect(request, redirect_uri)  # type: ignore[no-any-return]

    async def callback(request: Request) -> Response:
        if oauth is None:
            return RedirectResponse("/")
        try:
            token = await oauth.authorize_access_token(request)
        except Exception:  # Authlib raises several error types on a bad callback
            return RedirectResponse("/?signin=failed")
        claims = token.get("userinfo") or {}
        email = str(claims.get("email", "")).strip().lower()
        if not email or claims.get("email_verified") is False:
            return RedirectResponse("/?signin=failed")
        request.session.clear()
        request.session["user"] = email
        return RedirectResponse("/")

    async def logout(request: Request) -> Response:
        response = RedirectResponse("/")
        if codes is not None:
            # The device cookie stays, so this browser can enter the same code again.
            response.delete_cookie(CODE_COOKIE)
        else:
            request.session.clear()
        return response

    # --- API relay --------------------------------------------------------------------

    async def relay(request: Request) -> Response:
        path = request.path_params["path"]
        if not _route_allowed(request.method, path):
            return _error(404, "Not found")
        if request.method != "GET" and not request.headers.get(CSRF_HEADER):
            return _error(403, "Missing CSRF header")
        try:
            identity, code = identify(request)
        except NotSignedInError:
            return _error(401, "Sign-in required", login_url=_login_url(auth))
        except ForbiddenError:
            return _error(403, "This account does not have access")

        if path == "documents" and not documents_enabled:
            return _error(403, "Document upload is off in this deployment")
        is_analysis = path == "analyses/stream"
        if is_analysis and code is not None and codes is not None:
            if codes.store.used(code.code_id) >= code.quota:
                return _error(429, "This access code has used all its analyses")
            if codes.store.analyses_today() >= codes.global_daily_analyses:
                await codes.alerter.alert(
                    "daily limit reached",
                    f"The deployment reached its daily limit of {codes.global_daily_analyses} "
                    "analyses. New analyses are refused until 00:00 UTC.",
                    key=f"cap:{time.strftime('%Y-%m-%d', time.gmtime())}",
                )
                return _error(429, "The demo has reached its daily limit. Try again tomorrow")

        body = b""
        if request.method == "POST":
            declared = int(request.headers.get("content-length") or 0)
            if declared > max_body:
                return _error(413, "Upload is too large")
            chunks: list[bytes] = []
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > max_body:
                    return _error(413, "Upload is too large")
                chunks.append(chunk)
            body = b"".join(chunks)

        headers = {k: v for k in FORWARDED_REQUEST_HEADERS if (v := request.headers.get(k))}
        headers["x-user-id"] = identity
        if internal_token:
            headers["x-internal-token"] = internal_token

        client: httpx.AsyncClient = request.app.state.client
        upstream = client.build_request(
            request.method, f"/{path}", params=request.query_params, headers=headers, content=body
        )
        try:
            response = await client.send(upstream, stream=True)
        except httpx.HTTPError:
            return _error(502, "The analysis service is unavailable")

        if codes is not None:
            await _after_upstream(codes, code if is_analysis else None, response)

        return StreamingResponse(
            response.aiter_bytes(),  # decoded, since content-encoding is not forwarded
            status_code=response.status_code,
            headers={k: v for k in FORWARDED_RESPONSE_HEADERS if (v := response.headers.get(k))}
            | {"x-accel-buffering": "no"},
            background=BackgroundTask(response.aclose),
        )

    # --- Static app -------------------------------------------------------------------

    root = static_dir.resolve()
    public_files = {p.name for p in root.iterdir() if p.is_file()} if root.is_dir() else set()

    async def index(request: Request) -> Response:
        # Top-level build files (favicon etc.) are served as is; any other path
        # falls back to the single-page app.
        name = request.path_params.get("rest", "")
        if name in public_files and name != "index.html":
            return FileResponse(root / name)
        page = static_dir / "index.html"
        if not page.is_file():
            return _error(404, "The web app has not been built")
        return FileResponse(page, headers={"Cache-Control": "no-cache"})

    async def healthz(request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    routes = [
        Route("/healthz", healthz),
        Route("/api/me", me),
        Route("/auth/code", redeem, methods=["POST"]),
        Route("/api/{path:path}", relay, methods=["GET", "POST"]),
        Route("/auth/login", login),
        # Path kept stable so existing OAuth client redirect URIs remain valid.
        Route("/oauth2callback", callback),
        Route("/auth/logout", logout),
    ]
    assets = static_dir / "assets"
    if assets.is_dir():
        routes.append(Mount("/assets", StaticFiles(directory=assets), name="assets"))
    routes.append(Route("/{rest:path}", index))

    middleware = [Middleware(SecurityHeadersMiddleware, hsts=production)]
    if auth.provider == "oidc":
        middleware.append(
            Middleware(
                SessionMiddleware,
                secret_key=os.environ["OIDC_COOKIE_SECRET"],
                session_cookie="ie_session",
                same_site="lax",
                https_only=production,
                max_age=8 * 3600,
            )
        )
    return Starlette(routes=routes, middleware=middleware, lifespan=lifespan)


async def _after_upstream(
    codes: AccessCodes, code: AccessCode | None, response: httpx.Response
) -> None:
    """Count started analyses and raise owner alerts on unusual use."""
    if response.status_code == 429 and response.headers.get("x-limit-reason") == "provider":
        await codes.alerter.alert(
            "model provider rate limit",
            "The language model provider refused requests (rate limit). The API pauses "
            "new analyses for a short time. If this repeats, decrease GLOBAL_DAILY_ANALYSES.",
            key="provider",
        )
    if code is None or response.status_code != 200:
        return
    codes.store.record_analysis(code.code_id)
    recent = codes.store.used(code.code_id, since=time.time() - 3600)
    if code.quota >= 3 and recent >= max(3, int(code.quota * BURST_SHARE)):
        await codes.alerter.alert(
            "fast use of one code",
            f"Access code {code.short_id} started {recent} analyses in the last hour "
            f"(quota {code.quota}). If you did not expect this, run: "
            f"python -m gateway.codes revoke {code.short_id}",
            key=f"burst:{code.code_id}",
        )
    today = codes.store.analyses_today()
    if today >= int(codes.global_daily_analyses * 0.8):
        await codes.alerter.alert(
            "daily limit almost reached",
            f"{today} of {codes.global_daily_analyses} analyses for today are used.",
            key=f"cap80:{time.strftime('%Y-%m-%d', time.gmtime())}",
        )


def _sign_in_method(auth: AuthSettings) -> str:
    return {"accesscode": "code", "none": "none"}.get(auth.provider, "redirect")


def _login_url(auth: AuthSettings) -> str | None:
    return {"oidc": "/auth/login", "easyauth": "/.auth/login/github"}.get(auth.provider)


def _logout_url(auth: AuthSettings) -> str | None:
    return {
        "oidc": "/auth/logout",
        "easyauth": "/.auth/logout",
        "accesscode": "/auth/logout",
    }.get(auth.provider)


def _oauth_client() -> Any:
    from authlib.integrations.starlette_client import OAuth

    for name in ("OIDC_CLIENT_ID", "OIDC_CLIENT_SECRET", "OIDC_COOKIE_SECRET", "OIDC_REDIRECT_URI"):
        if not os.getenv(name):
            raise RuntimeError(f"{name} must be set when AUTH_PROVIDER=oidc")
    oauth = OAuth()
    return oauth.register(
        "oidc",
        client_id=os.environ["OIDC_CLIENT_ID"],
        client_secret=os.environ["OIDC_CLIENT_SECRET"],
        server_metadata_url=os.getenv(
            "OIDC_METADATA_URL", "https://accounts.google.com/.well-known/openid-configuration"
        ),
        client_kwargs={"scope": "openid email profile"},
    )
