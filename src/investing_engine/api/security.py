"""API authentication, authorisation and HTTP hardening.

Deployment model: the API is never exposed to browsers. Only the UI service
(or another trusted backend) calls it, over a private network, presenting a
shared internal token and the end user's identity, which the UI obtained
from the platform's login (Azure Easy Auth or OIDC). The API then applies
the user allowlist itself as well - defence in depth if the UI is
misconfigured.

Because the API authenticates with a header token rather than cookies, a
third-party website cannot make a victim's browser send authenticated
requests to it, so classic CSRF does not apply here. Browser-facing CSRF
protection is provided by the web gateway (custom-header check, SameSite cookies).
"""

from __future__ import annotations

import re
import secrets
import threading
import time
from collections import deque
from typing import Any

from fastapi import HTTPException, Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from investing_engine.config import Settings

LOCAL_PRINCIPAL = "local"
_IDENTITY = re.compile(r"^[a-z0-9._%+@:\-]{1,254}$")

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    # The API serves JSON, SSE and PDFs only; the interactive docs (dev only)
    # need inline scripts and the jsDelivr CDN.
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
}
_DOCS_CSP = (
    "default-src 'none'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: "
    "https://fastapi.tiangolo.com; connect-src 'self'; frame-ancestors 'none'"
)


class ConfigurationError(RuntimeError):
    pass


def normalise_identity(raw: str) -> str:
    """Lower-case and validate a user identity such as ``a@b.com`` or ``github:octocat``."""
    identity = raw.strip().lower()
    if not _IDENTITY.fullmatch(identity):
        raise ValueError("Malformed identity")
    return identity


# Allowlist entry that admits every anonymous access-code identity ("code:<id>").
# The web gateway validates the codes; the API only accepts them from the gateway,
# which must present the internal token.
ANY_ACCESS_CODE = "code:*"


def parse_allowlist(raw: str) -> frozenset[str]:
    entries = [x.strip().lower() for x in raw.split(",") if x.strip()]
    return frozenset(x if x == ANY_ACCESS_CODE else normalise_identity(x) for x in entries)


def is_allowed(identity: str, allowlist: frozenset[str]) -> bool:
    """Match ``provider:name`` identities against both qualified and bare entries."""
    if identity in allowlist:
        return True
    if ANY_ACCESS_CODE in allowlist and identity.startswith("code:"):
        return True
    _, _, bare = identity.partition(":")
    return bool(bare) and bare in allowlist


def validate_security_settings(settings: Settings) -> None:
    """Refuse to start a production deployment with an unsafe configuration."""
    if settings.environment != "production":
        return
    problems = []
    if settings.auth_mode != "trusted-proxy":
        problems.append("AUTH_MODE must be 'trusted-proxy'")
    token = settings.internal_api_token.get_secret_value() if settings.internal_api_token else ""
    if len(token) < 32:
        problems.append("INTERNAL_API_TOKEN must be at least 32 characters")
    if not settings.allowed_users.strip():
        problems.append("ALLOWED_USERS must list at least one user")
    if settings.telemetry_salt.get_secret_value() == "local-development-salt":
        problems.append("TELEMETRY_SALT must be set to a random value")
    if settings.enable_yfinance:
        problems.append("ENABLE_YFINANCE must be false (personal-use data source)")
    if problems:
        raise ConfigurationError("Unsafe production configuration: " + "; ".join(problems))


class Authenticator:
    """FastAPI dependency resolving the caller's identity."""

    def __init__(self, settings: Settings) -> None:
        self._mode = settings.auth_mode
        self._token = (
            settings.internal_api_token.get_secret_value().encode()
            if settings.internal_api_token
            else b""
        )
        self._allowlist = parse_allowlist(settings.allowed_users)

    def __call__(self, request: Request) -> str:
        if self._mode == "none":
            return LOCAL_PRINCIPAL

        presented = request.headers.get("x-internal-token", "").encode()
        if not self._token or not secrets.compare_digest(presented, self._token):
            raise HTTPException(status_code=401, detail="Unauthorised")
        try:
            identity = normalise_identity(request.headers.get("x-user-id", ""))
        except ValueError:
            raise HTTPException(status_code=401, detail="Unauthorised") from None
        if self._allowlist and not is_allowed(identity, self._allowlist):
            raise HTTPException(status_code=403, detail="Forbidden")
        return identity


class RequestRateLimiter:
    """Sliding-window limit on requests per principal (in-memory, per replica)."""

    def __init__(self, *, per_minute: int) -> None:
        self._limit = per_minute
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, principal: str) -> None:
        now = time.monotonic()
        with self._lock:
            window = self._hits.setdefault(principal, deque())
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= self._limit:
                retry_after = int(60 - (now - window[0])) + 1
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests",
                    headers={"Retry-After": str(retry_after)},
                )
            window.append(now)


class SecurityHeadersMiddleware:
    """Pure ASGI middleware (streaming-safe) adding hardening headers."""

    def __init__(self, app: ASGIApp, *, hsts: bool) -> None:
        self.app = app
        self.hsts = hsts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        is_docs = scope["path"] in ("/docs", "/openapi.json", "/docs/oauth2-redirect")

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers: list[Any] = list(message.get("headers", []))
                existing = {k.lower() for k, _ in headers}
                for name, value in SECURITY_HEADERS.items():
                    if name == "Content-Security-Policy" and is_docs:
                        value = _DOCS_CSP
                    if name.lower().encode() not in existing:
                        headers.append((name.lower().encode(), value.encode()))
                if self.hsts:
                    headers.append(
                        (b"strict-transport-security", b"max-age=31536000; includeSubDomains")
                    )
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)
