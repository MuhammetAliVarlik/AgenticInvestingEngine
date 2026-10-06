"""Sign-in and authorisation for the web gateway.

``AUTH_PROVIDER`` selects how the user is identified:

* ``easyauth`` - Azure Container Apps built-in authentication (GitHub,
  Google, Microsoft) runs in front of the gateway and injects the signed-in
  user into ``X-MS-CLIENT-PRINCIPAL-*`` headers. The platform strips any
  client-supplied copies of these headers, so they can be trusted.
* ``oidc`` - the gateway runs an OpenID Connect login itself (e.g. Google)
  and keeps the verified e-mail address in a signed session cookie. Used on
  hosts without a platform login, such as Hugging Face Spaces.
* ``accesscode`` - anonymous, time-limited access codes made by the owner
  (see :mod:`gateway.codes`). No account and no personal data; the gateway
  identifies the caller as ``code:<random id>``.
* ``none`` - local development only.

For ``easyauth`` and ``oidc``, every signed-in identity must also appear in
``ALLOWED_USERS``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from starlette.requests import Request

PROVIDERS = frozenset({"none", "easyauth", "oidc", "accesscode"})
ACCOUNT_PROVIDERS = frozenset({"easyauth", "oidc"})
LOCAL_IDENTITY = "local"
_IDENTITY = re.compile(r"^[a-z0-9._%+@:\-]{1,254}$")


class NotSignedInError(Exception):
    """The request carries no signed-in identity."""


class ForbiddenError(Exception):
    """The identity is valid but not invited."""


@dataclass(frozen=True)
class AuthSettings:
    provider: str = "none"
    allowed_users: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_env(cls) -> AuthSettings:
        provider = os.getenv("AUTH_PROVIDER", "none").strip().lower()
        if provider not in PROVIDERS:
            raise RuntimeError(f"Unsupported AUTH_PROVIDER: {provider!r}")
        raw = os.getenv("ALLOWED_USERS", "")
        allowed = frozenset(x.strip().lower() for x in raw.split(",") if x.strip())
        if provider in ACCOUNT_PROVIDERS and not allowed:
            raise RuntimeError("ALLOWED_USERS must list at least one user when sign-in is on")
        return cls(provider=provider, allowed_users=allowed)


def easyauth_identity(request: Request) -> str | None:
    name = request.headers.get("x-ms-client-principal-name", "")
    provider = request.headers.get("x-ms-client-principal-idp", "")
    if not name:
        return None
    identity = name if "@" in name else f"{provider}:{name}"
    return identity.strip().lower()


def is_allowed(identity: str, allowed: frozenset[str]) -> bool:
    """Match ``provider:name`` identities against both qualified and bare entries."""
    if identity in allowed:
        return True
    _, _, bare = identity.partition(":")
    return bool(bare) and bare in allowed


def resolve_identity(request: Request, settings: AuthSettings) -> str:
    """Return the signed-in, allowlisted identity for this request.

    Raises:
        NotSignedInError: no identity is present.
        ForbiddenError: the identity is malformed or not on the allowlist.
    """
    if settings.provider == "none":
        return LOCAL_IDENTITY

    if settings.provider == "easyauth":
        identity = easyauth_identity(request)
    else:
        identity = request.session.get("user")

    if not identity:
        raise NotSignedInError
    if not _IDENTITY.fullmatch(identity) or not is_allowed(identity, settings.allowed_users):
        raise ForbiddenError
    return identity
