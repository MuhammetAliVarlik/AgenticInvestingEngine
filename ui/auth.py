"""Sign-in and authorisation for the Streamlit UI.

``AUTH_PROVIDER`` selects how the user is identified:

* ``easyauth`` - Azure Container Apps built-in authentication (GitHub,
  Google, Microsoft) runs in front of the app and injects the signed-in
  user into ``X-MS-CLIENT-PRINCIPAL-*`` headers. The platform strips any
  client-supplied copies of these headers, so they can be trusted.
* ``oidc`` - Streamlit's native OpenID Connect login (``st.login``), e.g.
  Google, configured in ``.streamlit/secrets.toml``. Used on hosts without
  a platform login, such as Hugging Face Spaces.
* ``none`` - local development only.

Every signed-in identity must also appear in ``ALLOWED_USERS``.
"""

from __future__ import annotations

import os
import re

import streamlit as st

_IDENTITY = re.compile(r"^[a-z0-9._%+@:\-]{1,254}$")


def _allowlist() -> frozenset[str]:
    raw = os.getenv("ALLOWED_USERS", "")
    return frozenset(x.strip().lower() for x in raw.split(",") if x.strip())


def _allowed(identity: str) -> bool:
    allowlist = _allowlist()
    if not allowlist:
        return os.getenv("AUTH_PROVIDER", "none") == "none"
    _, _, bare = identity.partition(":")
    return identity in allowlist or (bool(bare) and bare in allowlist)


def _easyauth_identity() -> str | None:
    headers = st.context.headers
    name = headers.get("X-Ms-Client-Principal-Name", "")
    provider = headers.get("X-Ms-Client-Principal-Idp", "")
    if not name:
        return None
    identity = name if "@" in name else f"{provider}:{name}"
    return identity.strip().lower()


def _deny(message: str) -> None:
    st.error(message)
    st.stop()


def require_user() -> str:
    """Return the signed-in, allowlisted identity or stop the page."""
    provider = os.getenv("AUTH_PROVIDER", "none")

    if provider == "none":
        return "local"

    if provider == "easyauth":
        identity = _easyauth_identity()
        if identity is None:
            _deny("Sign-in is required. Please reload the page.")
    elif provider == "oidc":
        if not st.user.is_logged_in:
            st.title("📈 Investing Engine")
            st.caption("Private demo - sign in with an invited account.")
            if st.button("Sign in with Google", type="primary"):
                st.login("google")
            st.stop()
        identity = str(st.user.get("email", "")).strip().lower()
    else:
        _deny("Authentication is misconfigured.")
        raise AssertionError("unreachable")

    if not identity or not _IDENTITY.fullmatch(identity):
        _deny("Your account could not be identified.")
    if not _allowed(identity):
        if provider == "oidc":
            st.button("Sign out", on_click=st.logout)
        _deny("This is a private demo and your account has not been invited.")
    return identity


def api_headers(identity: str) -> dict[str, str]:
    """Headers authenticating this UI to the internal API on the user's behalf."""
    headers = {"X-User-Id": identity}
    if token := os.getenv("INTERNAL_API_TOKEN"):
        headers["X-Internal-Token"] = token
    return headers


def sign_out_control() -> None:
    provider = os.getenv("AUTH_PROVIDER", "none")
    if provider == "oidc":
        st.sidebar.button("Sign out", on_click=st.logout)
    elif provider == "easyauth":
        st.sidebar.link_button("Sign out", "/.auth/logout")
