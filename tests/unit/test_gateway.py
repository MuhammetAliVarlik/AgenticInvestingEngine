"""Web gateway: sign-in, allowlist, CSRF, route allowlist and relaying to the API."""

import sys
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[2] / "ui"))

from gateway.app import create_app
from gateway.auth import AuthSettings

TOKEN = "t" * 48
CSRF = {"X-Requested-With": "investing-engine"}
EASYAUTH = {"X-MS-CLIENT-PRINCIPAL-NAME": "octocat", "X-MS-CLIENT-PRINCIPAL-IDP": "github"}


class Upstream(httpx.AsyncBaseTransport):
    """Records what the gateway sends to the API and answers like it.

    Unlike ``httpx.MockTransport`` it leaves response bodies unread, so the
    gateway can stream them exactly as it does against the real API.
    """

    def __init__(self):
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.requests.append(request)
        if request.url.path == "/analyses/stream":
            body = b'data: {"type": "status", "text": "Consulting news_analyst"}\n\n'
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
        if request.url.path == "/datasets":
            return httpx.Response(201, json={"dataset_id": "d1", "rows": 60})
        return httpx.Response(200, json=[{"symbol": "XU100"}], headers={"set-cookie": "leak=1"})


@pytest.fixture
def static_dir(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app.js").write_text("console.log(1)")
    (tmp_path / "index.html").write_text("<!doctype html><title>Investing Engine</title>")
    (tmp_path / "favicon.svg").write_text("<svg/>")
    return tmp_path


def make_client(static_dir, auth=None, **kwargs):
    upstream = Upstream()
    app = create_app(
        auth or AuthSettings(),
        api_base_url="http://api",
        internal_token=TOKEN,
        static_dir=static_dir,
        transport=upstream,
        **kwargs,
    )
    return TestClient(app), upstream


def test_local_mode_relays_with_internal_token_and_identity(static_dir):
    client, upstream = make_client(static_dir)
    with client:
        assert client.get("/api/me").json()["user"] == "local"
        response = client.get("/api/instruments")

    assert response.status_code == 200
    assert response.json() == [{"symbol": "XU100"}]
    sent = upstream.requests[0]
    assert sent.url.path == "/instruments"
    assert sent.headers["x-internal-token"] == TOKEN
    assert sent.headers["x-user-id"] == "local"
    assert "set-cookie" not in response.headers  # upstream headers are allowlisted


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/docs"),
        ("GET", "/api/openapi.json"),
        ("POST", "/api/analyses"),
        ("GET", "/api/datasets"),
    ],
)
def test_routes_outside_the_allowlist_are_not_relayed(static_dir, method, path):
    client, upstream = make_client(static_dir)
    with client:
        response = client.request(method, path, headers=CSRF)
    assert response.status_code == 404
    assert upstream.requests == []


def test_state_changing_requests_need_the_csrf_header(static_dir):
    client, upstream = make_client(static_dir)
    with client:
        rejected = client.post(
            "/api/datasets", files={"file": ("a.csv", b"x")}, data={"symbol": "THYAO"}
        )
        accepted = client.post(
            "/api/datasets", files={"file": ("a.csv", b"x")}, data={"symbol": "THYAO"}, headers=CSRF
        )
    assert rejected.status_code == 403
    assert accepted.status_code == 201
    assert len(upstream.requests) == 1
    assert b'name="symbol"' in upstream.requests[0].content


def test_oversized_uploads_are_refused_before_reaching_the_api(static_dir, monkeypatch):
    monkeypatch.setenv("MAX_BODY_BYTES", "1024")
    client, upstream = make_client(static_dir)
    with client:
        response = client.post(
            "/api/documents",
            files={"file": ("a.pdf", b"x" * 4096)},
            data={"symbol": "THYAO"},
            headers=CSRF,
        )
    assert response.status_code == 413
    assert upstream.requests == []


def test_analysis_events_are_streamed_through(static_dir):
    client, _ = make_client(static_dir)
    with client:
        response = client.post("/api/analyses/stream", json={"symbols": ["XU100"]}, headers=CSRF)
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "Consulting news_analyst" in response.text


def test_easyauth_requires_platform_identity_on_the_allowlist(static_dir):
    auth = AuthSettings(provider="easyauth", allowed_users=frozenset({"octocat"}))
    client, upstream = make_client(static_dir, auth)
    with client:
        anonymous = client.get("/api/instruments")
        stranger = client.get(
            "/api/instruments",
            headers={
                "X-MS-CLIENT-PRINCIPAL-NAME": "mallory",
                "X-MS-CLIENT-PRINCIPAL-IDP": "github",
            },
        )
        invited = client.get("/api/instruments", headers=EASYAUTH)
        me = client.get("/api/me", headers=EASYAUTH).json()

    assert anonymous.status_code == 401
    assert anonymous.json()["login_url"] == "/.auth/login/github"
    assert stranger.status_code == 403
    assert invited.status_code == 200
    assert upstream.requests[-1].headers["x-user-id"] == "github:octocat"
    assert me == {
        "user": "github:octocat",
        "provider": "easyauth",
        "logout_url": "/.auth/logout",
        "features": {"documents": True},
    }


def test_spoofed_identity_header_is_overwritten(static_dir):
    auth = AuthSettings(provider="easyauth", allowed_users=frozenset({"octocat"}))
    client, upstream = make_client(static_dir, auth)
    with client:
        client.get("/api/usage", headers={**EASYAUTH, "X-User-Id": "admin@example.com"})
    assert upstream.requests[0].headers["x-user-id"] == "github:octocat"


def test_oidc_mode_sends_unauthenticated_users_to_sign_in(static_dir, monkeypatch):
    for name, value in {
        "OIDC_CLIENT_ID": "id",
        "OIDC_CLIENT_SECRET": "secret",
        "OIDC_COOKIE_SECRET": "c" * 32,
        "OIDC_REDIRECT_URI": "https://example.test/oauth2callback",
    }.items():
        monkeypatch.setenv(name, value)
    auth = AuthSettings(provider="oidc", allowed_users=frozenset({"alice@example.com"}))
    client, upstream = make_client(static_dir, auth)
    with client:
        me = client.get("/api/me")
        relayed = client.get("/api/instruments")
    assert me.status_code == 401
    assert me.json()["login_url"] == "/auth/login"
    assert relayed.status_code == 401
    assert upstream.requests == []


def test_sign_in_is_mandatory_with_an_allowlist(monkeypatch):
    monkeypatch.setenv("AUTH_PROVIDER", "easyauth")
    monkeypatch.delenv("ALLOWED_USERS", raising=False)
    with pytest.raises(RuntimeError, match="ALLOWED_USERS"):
        AuthSettings.from_env()


def test_serves_the_single_page_app_with_security_headers(static_dir):
    client, _ = make_client(static_dir)
    with client:
        page = client.get("/some/deep/link")
        asset = client.get("/assets/app.js")
        icon = client.get("/favicon.svg")
        traversal = client.get("/..%2F..%2Fetc%2Fpasswd")

    assert "Investing Engine" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert page.headers["x-frame-options"] == "DENY"
    assert asset.text == "console.log(1)"
    assert icon.text == "<svg/>"
    assert "root:" not in traversal.text
