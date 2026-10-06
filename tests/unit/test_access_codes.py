"""Anonymous access codes: signing, device binding, quotas, caps and owner alerts."""

import sys
import time
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[2] / "ui"))

from gateway.alerts import Alerter
from gateway.app import AccessCodes, create_app
from gateway.auth import AuthSettings
from gateway.codes import CodeSigner, CodeStore, InvalidCodeError, main

SECRET = "s" * 40
CSRF = {"X-Requested-With": "investing-engine"}


class Upstream(httpx.AsyncBaseTransport):
    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.status = 200
        self.headers: dict[str, str] = {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.requests.append(request)
        if request.url.path == "/analyses/stream":
            return httpx.Response(
                self.status, content=b'data: {"type": "final"}\n\n', headers=self.headers
            )
        return httpx.Response(200, json=[])


class RecordingAlerter(Alerter):
    def __init__(self, store):
        super().__init__(store, None)
        self.sent: list[str] = []

    async def alert(self, kind, message, *, key=None):
        if self._store.claim_alert(key or kind, every=3600):
            self.sent.append(kind)


@pytest.fixture
def signer():
    return CodeSigner(SECRET)


@pytest.fixture
def store(tmp_path):
    s = CodeStore(tmp_path / "gw.db")
    s.init()
    return s


@pytest.fixture
def setup(tmp_path, signer, store):
    alerter = RecordingAlerter(store)
    codes = AccessCodes(signer, store, alerter, global_daily_analyses=5, secure_cookies=False)
    upstream = Upstream()
    (tmp_path / "index.html").write_text("<!doctype html>")
    app = create_app(
        AuthSettings(provider="accesscode"),
        api_base_url="http://api",
        internal_token="t" * 48,
        static_dir=tmp_path,
        transport=upstream,
        codes=codes,
    )
    return app, upstream, alerter


def redeem(client, code):
    return client.post("/auth/code", json={"code": code}, headers=CSRF)


# --- Codes ------------------------------------------------------------------------------


def test_codes_round_trip_and_reject_tampering(signer):
    code, info = signer.create(days=1, quota=4)
    assert signer.parse(code) == info
    tampered = code[:-2] + ("A" if code[-2] != "A" else "B") + code[-1]
    with pytest.raises(InvalidCodeError):
        signer.parse(tampered)
    with pytest.raises(InvalidCodeError):
        CodeSigner("x" * 40).parse(code)  # signed with a different secret


def test_expired_codes_are_rejected(signer):
    code, info = signer.create(days=1, quota=1)
    with pytest.raises(InvalidCodeError):
        signer.parse(code, now=info.expires_at + 1)


def test_short_secrets_are_refused():
    with pytest.raises(RuntimeError, match="32"):
        CodeSigner("short")


# --- Gateway flow ---------------------------------------------------------------------


def test_redeem_then_use_the_api_as_an_anonymous_identity(setup, signer):
    app, upstream, _ = setup
    code, info = signer.create(days=1, quota=3)
    with TestClient(app) as client:
        assert client.get("/api/me").json()["method"] == "code"
        me = redeem(client, code).json()
        client.get("/api/instruments")

    assert me["code"]["id"] == info.short_id
    assert me["features"]["documents"] is False
    assert "Access code" in me["user"]
    assert upstream.requests[-1].headers["x-user-id"] == f"code:{info.code_id}"


def test_code_is_bound_to_the_first_device(setup, signer):
    app, _, alerter = setup
    code, _ = signer.create(days=1, quota=3)
    with TestClient(app) as first, TestClient(app) as second:
        assert redeem(first, code).status_code == 200
        refused = redeem(second, code)
        assert redeem(first, code).status_code == 200  # same browser may enter it again

    assert refused.status_code == 403
    assert alerter.sent == ["code used on a second device"]


def test_quota_per_code_is_enforced(setup, signer):
    app, upstream, _ = setup
    code, _ = signer.create(days=1, quota=2)
    with TestClient(app) as client:
        redeem(client, code)
        statuses = [
            client.post(
                "/api/analyses/stream", json={"symbols": ["XU100"]}, headers=CSRF
            ).status_code
            for _ in range(3)
        ]
    assert statuses == [200, 200, 429]
    assert len([r for r in upstream.requests if r.url.path == "/analyses/stream"]) == 2


def test_daily_cap_for_the_deployment(setup, signer):
    app, _, alerter = setup  # cap is 5
    with TestClient(app) as client:
        results = []
        for _ in range(3):
            code, _ = signer.create(days=1, quota=10)
            client.cookies.clear()
            redeem(client, code)
            for _ in range(2):
                results.append(
                    client.post(
                        "/api/analyses/stream", json={"symbols": ["XU100"]}, headers=CSRF
                    ).status_code
                )
    assert results == [200, 200, 200, 200, 200, 429]
    assert "daily limit almost reached" in alerter.sent
    assert "daily limit reached" in alerter.sent


def test_document_upload_is_off_for_anonymous_users(setup, signer):
    app, upstream, _ = setup
    code, _ = signer.create(days=1, quota=1)
    with TestClient(app) as client:
        redeem(client, code)
        response = client.post(
            "/api/documents", data={"symbol": "THYAO"}, files={"file": ("a.pdf", b"%PDF")},
            headers=CSRF,
        )  # fmt: skip
    assert response.status_code == 403
    assert all(r.url.path != "/documents" for r in upstream.requests)


def test_revoked_code_stops_working_immediately(setup, signer, store):
    app, _, _ = setup
    code, info = signer.create(days=1, quota=3)
    with TestClient(app) as client:
        redeem(client, code)
        assert client.get("/api/instruments").status_code == 200
        store.revoke(info.code_id)
        assert client.get("/api/instruments").status_code == 403
        assert redeem(client, code).status_code == 400


def test_many_invalid_codes_raise_one_alert(setup):
    app, _, alerter = setup
    with TestClient(app) as client:
        for _ in range(25):
            assert redeem(client, "IE-not-a-real-code").status_code == 400
    assert alerter.sent == ["many invalid codes"]


def test_burst_use_of_one_code_raises_an_alert(setup, signer):
    app, _, alerter = setup
    code, _ = signer.create(days=1, quota=5)
    with TestClient(app) as client:
        redeem(client, code)
        for _ in range(4):
            client.post("/api/analyses/stream", json={"symbols": ["XU100"]}, headers=CSRF)
    assert "fast use of one code" in alerter.sent


def test_provider_rate_limit_raises_an_alert(setup, signer):
    app, upstream, alerter = setup
    upstream.status = 429
    upstream.headers = {"x-limit-reason": "provider"}
    code, _ = signer.create(days=1, quota=5)
    with TestClient(app) as client:
        redeem(client, code)
        client.post("/api/analyses/stream", json={"symbols": ["XU100"]}, headers=CSRF)
    assert alerter.sent == ["model provider rate limit"]


def test_alerts_and_digest_hold_no_personal_data(store):
    alerter = Alerter(store, None)
    store.record_analysis("a" * 16)
    text = alerter.digest_text(now=time.time())
    assert "Analyses started today: 1" in text
    assert "@" not in text


# --- Command line ---------------------------------------------------------------------


def test_cli_creates_lists_and_revokes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ACCESS_CODE_SECRET", SECRET)
    monkeypatch.setenv("GATEWAY_DB_PATH", str(tmp_path / "cli.db"))
    assert main(["create", "--days", "2", "--quota", "3", "--count", "2"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    code = lines[0].split()[0]

    assert main(["revoke", code]) == 0
    assert "revoked" in capsys.readouterr().out
    store = CodeStore(tmp_path / "cli.db")
    assert store.is_revoked(CodeSigner(SECRET).parse(code).code_id)
