import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from investing_engine.api.app import create_app
from investing_engine.api.security import (
    ConfigurationError,
    is_allowed,
    normalise_identity,
    parse_allowlist,
    validate_security_settings,
)
from investing_engine.services import MarketData
from tests.factories import synthetic_prices
from tests.unit.test_services import StubEvds, StubGdelt

TOKEN = "t" * 40


def _headers(user: str = "alice@example.com", token: str = TOKEN) -> dict[str, str]:
    return {"X-Internal-Token": token, "X-User-Id": user}


@pytest.fixture
def secured(settings):
    configured = settings.model_copy(
        update={
            "auth_mode": "trusted-proxy",
            "internal_api_token": SecretStr(TOKEN),
            "allowed_users": "alice@example.com, github:octocat",
            "requests_per_minute": 5,
        }
    )
    market = MarketData.from_settings(configured)
    market.gdelt = StubGdelt()
    market.evds = StubEvds(synthetic_prices(400, ohlcv=False))
    with TestClient(create_app(configured, market=market)) as client:
        yield client


def test_health_is_public_everything_else_is_not(secured):
    assert secured.get("/healthz").status_code == 200
    for path in ("/instruments", "/sources", "/history/XU100", "/technical/XU100", "/usage"):
        assert secured.get(path).status_code == 401, path


@pytest.mark.parametrize(
    ("headers", "status"),
    [
        ({"X-User-Id": "alice@example.com"}, 401),  # no token
        (_headers(token="wrong"), 401),
        (_headers(user=""), 401),
        (_headers(user="alice@example.com\r\nX-Evil: 1"), 401),
        (_headers(user="mallory@example.com"), 403),
        (_headers(user="ALICE@example.com"), 200),  # case-insensitive
        (_headers(user="github:octocat"), 200),
    ],
)
def test_token_and_allowlist_are_enforced(secured, headers, status):
    assert secured.get("/instruments", headers=headers).status_code == status


def test_uploads_are_isolated_between_authenticated_users(secured):
    frame = synthetic_prices(300).round(4)
    frame.index.name = "Date"
    upload = secured.post(
        "/datasets",
        headers=_headers(),
        data={"symbol": "THYAO"},
        files={"file": ("p.csv", frame.to_csv().encode())},
    )
    dataset_id = upload.json()["dataset_id"]

    own = secured.get(f"/technical/THYAO?dataset_id={dataset_id}", headers=_headers())
    other = secured.get(
        f"/technical/THYAO?dataset_id={dataset_id}", headers=_headers("github:octocat")
    )
    assert own.status_code == 200
    assert other.status_code == 422
    assert "not found" in other.json()["detail"]


def test_per_user_rate_limit(secured):
    statuses = [secured.get("/instruments", headers=_headers()).status_code for _ in range(6)]
    assert statuses[:5] == [200] * 5
    assert statuses[5] == 429
    # Another user is unaffected.
    assert secured.get("/instruments", headers=_headers("github:octocat")).status_code == 200


def test_security_headers_are_set(secured):
    response = secured.get("/healthz")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert response.headers["x-request-id"]


def _production(settings, **overrides):
    values = {
        "environment": "production",
        "auth_mode": "trusted-proxy",
        "internal_api_token": SecretStr("x" * 40),
        "allowed_users": "alice@example.com",
        "telemetry_salt": SecretStr("random-salt-value"),
    }
    values.update(overrides)
    return settings.model_copy(update=values)


def test_production_configuration_is_validated(settings):
    validate_security_settings(_production(settings))

    for overrides, message in [
        ({"auth_mode": "none"}, "AUTH_MODE"),
        ({"internal_api_token": SecretStr("short")}, "INTERNAL_API_TOKEN"),
        ({"allowed_users": ""}, "ALLOWED_USERS"),
        ({"telemetry_salt": SecretStr("local-development-salt")}, "TELEMETRY_SALT"),
        ({"enable_yfinance": True}, "ENABLE_YFINANCE"),
    ]:
        with pytest.raises(ConfigurationError, match=message):
            validate_security_settings(_production(settings, **overrides))


def test_production_hides_docs_and_sends_hsts(settings):
    configured = _production(settings)
    with TestClient(create_app(configured)) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404
        assert "max-age" in client.get("/healthz").headers["strict-transport-security"]


def test_identity_helpers():
    assert normalise_identity(" Alice@Example.COM ") == "alice@example.com"
    with pytest.raises(ValueError, match="Malformed"):
        normalise_identity("alice<script>")
    allowlist = parse_allowlist("alice@example.com, octocat")
    assert is_allowed("github:octocat", allowlist)
    assert not is_allowed("github:someone", allowlist)
    assert is_allowed("alice@example.com", allowlist)


def test_access_code_entry_admits_only_code_identities():
    allowlist = parse_allowlist("code:*")
    assert is_allowed("code:0123456789abcdef", allowlist)
    assert not is_allowed("github:octocat", allowlist)
    assert not is_allowed("alice@example.com", allowlist)
