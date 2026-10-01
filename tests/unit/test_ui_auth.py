"""UI sign-in logic and a smoke run of the Streamlit script."""

import sys
import types
from pathlib import Path

import pytest

UI_DIR = Path(__file__).parents[2] / "ui"
sys.path.insert(0, str(UI_DIR))

import auth  # noqa: E402


class PageStoppedError(Exception):
    pass


class FakeUser(dict):
    def __init__(self, logged_in: bool, email: str = ""):
        super().__init__(email=email)
        self.is_logged_in = logged_in


@pytest.fixture
def fake_st(monkeypatch):
    calls = {"errors": [], "logins": 0}

    def stop():
        raise PageStoppedError

    fake = types.SimpleNamespace(
        context=types.SimpleNamespace(headers={}),
        user=FakeUser(False),
        error=lambda msg: calls["errors"].append(msg),
        stop=stop,
        title=lambda *a, **k: None,
        caption=lambda *a, **k: None,
        button=lambda *a, **k: False,
        login=lambda provider: calls.__setitem__("logins", calls["logins"] + 1),
        logout=lambda: None,
    )
    monkeypatch.setattr(auth, "st", fake)
    return fake, calls


def test_local_mode_needs_no_sign_in(fake_st, monkeypatch):
    monkeypatch.setenv("AUTH_PROVIDER", "none")
    monkeypatch.delenv("ALLOWED_USERS", raising=False)
    assert auth.require_user() == "local"


@pytest.mark.parametrize(
    ("name", "idp", "expected"),
    [("octocat", "github", "github:octocat"), ("Alice@Example.com", "google", "alice@example.com")],
)
def test_easyauth_identity_from_platform_headers(fake_st, monkeypatch, name, idp, expected):
    fake, _ = fake_st
    monkeypatch.setenv("AUTH_PROVIDER", "easyauth")
    monkeypatch.setenv("ALLOWED_USERS", "octocat,alice@example.com")
    fake.context.headers = {"X-Ms-Client-Principal-Name": name, "X-Ms-Client-Principal-Idp": idp}
    assert auth.require_user() == expected


def test_easyauth_without_headers_is_refused(fake_st, monkeypatch):
    monkeypatch.setenv("AUTH_PROVIDER", "easyauth")
    monkeypatch.setenv("ALLOWED_USERS", "octocat")
    with pytest.raises(PageStoppedError):
        auth.require_user()


def test_uninvited_user_is_refused(fake_st, monkeypatch):
    fake, calls = fake_st
    monkeypatch.setenv("AUTH_PROVIDER", "easyauth")
    monkeypatch.setenv("ALLOWED_USERS", "alice@example.com")
    fake.context.headers = {
        "X-Ms-Client-Principal-Name": "mallory",
        "X-Ms-Client-Principal-Idp": "github",
    }
    with pytest.raises(PageStoppedError):
        auth.require_user()
    assert "not been invited" in calls["errors"][-1]


def test_deployed_mode_with_empty_allowlist_denies_everyone(fake_st, monkeypatch):
    fake, _ = fake_st
    monkeypatch.setenv("AUTH_PROVIDER", "easyauth")
    monkeypatch.setenv("ALLOWED_USERS", "")
    fake.context.headers = {"X-Ms-Client-Principal-Name": "alice@example.com"}
    with pytest.raises(PageStoppedError):
        auth.require_user()


def test_oidc_requires_login_then_checks_allowlist(fake_st, monkeypatch):
    fake, _ = fake_st
    monkeypatch.setenv("AUTH_PROVIDER", "oidc")
    monkeypatch.setenv("ALLOWED_USERS", "alice@example.com")
    with pytest.raises(PageStoppedError):
        auth.require_user()

    fake.user = FakeUser(True, "Alice@Example.com")
    assert auth.require_user() == "alice@example.com"


def test_api_headers_carry_identity_and_internal_token(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_TOKEN", "secret-token")
    assert auth.api_headers("alice@example.com") == {
        "X-User-Id": "alice@example.com",
        "X-Internal-Token": "secret-token",
    }


def test_streamlit_app_renders_without_api(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("AUTH_PROVIDER", "none")
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:9")  # nothing listens here
    app = AppTest.from_file(str(UI_DIR / "streamlit_app.py"), default_timeout=30).run()

    assert not app.exception
    assert "Cannot reach the API" in app.error[0].value
