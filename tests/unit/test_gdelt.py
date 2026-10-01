import httpx
import pytest
import respx

from investing_engine.providers.base import ProviderError
from investing_engine.providers.gdelt import GdeltNewsProvider
from investing_engine.universe import resolve

BASE_URL = "https://gdelt.test/api/v2/doc/doc"


def _provider() -> GdeltNewsProvider:
    return GdeltNewsProvider(base_url=BASE_URL, timeout=5, min_interval_seconds=0)


@respx.mock
def test_headlines_use_allowlisted_query_and_keep_metadata_only():
    route = respx.get(BASE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "articles": [
                    {
                        "url": "https://news.example/a",
                        "title": "Turkish Airlines expands fleet " + "x" * 500,
                        "seendate": "20261001T083000Z",
                        "domain": "news.example",
                        "language": "English",
                    },
                    {"title": "missing url is dropped"},
                ]
            },
        )
    )
    headlines = _provider().headlines(resolve("THYAO"), days=3, limit=5)

    params = route.calls.last.request.url.params
    assert params["query"] == '"Turkish Airlines"'
    assert params["timespan"] == "3d"
    assert params["mode"] == "ArtList"
    assert len(headlines) == 1
    assert len(headlines[0]["title"]) == 300
    assert headlines[0]["published_at"] == "2026-10-01T08:30:00+00:00"
    assert set(headlines[0]) == {"title", "outlet", "url", "language", "published_at"}


@respx.mock
def test_average_tone_means_all_points():
    respx.get(BASE_URL).mock(
        return_value=httpx.Response(
            200,
            json={"timeline": [{"data": [{"value": -2.0}, {"value": 1.0}, {"value": "bad"}]}]},
        )
    )
    assert _provider().average_tone(resolve("THYAO")) == -0.5


@respx.mock
def test_empty_body_means_no_coverage():
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, text=""))
    assert _provider().headlines(resolve("THYAO")) == []
    assert _provider().average_tone(resolve("THYAO")) is None


@respx.mock
def test_plain_text_error_is_reported_cleanly():
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, text="Your search was too short"))
    with pytest.raises(ProviderError, match="non-JSON"):
        _provider().headlines(resolve("THYAO"))


@respx.mock
def test_rate_limit_is_reported():
    respx.get(BASE_URL).mock(return_value=httpx.Response(429))
    with pytest.raises(ProviderError, match="rate limit"):
        _provider().headlines(resolve("THYAO"))


@respx.mock
def test_requests_are_spaced_by_the_minimum_interval(monkeypatch):
    clock = {"now": 100.0}
    sleeps = []
    monkeypatch.setattr("investing_engine.providers.gdelt.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        "investing_engine.providers.gdelt.time.sleep",
        lambda s: (sleeps.append(s), clock.__setitem__("now", clock["now"] + s)),
    )
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, json={}))
    provider = GdeltNewsProvider(base_url=BASE_URL, timeout=5, min_interval_seconds=5.5)

    provider.headlines(resolve("THYAO"))
    clock["now"] += 1.0
    provider.average_tone(resolve("THYAO"))

    assert sleeps == [pytest.approx(4.5)]
