from datetime import date, timedelta

import httpx
import pytest
import respx

from investing_engine.providers.base import ProviderError
from investing_engine.providers.evds import (
    EvdsClient,
    EvdsIndexPriceProvider,
    MacroSeries,
    macro_snapshot,
)

BASE_URL = "https://evds.test/service/evds"


def _client() -> EvdsClient:
    return EvdsClient("secret-key", base_url=BASE_URL, timeout=5)


def _index_items(rows: int) -> list[dict]:
    start = date.today() - timedelta(days=rows)
    return [
        {
            "Tarih": (start + timedelta(days=i)).strftime("%d-%m-%Y"),
            "TP_MK_F_BILESIK": None if i % 7 == 6 else f"{10000 + i * 3.5:.2f}",
        }
        for i in range(rows)
    ]


@respx.mock
def test_fetch_series_sends_key_in_header_and_builds_path():
    route = respx.get(url__startswith=f"{BASE_URL}/series=TP.A-TP.B").mock(
        return_value=httpx.Response(200, json={"items": []})
    )
    _client().fetch_series(["TP.A", "TP.B"], start=date(2026, 1, 2), end=date(2026, 2, 3))

    request = route.calls.last.request
    assert request.headers["key"] == "secret-key"
    assert "secret-key" not in str(request.url)
    assert "startDate=02-01-2026" in str(request.url)
    assert "endDate=03-02-2026" in str(request.url)
    assert "type=json" in str(request.url)


@respx.mock
def test_index_provider_parses_closes_and_skips_holidays():
    respx.get(url__startswith=BASE_URL).mock(
        return_value=httpx.Response(200, json={"totalCount": 200, "items": _index_items(200)})
    )
    frame = EvdsIndexPriceProvider(_client()).get_history("XU100", lookback_days=200)

    assert list(frame.columns) == ["Close"]
    assert frame["Close"].notna().all()
    assert frame.index.is_monotonic_increasing
    assert len(frame) < 200  # null (holiday) rows removed


@respx.mock
@pytest.mark.parametrize("status", [401, 403])
def test_rejected_key_raises_clean_error(status):
    respx.get(url__startswith=BASE_URL).mock(return_value=httpx.Response(status))
    with pytest.raises(ProviderError, match="rejected the API key") as info:
        _client().fetch_series(["TP.A"], start=date(2026, 1, 1), end=date(2026, 1, 2))
    assert "secret-key" not in str(info.value)


@respx.mock
def test_non_json_response_raises():
    respx.get(url__startswith=BASE_URL).mock(return_value=httpx.Response(200, text="<html>"))
    with pytest.raises(ProviderError, match="non-JSON"):
        _client().fetch_series(["TP.A"], start=date(2026, 1, 1), end=date(2026, 1, 2))


def test_index_provider_refuses_symbols_without_evds_series():
    with pytest.raises(ProviderError, match="does not publish"):
        EvdsIndexPriceProvider(_client()).get_history("THYAO", lookback_days=100)


def test_missing_key_is_rejected_up_front():
    with pytest.raises(ProviderError, match="not configured"):
        EvdsClient("", base_url=BASE_URL, timeout=5)


@respx.mock
def test_macro_snapshot_computes_changes():
    today = date(2026, 6, 30)
    items = [
        {"Tarih": "29-05-2026", "TP_FX": "40.00", "TP_CPI": None},
        {"Tarih": "30-06-2026", "TP_FX": "42.00", "TP_CPI": None},
        {"Tarih": "01-06-2025", "TP_FX": None, "TP_CPI": "2000"},
        {"Tarih": "01-06-2026", "TP_FX": None, "TP_CPI": "2700"},
    ]
    respx.get(url__startswith=BASE_URL).mock(
        return_value=httpx.Response(200, json={"items": items})
    )
    series = (
        MacroSeries("TP.FX", "FX", "TRY"),
        MacroSeries("TP.CPI", "CPI", "index", yoy=True),
        MacroSeries("TP.MISSING", "Missing", "%"),
    )
    snapshot = {s["code"]: s for s in macro_snapshot(_client(), series, today=today)}

    assert snapshot["TP.FX"]["value"] == 42.0
    assert snapshot["TP.FX"]["change_pct"] == 5.0
    assert snapshot["TP.FX"]["change_window"] == "30d"
    assert snapshot["TP.CPI"]["change_pct"] == 35.0
    assert snapshot["TP.CPI"]["change_window"] == "1y"
    assert snapshot["TP.MISSING"]["available"] is False


@respx.mock
def test_series_name_reads_evds_metadata():
    respx.get(url__startswith=f"{BASE_URL}/serieList/").mock(
        return_value=httpx.Response(
            200,
            json=[{"SERIE_CODE": "TP.MK.F.BILESIK", "SERIE_NAME_ENG": "BIST-100 (XU100)"}],
        )
    )
    client = EvdsClient("key", base_url=BASE_URL, timeout=5)
    assert "(XU100)" in client.series_name("TP.MK.F.BILESIK")


@respx.mock
def test_series_name_of_unknown_code_is_an_error():
    respx.get(url__startswith=f"{BASE_URL}/serieList/").mock(
        return_value=httpx.Response(200, json=[])
    )
    with pytest.raises(ProviderError, match="no series"):
        EvdsClient("key", base_url=BASE_URL, timeout=5).series_name("TP.NOPE")
