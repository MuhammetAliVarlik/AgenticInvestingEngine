"""TCMB EVDS (Electronic Data Delivery System) provider.

Terms: EVDS data may be used and republished by third parties provided the
source is cited (TCMB EVDS terms of use). Every response from this module
therefore carries :data:`EVDS_SOURCE` so callers can render attribution.

API notes: the key is sent in the ``key`` request header (it was removed
from the query string in April 2024) and dates use ``dd-mm-yyyy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import httpx
import pandas as pd

from investing_engine.providers.base import ProviderError, SourceInfo, validate_price_frame
from investing_engine.universe import UNIVERSE

EVDS_SOURCE = SourceInfo(
    name="TCMB EVDS",
    url="https://evds3.tcmb.gov.tr",
    terms="Free to use and republish with attribution to the Central Bank of the "
    "Republic of Türkiye (TCMB) EVDS.",
    attribution="Source: Central Bank of the Republic of Türkiye (TCMB), EVDS.",
    deployable=True,
)

_DATE_FORMAT = "%d-%m-%Y"


@dataclass(frozen=True, slots=True)
class MacroSeries:
    code: str
    label: str
    unit: str
    yoy: bool = False
    """Report the year-over-year change (used for price indices such as CPI)."""


DEFAULT_MACRO_SERIES: tuple[MacroSeries, ...] = (
    MacroSeries("TP.DK.USD.A.YTL", "USD/TRY (indicative buying rate)", "TRY"),
    MacroSeries("TP.DK.EUR.A.YTL", "EUR/TRY (indicative buying rate)", "TRY"),
    MacroSeries("TP.APIFON4", "CBRT weighted average cost of funding", "%"),
    MacroSeries("TP.FG.J0", "Consumer price index (2003=100)", "index", yoy=True),
)


class EvdsClient:
    """Thin, typed client for the EVDS series endpoint."""

    source = EVDS_SOURCE

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str,
        timeout: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError("EVDS API key is not configured")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            headers={"key": api_key, "Accept": "application/json"},
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
        )

    def close(self) -> None:
        self._client.close()

    def fetch_series(self, codes: list[str], *, start: date, end: date) -> pd.DataFrame:
        """Fetch one or more series as a date-indexed DataFrame keyed by series code."""
        if not codes:
            raise ValueError("At least one series code is required")
        path = (
            f"series={'-'.join(codes)}"
            f"&startDate={start.strftime(_DATE_FORMAT)}"
            f"&endDate={end.strftime(_DATE_FORMAT)}"
            "&type=json"
        )
        try:
            response = self._client.get(path)
        except httpx.HTTPError as exc:
            raise ProviderError(f"EVDS request failed: {type(exc).__name__}") from exc

        if response.status_code in (401, 403):
            raise ProviderError("EVDS rejected the API key")
        if response.status_code != 200:
            raise ProviderError(f"EVDS returned HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError("EVDS returned a non-JSON response") from exc

        return _items_to_frame(payload, codes)


def _items_to_frame(payload: Any, codes: list[str]) -> pd.DataFrame:
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ProviderError("EVDS response did not contain an 'items' list")

    # EVDS replaces dots with underscores in item keys.
    columns = {code: code.replace(".", "_") for code in codes}
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or "Tarih" not in item:
            continue
        row: dict[str, Any] = {"date": item["Tarih"]}
        for code, key in columns.items():
            row[code] = item.get(key)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=codes, dtype="float64")

    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], format="mixed", dayfirst=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
    numeric: pd.DataFrame = frame[codes].apply(pd.to_numeric, errors="coerce")
    return numeric


class EvdsIndexPriceProvider:
    """Daily closing levels for indices whose prices TCMB republishes on EVDS."""

    source = EVDS_SOURCE

    def __init__(self, client: EvdsClient) -> None:
        self._client = client

    def get_history(self, symbol: str, *, lookback_days: int) -> pd.DataFrame:
        instrument = UNIVERSE.get(symbol)
        if instrument is None or instrument.evds_series is None:
            raise ProviderError(f"EVDS does not publish prices for {symbol}")

        end = date.today()
        start = end - timedelta(days=lookback_days)
        frame = self._client.fetch_series([instrument.evds_series], start=start, end=end)
        closes = frame.rename(columns={instrument.evds_series: "Close"})
        return validate_price_frame(closes, symbol=symbol)


def macro_snapshot(
    client: EvdsClient,
    series: tuple[MacroSeries, ...] = DEFAULT_MACRO_SERIES,
    *,
    today: date | None = None,
) -> list[dict[str, Any]]:
    """Latest value and recent change for each macro series.

    Daily series report the change over ~30 days; series flagged ``yoy``
    report the year-over-year change of the latest observation.
    """
    end = today or date.today()
    start = end - timedelta(days=500)
    frame = client.fetch_series([s.code for s in series], start=start, end=end)

    snapshot: list[dict[str, Any]] = []
    for spec in series:
        column = frame[spec.code].dropna() if spec.code in frame else pd.Series(dtype="float64")
        if column.empty:
            snapshot.append({"code": spec.code, "label": spec.label, "available": False})
            continue

        latest_date = column.index[-1]
        latest = float(column.iloc[-1])
        window = timedelta(days=365) if spec.yoy else timedelta(days=30)
        reference = column[column.index <= latest_date - window]
        change_pct = (
            round((latest / float(reference.iloc[-1]) - 1) * 100, 2)
            if not reference.empty and float(reference.iloc[-1]) != 0
            else None
        )
        snapshot.append(
            {
                "code": spec.code,
                "label": spec.label,
                "unit": spec.unit,
                "available": True,
                "date": latest_date.date().isoformat(),
                "value": round(latest, 4),
                "change_pct": change_pct,
                "change_window": "1y" if spec.yoy else "30d",
            }
        )
    return snapshot
