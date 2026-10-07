#!/usr/bin/env python3
"""Check that every configured EVDS series code returns live data.

Run once after setting ``EVDS_API_KEY`` (in the environment or ``.env``):

    python scripts/verify_evds_series.py

Exits non-zero if any index or macro series comes back empty, so a renamed
or retired TCMB series is caught before it reaches users.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

from investing_engine.config import get_settings
from investing_engine.providers.base import ProviderError
from investing_engine.providers.evds import DEFAULT_MACRO_SERIES, EvdsClient
from investing_engine.universe import UNIVERSE


def main() -> int:
    settings = get_settings()
    if settings.evds_api_key is None:
        print("EVDS_API_KEY is not set.", file=sys.stderr)
        return 2

    client = EvdsClient(
        settings.evds_api_key.get_secret_value(),
        base_url=settings.evds_base_url,
        timeout=settings.http_timeout_seconds,
    )
    series = {i.evds_series: f"{i.symbol} close" for i in UNIVERSE.values() if i.evds_series}
    series |= {s.code: s.label for s in DEFAULT_MACRO_SERIES}

    failures = 0
    # A series can return data and still be the wrong one (XU100 once pointed at
    # the BIST All Shares index), so index series must name their symbol.
    for instrument in UNIVERSE.values():
        if not instrument.evds_series:
            continue
        try:
            name = client.series_name(instrument.evds_series)
        except ProviderError as exc:
            print(f"FAIL  {instrument.evds_series:<22} name lookup: {exc}")
            failures += 1
            continue
        if f"({instrument.symbol})" not in name:
            print(f"FAIL  {instrument.evds_series:<22} is '{name}', not {instrument.symbol}")
            failures += 1
        else:
            print(f"OK    {instrument.evds_series:<22} {name}")

    end = date.today()
    start = end - timedelta(days=400)
    for code, label in series.items():
        try:
            column = client.fetch_series([code], start=start, end=end)[code].dropna()
        except ProviderError as exc:
            print(f"FAIL  {code:<22} {label}: {exc}")
            failures += 1
            continue
        if column.empty:
            print(f"FAIL  {code:<22} {label}: no observations")
            failures += 1
        else:
            latest = column.index[-1].date()
            print(
                f"OK    {code:<22} {label}: {len(column)} obs, latest {latest} = {column.iloc[-1]}"
            )
    client.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
