"""Yahoo Finance price provider - local, personal use only.

Yahoo's terms restrict this data to personal use and prohibit
redistribution, so the provider is marked non-deployable and is only
constructed when ``ENABLE_YFINANCE=true`` is set on a developer machine.
The ``yfinance`` package is an optional dependency (``pip install .[local]``).
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from investing_engine.providers.base import (
    OHLCV_COLUMNS,
    ProviderError,
    SourceInfo,
    validate_price_frame,
)

YAHOO_SOURCE = SourceInfo(
    name="Yahoo Finance (via yfinance)",
    url="https://finance.yahoo.com",
    terms="Personal, non-commercial use only; redistribution is not permitted.",
    attribution="Price data: Yahoo Finance, for personal research use.",
    deployable=False,
)


class YahooPriceProvider:
    source = YAHOO_SOURCE

    def get_history(self, symbol: str, *, lookback_days: int) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ProviderError("yfinance is not installed (pip install .[local])") from exc

        start = date.today() - timedelta(days=lookback_days)
        try:
            frame = yf.download(
                f"{symbol}.IS",
                start=start.isoformat(),
                interval="1d",
                progress=False,
                auto_adjust=False,
                multi_level_index=False,
            )
        except Exception as exc:  # yfinance raises a wide range of exception types
            raise ProviderError(f"Yahoo Finance request failed: {type(exc).__name__}") from exc

        if frame is None or frame.empty:
            raise ProviderError(f"Yahoo Finance returned no data for {symbol}")
        columns = [c for c in OHLCV_COLUMNS if c in frame.columns]
        return validate_price_frame(frame[columns], symbol=symbol)
