"""Provider contracts shared by every data source.

Each provider declares where its data comes from, under which terms, and
whether those terms allow it to run in a deployed (multi-user) build. The
registry refuses to construct a non-deployable provider unless it has been
explicitly enabled for local use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Licensing and attribution metadata for a data source."""

    name: str
    url: str
    terms: str
    attribution: str
    deployable: bool


class ProviderError(RuntimeError):
    """A data source failed or returned unusable data.

    Messages are safe to surface to end users and to the LLM: they never
    include credentials, raw upstream payloads or stack traces.
    """


@runtime_checkable
class PriceProvider(Protocol):
    """Returns daily price history as a DataFrame indexed by date.

    The frame always has a ``Close`` column; ``Open``, ``High``, ``Low`` and
    ``Volume`` are included when the source publishes them.
    """

    source: SourceInfo

    def get_history(self, symbol: str, *, lookback_days: int) -> pd.DataFrame: ...


def validate_price_frame(frame: pd.DataFrame, *, symbol: str, min_rows: int = 30) -> pd.DataFrame:
    """Normalise and sanity-check a price frame coming from any provider.

    Ensures a sorted, de-duplicated ``DatetimeIndex``, numeric columns, no
    non-positive closes, and enough rows for indicator warm-up.
    """
    if "Close" not in frame.columns:
        raise ProviderError(f"No closing prices available for {symbol}")

    columns = [c for c in OHLCV_COLUMNS if c in frame.columns]
    clean: pd.DataFrame = frame[columns].apply(pd.to_numeric, errors="coerce")
    clean.index = pd.to_datetime(clean.index)
    clean = clean[~clean.index.duplicated(keep="last")].sort_index()
    clean = clean.dropna(subset=["Close"])
    clean = clean[clean["Close"] > 0]

    if len(clean) < min_rows:
        raise ProviderError(
            f"Not enough price history for {symbol}: {len(clean)} rows, need {min_rows}"
        )
    return clean
