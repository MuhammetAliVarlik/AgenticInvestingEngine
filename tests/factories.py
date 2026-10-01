"""Deterministic test data builders."""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_prices(rows: int = 300, *, seed: int = 7, ohlcv: bool = True) -> pd.DataFrame:
    """Deterministic geometric random walk with plausible OHLCV bars."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2025-01-01", periods=rows)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, rows)))
    frame = pd.DataFrame({"Close": close}, index=index)
    if ohlcv:
        spread = np.abs(rng.normal(0, 0.01, rows)) * close
        frame["Open"] = close * (1 + rng.normal(0, 0.003, rows))
        frame["High"] = np.maximum(frame["Open"], close) + spread
        frame["Low"] = np.minimum(frame["Open"], close) - spread
        frame["Volume"] = rng.integers(1_000_000, 5_000_000, rows)
    return frame
