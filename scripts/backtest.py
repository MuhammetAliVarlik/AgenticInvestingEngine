#!/usr/bin/env python3
"""Directional backtest of the RSI mean-reversion signal thresholds.

Scores ``classify_signal`` (bullish below RSI 30, bearish above 70) on
realised historical RSI: for every day a signal fires, it checks whether the
close ``--horizon`` trading days later moved in the signalled direction.
Realised RSI stands in for the live model's one-step-ahead forecast, so this
measures whether the threshold framing is directionally meaningful, not the
accuracy of the RandomForest forecast or of the generated report text.

Price history is fetched from Yahoo Finance, which is for personal research
use only - this script is a local research tool and is not part of any
deployed build. Requires ``pip install -e .[local]``.

Usage:
    python scripts/backtest.py
    python scripts/backtest.py --symbols THYAO,ASELS --days 730 --horizon 5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd
import ta

from investing_engine.analysis.indicators import classify_signal
from investing_engine.providers.base import ProviderError
from investing_engine.providers.yahoo import YahooPriceProvider

DEFAULT_SYMBOLS = ("THYAO", "ASELS", "TUPRS", "EREGL", "BIMAS")


@dataclass
class TickerResult:
    symbol: str
    bullish: int
    bullish_hits: int
    bearish: int
    bearish_hits: int


def backtest(prices: pd.DataFrame, symbol: str, horizon: int) -> TickerResult:
    close = prices["Close"]
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
    signal = rsi.map(lambda v: classify_signal(v) if pd.notna(v) else None)
    forward = close.shift(-horizon) / close - 1
    frame = pd.DataFrame({"signal": signal, "forward": forward}).dropna()

    bullish = frame[frame["signal"] == "bullish"]
    bearish = frame[frame["signal"] == "bearish"]
    return TickerResult(
        symbol=symbol,
        bullish=len(bullish),
        bullish_hits=int((bullish["forward"] > 0).sum()),
        bearish=len(bearish),
        bearish_hits=int((bearish["forward"] < 0).sum()),
    )


def _rate(hits: int, total: int) -> str:
    return f"{hits / total * 100:.1f}%" if total else "-"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--days", type=int, default=730, help="calendar days of history")
    parser.add_argument("--horizon", type=int, default=5, help="trading days ahead")
    args = parser.parse_args()

    provider = YahooPriceProvider()
    results: list[TickerResult] = []
    print(f"RSI-threshold backtest, horizon = {args.horizon} trading days\n")
    print(f"{'Symbol':<8}{'Bullish n':>11}{'Hit rate':>10}{'Bearish n':>11}{'Hit rate':>10}")

    for symbol in (s.strip().upper() for s in args.symbols.split(",") if s.strip()):
        try:
            prices = provider.get_history(symbol, lookback_days=args.days)
        except ProviderError as exc:
            print(f"{symbol:<8}{exc}")
            continue
        r = backtest(prices, symbol, args.horizon)
        results.append(r)
        print(
            f"{r.symbol:<8}{r.bullish:>11}{_rate(r.bullish_hits, r.bullish):>10}"
            f"{r.bearish:>11}{_rate(r.bearish_hits, r.bearish):>10}"
        )

    bullish = sum(r.bullish for r in results)
    bearish = sum(r.bearish for r in results)
    print(f"\nOverall bullish hit rate: {_rate(sum(r.bullish_hits for r in results), bullish)}"
          f" over {bullish} signals")  # fmt: skip
    print(f"Overall bearish hit rate: {_rate(sum(r.bearish_hits for r in results), bearish)}"
          f" over {bearish} signals")  # fmt: skip


if __name__ == "__main__":
    main()
