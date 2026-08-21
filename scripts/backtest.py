#!/usr/bin/env python3
"""Rough backtest of the rule-based RSI signal threshold logic.

This evaluates ONLY the rule-based classify_signal() threshold logic in
api/tools/stockPriceAnaliserTool.py (bullish when RSI < 30, bearish when
RSI > 70) against realized historical RSI and subsequent price movement.

Important scope note: the live system predicts *next-period* RSI with a
RandomForestRegressor and classifies that prediction; retraining that model
at every historical point would be expensive and is out of scope for a
lightweight backtest. This script instead uses REALIZED (actual) historical
RSI as a proxy for what the model tries to approximate one step ahead, and
checks whether price N trading days later moved in the direction the
threshold-based signal would suggest. It answers "is this threshold
framing directionally meaningful on history", not "how accurate is the
live model's prediction" and NOT "how good is the LLM-generated investment
strategy text" - the latter isn't something a rule-based backtest can
evaluate.

Usage:
    python scripts/backtest.py
    python scripts/backtest.py --tickers THYAO.IS,ASELS.IS --period 2y --horizon 5
"""
import argparse
import os
import sys

import pandas as pd
import ta
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
from tools.stockPriceAnaliserTool import classify_signal  # noqa: E402

DEFAULT_TICKERS = ["THYAO.IS", "ASELS.IS", "TUPRS.IS", "VESTL.IS", "BAYRK.IS"]


def backtest_ticker(ticker: str, period: str, horizon_days: int) -> dict:
    df = yf.download(ticker, interval="1d", period=period, progress=False)[["Close"]]
    if df.empty:
        return {"ticker": ticker, "error": "no data"}

    close = df["Close"].squeeze()
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
    signal = rsi.apply(lambda v: classify_signal(v) if pd.notna(v) else None)
    future_return = close.shift(-horizon_days) / close - 1

    frame = pd.DataFrame({"signal": signal, "future_return": future_return}).dropna()

    bullish = frame[frame["signal"] == "bullish"]
    bearish = frame[frame["signal"] == "bearish"]

    return {
        "ticker": ticker,
        "bullish_signals": len(bullish),
        "bullish_hit_rate": float((bullish["future_return"] > 0).mean()) if len(bullish) else None,
        "bearish_signals": len(bearish),
        "bearish_hit_rate": float((bearish["future_return"] < 0).mean()) if len(bearish) else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS), help="comma-separated BIST100 tickers")
    parser.add_argument("--period", default="2y", help="yfinance history period, e.g. 1y, 2y, 5y")
    parser.add_argument("--horizon", type=int, default=5, help="trading days ahead to check directional correctness")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    print(f"Backtesting classify_signal() RSI thresholds, horizon={args.horizon} trading days\n")
    print(f"{'Ticker':<10} {'Bullish n':>10} {'Bullish hit%':>13} {'Bearish n':>10} {'Bearish hit%':>13}")

    all_bullish_hits, all_bullish_n = 0.0, 0
    all_bearish_hits, all_bearish_n = 0.0, 0

    for ticker in tickers:
        result = backtest_ticker(ticker, args.period, args.horizon)
        if "error" in result:
            print(f"{ticker:<10} {result['error']}")
            continue

        bh = result["bullish_hit_rate"]
        be = result["bearish_hit_rate"]
        print(
            f"{ticker:<10} {result['bullish_signals']:>10} "
            f"{'' if bh is None else f'{bh * 100:.1f}%':>13} "
            f"{result['bearish_signals']:>10} "
            f"{'' if be is None else f'{be * 100:.1f}%':>13}"
        )

        if bh is not None:
            all_bullish_hits += bh * result["bullish_signals"]
            all_bullish_n += result["bullish_signals"]
        if be is not None:
            all_bearish_hits += be * result["bearish_signals"]
            all_bearish_n += result["bearish_signals"]

    print()
    if all_bullish_n:
        print(f"Overall bullish hit rate:  {all_bullish_hits / all_bullish_n * 100:.1f}% over {all_bullish_n} signals")
    if all_bearish_n:
        print(f"Overall bearish hit rate:  {all_bearish_hits / all_bearish_n * 100:.1f}% over {all_bearish_n} signals")
    print(
        "\nNote: this scores the rule-based RSI-threshold signal only, using realized "
        "historical RSI as a stand-in for the live model's next-period prediction. It is "
        "not an evaluation of the trained RandomForest model's live predictions or of the "
        "LLM-generated investment strategy text."
    )


if __name__ == "__main__":
    main()
