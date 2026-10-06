"""Technical-indicator feature engineering.

Works on any validated price frame. Close-only sources (such as index levels
from EVDS) get the close-based feature set; full OHLCV sources additionally
get range- and volume-based indicators.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd
import ta

CLOSE_FEATURES: tuple[str, ...] = (
    "Close", "ma_5", "returns", "rsi", "ema34", "ema89",
    "macd", "macd_signal", "macd_diff",
    "bb_high", "bb_low", "bb_mid", "bb_pct",
)  # fmt: skip
RANGE_FEATURES: tuple[str, ...] = ("atr", "adx", "adx_pos", "adx_neg", "stoch_k", "stoch_d")
VOLUME_FEATURES: tuple[str, ...] = ("obv",)

TARGET_COLUMN = "rsi_target"


class FeatureSet(str, Enum):
    CLOSE = "close"
    OHLC = "ohlc"
    OHLCV = "ohlcv"

    @property
    def columns(self) -> tuple[str, ...]:
        if self is FeatureSet.CLOSE:
            return CLOSE_FEATURES
        if self is FeatureSet.OHLC:
            return CLOSE_FEATURES + RANGE_FEATURES
        return CLOSE_FEATURES + RANGE_FEATURES + VOLUME_FEATURES


def detect_feature_set(frame: pd.DataFrame) -> FeatureSet:
    if not {"High", "Low"}.issubset(frame.columns):
        return FeatureSet.CLOSE
    has_range = bool(frame[["High", "Low"]].notna().all().all())
    if not has_range:
        return FeatureSet.CLOSE
    if "Volume" in frame.columns and frame["Volume"].fillna(0).gt(0).any():
        return FeatureSet.OHLCV
    return FeatureSet.OHLC


def compute_features(prices: pd.DataFrame) -> tuple[pd.DataFrame, FeatureSet]:
    """Add indicator columns plus the next-period RSI target.

    Rows still warming up (e.g. the first 88 rows for EMA89) are dropped. The
    final row is kept even though its target is unknown, because it is the
    row the model must predict from.
    """
    feature_set = detect_feature_set(prices)
    df = prices.copy()
    close = df["Close"]

    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["ema34"] = ta.trend.EMAIndicator(close, window=34).ema_indicator()
    df["ema89"] = ta.trend.EMAIndicator(close, window=89).ema_indicator()
    df["ma_5"] = close.rolling(5).mean()
    df["returns"] = close.pct_change()

    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    band_width = df["bb_high"] - df["bb_low"]
    # 0.5 (mid-band) is the neutral value when the bands are flat.
    df["bb_pct"] = np.where(band_width != 0, (close - df["bb_low"]) / band_width, 0.5)

    if feature_set in (FeatureSet.OHLC, FeatureSet.OHLCV):
        high, low = df["High"], df["Low"]
        df["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
        adx = ta.trend.ADXIndicator(high, low, close, window=14)
        df["adx"] = adx.adx()
        df["adx_pos"] = adx.adx_pos()
        df["adx_neg"] = adx.adx_neg()
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

    if feature_set is FeatureSet.OHLCV:
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(close, df["Volume"]).on_balance_volume()

    df[TARGET_COLUMN] = df["rsi"].shift(-1)
    df = df.dropna(subset=list(feature_set.columns))
    return df, feature_set


def rsi_fibonacci_levels(df: pd.DataFrame, *, lookback: int = 50) -> dict[str, float]:
    """Fibonacci retracement levels between recent RSI swing high and low.

    The swing series follows the previous bar's RSI, so the levels are based
    only on completed bars.
    """
    swing = df["rsi"].shift(1)
    peak = float(swing.rolling(window=lookback, min_periods=1).max().iloc[-1])
    trough = float(swing.rolling(window=lookback, min_periods=1).min().iloc[-1])
    span = peak - trough
    return {
        "rsi_range_high": round(peak, 2),
        "rsi_range_low": round(trough, 2),
        "rsi_fib_236": round(trough + span * 0.236, 2),
        "rsi_fib_382": round(trough + span * 0.382, 2),
        "rsi_fib_500": round(trough + span * 0.500, 2),
        "rsi_fib_618": round(trough + span * 0.618, 2),
        "rsi_fib_786": round(trough + span * 0.786, 2),
    }


def classify_signal(predicted_rsi: float) -> str:
    """Mean-reversion reading of the *predicted* next-period RSI.

    ``bullish`` below 30 (oversold, bounce expected), ``bearish`` above 70
    (overbought, pull-back expected), otherwise ``neutral``. The boundaries
    themselves resolve to ``neutral``.
    """
    if predicted_rsi < 30:
        return "bullish"
    if predicted_rsi > 70:
        return "bearish"
    return "neutral"


def momentum_divergence(df: pd.DataFrame, *, window: int = 5) -> bool:
    """True when RSI and price have moved in opposite directions over ``window`` bars."""
    rsi_trend = df["rsi"].diff().iloc[-window:].mean()
    price_trend = df["Close"].diff().iloc[-window:].mean()
    return bool((rsi_trend > 0 > price_trend) or (rsi_trend < 0 < price_trend))


def channel_position(df: pd.DataFrame, *, window: int = 20) -> str:
    """Where the last close sits within the recent trading range (bottom/middle/top quartiles)."""
    lows = df["Low"] if "Low" in df.columns and df["Low"].notna().all() else df["Close"]
    highs = df["High"] if "High" in df.columns and df["High"].notna().all() else df["Close"]
    floor = float(lows.rolling(window).min().iloc[-1])
    ceiling = float(highs.rolling(window).max().iloc[-1])
    close = float(df["Close"].iloc[-1])
    span = ceiling - floor
    if span <= 0:
        return "middle"
    if close < floor + span * 0.25:
        return "bottom"
    if close > floor + span * 0.75:
        return "top"
    return "middle"
