import os
import time

import numpy as np
import pandas as pd
import pytest

from tools import stockPriceAnaliserTool as spat


def make_ohlcv(n=250, seed=42, trend=0.0):
    """Synthetic OHLCV data: a mild random walk with an optional drift, long
    enough for EMA89/RSI/etc. to have valid values after dropna()."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(trend, 1, n))
    high = close + rng.uniform(0.5, 2, n)
    low = close - rng.uniform(0.5, 2, n)
    open_ = np.clip(close + rng.normal(0, 0.5, n), low, high)
    volume = rng.integers(1000, 100_000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


# --- classify_signal --------------------------------------------------

@pytest.mark.parametrize(
    "pred_rsi,expected",
    [
        (10.0, "bullish"),
        (29.999, "bullish"),
        (30.0, "neutral"),   # boundary: current code uses strict `<`, so exactly 30 is neutral
        (50.0, "neutral"),
        (70.0, "neutral"),   # boundary: exactly 70 is neutral
        (70.001, "bearish"),
        (90.0, "bearish"),
    ],
)
def test_classify_signal_thresholds(pred_rsi, expected):
    assert spat.classify_signal(pred_rsi) == expected


# --- bb_pct divide-by-zero guard ---------------------------------------

def test_bb_pct_guard_formula_returns_neutral_when_bands_flat():
    close = pd.Series([100.0, 100.0, 100.0])
    bb_high = pd.Series([100.0, 100.0, 100.0])
    bb_low = pd.Series([100.0, 100.0, 100.0])
    denom = bb_high - bb_low
    bb_pct = np.where(denom != 0, (close - bb_low) / denom, 0.5)
    assert (bb_pct == 0.5).all()


def test_compute_features_bb_pct_has_no_nan_or_inf(monkeypatch):
    monkeypatch.setattr(spat.yf, "download", lambda *a, **k: make_ohlcv())
    df = spat.compute_features("TEST.IS")
    assert not df["bb_pct"].isna().any()
    assert not np.isinf(df["bb_pct"]).any()


# --- calculate_rsi_fibo_levels -----------------------------------------

def test_calculate_rsi_fibo_levels_adds_expected_columns(monkeypatch):
    monkeypatch.setattr(spat.yf, "download", lambda *a, **k: make_ohlcv())
    df = spat.compute_features("TEST.IS")
    df = spat.calculate_rsi_fibo_levels(df)
    for col in ["peak", "trough", "M236", "M386", "M500", "M618", "M786"]:
        assert col in df.columns

    latest = df.iloc[-1]
    if pd.notna(latest["peak"]) and pd.notna(latest["trough"]):
        # Fibonacci levels must sit between trough and peak, in ascending order.
        assert latest["trough"] <= latest["M236"] <= latest["M500"] <= latest["M786"] <= latest["peak"]


# --- price_above_ema34 crossover ----------------------------------------

def test_rsi_predictor_price_above_ema34_reflects_uptrend(tmp_path, monkeypatch):
    monkeypatch.setattr(spat, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(spat.yf, "download", lambda *a, **k: make_ohlcv(n=250, trend=0.3))

    result = spat.rsi_predictor.invoke({"ticker": "UP.IS"})

    assert "error" not in result
    assert result["price_above_ema34"] is True


def test_rsi_predictor_price_above_ema34_reflects_downtrend(tmp_path, monkeypatch):
    monkeypatch.setattr(spat, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(spat.yf, "download", lambda *a, **k: make_ohlcv(n=250, trend=-0.3))

    result = spat.rsi_predictor.invoke({"ticker": "DOWN.IS"})

    assert "error" not in result
    assert result["price_above_ema34"] is False


def test_rsi_predictor_includes_price_field(tmp_path, monkeypatch):
    monkeypatch.setattr(spat, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(spat.yf, "download", lambda *a, **k: make_ohlcv())

    result = spat.rsi_predictor.invoke({"ticker": "TEST.IS"})

    assert "error" not in result
    assert isinstance(result["price"], float)


def test_rsi_predictor_handles_empty_data_gracefully(monkeypatch):
    monkeypatch.setattr(spat.yf, "download", lambda *a, **k: pd.DataFrame())

    result = spat.rsi_predictor.invoke({"ticker": "NODATA.IS"})

    assert "error" in result


# --- model caching (_is_model_fresh / get_model) ------------------------

def test_is_model_fresh_missing_file_returns_false(tmp_path):
    assert spat._is_model_fresh(str(tmp_path / "missing.joblib")) is False


def test_is_model_fresh_recent_file_returns_true(tmp_path):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"placeholder")
    assert spat._is_model_fresh(str(path), max_age_hours=24) is True


def test_is_model_fresh_stale_file_returns_false(tmp_path):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"placeholder")
    old_time = time.time() - (25 * 3600)
    os.utime(path, (old_time, old_time))
    assert spat._is_model_fresh(str(path), max_age_hours=24) is False


def test_get_model_uses_cached_model_when_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(spat, "MODEL_DIR", str(tmp_path))

    # Build a real df via a mocked download, train once, then confirm a
    # second call with training spied on does NOT retrain.
    monkeypatch.setattr(spat.yf, "download", lambda *a, **k: make_ohlcv())
    df = spat.compute_features("CACHE.IS")

    trained = spat.train_model("CACHE.IS", df)
    assert os.path.exists(spat.get_model_path("CACHE.IS"))

    calls = []
    original_train = spat.train_model

    def spy_train(ticker, df):
        calls.append(ticker)
        return original_train(ticker, df)

    monkeypatch.setattr(spat, "train_model", spy_train)

    spat.get_model("CACHE.IS", df)  # fresh cache exists -> should NOT call train_model

    assert calls == []


def test_get_model_retrains_when_cache_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(spat, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(spat.yf, "download", lambda *a, **k: make_ohlcv())
    df = spat.compute_features("STALE.IS")

    spat.train_model("STALE.IS", df)
    path = spat.get_model_path("STALE.IS")
    old_time = time.time() - (25 * 3600)
    os.utime(path, (old_time, old_time))

    calls = []
    original_train = spat.train_model

    def spy_train(ticker, df):
        calls.append(ticker)
        return original_train(ticker, df)

    monkeypatch.setattr(spat, "train_model", spy_train)

    spat.get_model("STALE.IS", df)

    assert calls == ["STALE.IS"]
