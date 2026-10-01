import pandas as pd
import pytest

from investing_engine.analysis.indicators import (
    FeatureSet,
    channel_position,
    classify_signal,
    compute_features,
    detect_feature_set,
)
from investing_engine.analysis.technical import ModelCache, analyze
from investing_engine.providers.base import ProviderError
from tests.factories import synthetic_prices


@pytest.mark.parametrize(
    ("rsi", "signal"),
    [(10, "bullish"), (29.99, "bullish"), (30, "neutral"), (50, "neutral"),
     (70, "neutral"), (70.01, "bearish"), (95, "bearish")],
)  # fmt: skip
def test_classify_signal_thresholds(rsi, signal):
    assert classify_signal(rsi) == signal


def test_feature_set_detection():
    assert detect_feature_set(synthetic_prices(ohlcv=False)) is FeatureSet.CLOSE
    no_volume = synthetic_prices().drop(columns="Volume")
    assert detect_feature_set(no_volume) is FeatureSet.OHLC
    assert detect_feature_set(synthetic_prices()) is FeatureSet.OHLCV


def test_compute_features_keeps_the_latest_bar():
    prices = synthetic_prices()
    features, _ = compute_features(prices)
    assert features.index[-1] == prices.index[-1]
    assert pd.isna(features["rsi_target"].iloc[-1])


def test_flat_bollinger_bands_fall_back_to_mid_band():
    prices = pd.DataFrame({"Close": [100.0] * 150}, index=pd.bdate_range("2025-01-01", periods=150))
    features, _ = compute_features(prices)
    assert (features["bb_pct"] == 0.5).all()


def test_analyze_close_only_source():
    snapshot = analyze(synthetic_prices(ohlcv=False), symbol="XU100", source="TCMB EVDS")
    assert snapshot.feature_set is FeatureSet.CLOSE
    assert snapshot.atr is None
    assert snapshot.obv is None
    assert snapshot.signal in {"bullish", "bearish", "neutral"}
    assert snapshot.as_of == synthetic_prices().index[-1].date()


def test_analyze_full_ohlcv_source():
    snapshot = analyze(synthetic_prices(), symbol="THYAO", source="upload")
    assert snapshot.feature_set is FeatureSet.OHLCV
    assert snapshot.atr is not None
    assert snapshot.obv is not None
    assert snapshot.price_above_ema34 == (snapshot.price > snapshot.ema34)
    levels = snapshot.rsi_levels
    assert levels["rsi_range_low"] <= levels["rsi_fib_382"] <= levels["rsi_range_high"]


def test_analyze_requires_enough_history():
    with pytest.raises(ProviderError, match="Not enough history"):
        analyze(synthetic_prices(130), symbol="THYAO", source="upload")


def test_model_cache_round_trip(tmp_path):
    cache = ModelCache(tmp_path, max_age_hours=1)
    analyze(synthetic_prices(), symbol="THYAO", source="s", cache=cache)
    assert len(list(tmp_path.glob("*.joblib"))) == 1

    analyze(synthetic_prices(), symbol="THYAO", source="s", cache=cache)
    assert len(list(tmp_path.glob("*.joblib"))) == 1


def test_model_cache_ignores_expired_and_corrupt_entries(tmp_path):
    cache = ModelCache(tmp_path, max_age_hours=1)
    cache.path_for("k").write_bytes(b"not a model")
    assert cache.load("k") is None
    assert ModelCache(tmp_path, max_age_hours=1e-9).load("k") is None


def test_channel_position_uses_close_when_no_range():
    frame = pd.DataFrame(
        {"Close": list(range(1, 21))}, index=pd.bdate_range("2026-01-01", periods=20)
    )
    assert channel_position(frame) == "top"
