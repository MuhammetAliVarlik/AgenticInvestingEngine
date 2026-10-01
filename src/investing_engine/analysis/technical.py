"""Technical snapshot: indicators plus a next-period RSI forecast.

A RandomForest regressor is trained on the instrument's own history to
predict the next bar's RSI, and the prediction is mapped to a mean-reversion
signal. Models trained on public sources are cached on disk for a limited
time; models trained on user uploads are never persisted.
"""

from __future__ import annotations

import hashlib
import os
import time
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor

from investing_engine.analysis.indicators import (
    TARGET_COLUMN,
    FeatureSet,
    channel_position,
    classify_signal,
    compute_features,
    momentum_divergence,
    rsi_fibonacci_levels,
)
from investing_engine.providers.base import ProviderError

MIN_TRAINING_ROWS = 60


class TechnicalSnapshot(BaseModel):
    symbol: str
    as_of: date
    source: str
    feature_set: FeatureSet
    price: float
    rsi: float
    predicted_next_rsi: float
    signal: str = Field(description="bullish | bearish | neutral")
    ema34: float
    ema89: float
    price_above_ema34: bool
    momentum_divergence: bool
    channel_position: str = Field(description="bottom | middle | top")
    macd: float
    macd_signal: float
    bb_pct: float
    atr: float | None = None
    adx: float | None = None
    stoch_k: float | None = None
    stoch_d: float | None = None
    obv: int | None = None
    rsi_levels: dict[str, float]


class ModelCache:
    """Disk cache for trained models, keyed by symbol, source and feature set."""

    def __init__(self, directory: str | Path, *, max_age_hours: float) -> None:
        self._directory = Path(directory)
        self._max_age_seconds = max_age_hours * 3600

    def path_for(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self._directory / f"rsi_model_{digest}.joblib"

    def load(self, key: str) -> RandomForestRegressor | None:
        path = self.path_for(key)
        if not path.exists() or time.time() - path.stat().st_mtime > self._max_age_seconds:
            return None
        try:
            model = joblib.load(path)
        except Exception:  # corrupt or incompatible cache entry: retrain
            return None
        return model if isinstance(model, RandomForestRegressor) else None

    def save(self, key: str, model: RandomForestRegressor) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        tmp = self.path_for(key).with_suffix(".tmp")
        joblib.dump(model, tmp)
        os.replace(tmp, self.path_for(key))


def _train(features: pd.DataFrame, columns: tuple[str, ...]) -> RandomForestRegressor:
    training = features.dropna(subset=[TARGET_COLUMN])
    if len(training) < MIN_TRAINING_ROWS:
        raise ProviderError(
            f"Not enough history to train the forecaster: {len(training)} rows, "
            f"need {MIN_TRAINING_ROWS}"
        )
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(training[list(columns)], training[TARGET_COLUMN])
    return model


def _optional(row: pd.Series, column: str, digits: int) -> float | None:
    value = row.get(column)
    return None if value is None or pd.isna(value) else round(float(value), digits)


def analyze(
    prices: pd.DataFrame,
    *,
    symbol: str,
    source: str,
    cache: ModelCache | None = None,
) -> TechnicalSnapshot:
    """Compute the technical snapshot for a validated price frame.

    Pass ``cache=None`` for user-supplied data so no model derived from it is
    written to disk.
    """
    features, feature_set = compute_features(prices)
    columns = feature_set.columns

    cache_key = f"{symbol}|{source}|{feature_set.value}"
    model = cache.load(cache_key) if cache else None
    if model is None:
        model = _train(features, columns)
        if cache:
            cache.save(cache_key, model)

    latest = features.iloc[-1]
    predicted = float(model.predict(features[list(columns)].iloc[[-1]])[0])
    obv = latest.get("obv")

    return TechnicalSnapshot(
        symbol=symbol,
        as_of=pd.Timestamp(features.index[-1]).date(),
        source=source,
        feature_set=feature_set,
        price=round(float(latest["Close"]), 2),
        rsi=round(float(latest["rsi"]), 2),
        predicted_next_rsi=round(predicted, 2),
        signal=classify_signal(predicted),
        ema34=round(float(latest["ema34"]), 2),
        ema89=round(float(latest["ema89"]), 2),
        price_above_ema34=bool(latest["Close"] > latest["ema34"]),
        momentum_divergence=momentum_divergence(features),
        channel_position=channel_position(features),
        macd=round(float(latest["macd"]), 4),
        macd_signal=round(float(latest["macd_signal"]), 4),
        bb_pct=round(float(latest["bb_pct"]), 4),
        atr=_optional(latest, "atr", 4),
        adx=_optional(latest, "adx", 2),
        stoch_k=_optional(latest, "stoch_k", 2),
        stoch_d=_optional(latest, "stoch_d", 2),
        obv=None if obv is None or pd.isna(obv) else int(obv),
        rsi_levels=rsi_fibonacci_levels(features),
    )
