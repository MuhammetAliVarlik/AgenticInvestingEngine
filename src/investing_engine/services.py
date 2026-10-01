"""Application services: the single entry point tools and APIs call into.

``MarketData`` wires providers, caches, uploads and history together and
encodes the data-licensing policy in one place: which source may serve which
instrument, and what attribution travels with every result.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, TypeVar

import pandas as pd
from cachetools import TTLCache

from investing_engine.analysis.technical import ModelCache, TechnicalSnapshot, analyze
from investing_engine.config import Settings
from investing_engine.history import HistoryStore
from investing_engine.providers.base import PriceProvider, ProviderError, SourceInfo
from investing_engine.providers.csv_upload import USER_UPLOAD_SOURCE, parse_ohlcv_csv
from investing_engine.providers.evds import (
    EVDS_SOURCE,
    EvdsClient,
    EvdsIndexPriceProvider,
    macro_snapshot,
)
from investing_engine.providers.gdelt import GDELT_SOURCE, GdeltNewsProvider
from investing_engine.providers.yahoo import YahooPriceProvider
from investing_engine.universe import Instrument, InstrumentKind, resolve
from investing_engine.uploads import UploadNotFoundError, UploadStore

T = TypeVar("T")

PRICE_UPLOAD_KIND = "prices"
_NEWS_TTL_SECONDS = 15 * 60
_MACRO_TTL_SECONDS = 60 * 60


@dataclass
class _Memo:
    """Small thread-safe TTL memoiser for upstream calls with strict rate limits."""

    ttl: int
    _cache: TTLCache[str, Any] = field(init=False)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self._cache = TTLCache(maxsize=256, ttl=self.ttl)

    def get_or_set(self, key: str, factory: Callable[[], T]) -> T:
        with self._lock:
            if key in self._cache:
                return self._cache[key]  # type: ignore[no-any-return]
        value = factory()
        with self._lock:
            self._cache[key] = value
        return value


def _source_dict(source: SourceInfo) -> dict[str, Any]:
    return {k: v for k, v in asdict(source).items() if k in ("name", "url", "attribution")}


class MarketData:
    def __init__(
        self,
        settings: Settings,
        *,
        evds: EvdsClient | None,
        gdelt: GdeltNewsProvider,
        yahoo: YahooPriceProvider | None,
        uploads: UploadStore,
        history: HistoryStore,
        model_cache: ModelCache,
    ) -> None:
        self.settings = settings
        self.evds = evds
        self.gdelt = gdelt
        self.yahoo = yahoo
        self.uploads = uploads
        self.history = history
        self.model_cache = model_cache
        self._news_memo = _Memo(_NEWS_TTL_SECONDS)
        self._macro_memo = _Memo(_MACRO_TTL_SECONDS)

    @classmethod
    def from_settings(cls, settings: Settings) -> MarketData:
        evds = None
        if settings.evds_api_key is not None:
            evds = EvdsClient(
                settings.evds_api_key.get_secret_value(),
                base_url=settings.evds_base_url,
                timeout=settings.http_timeout_seconds,
            )
        history = HistoryStore(settings.db_path)
        history.init()
        return cls(
            settings,
            evds=evds,
            gdelt=GdeltNewsProvider(
                base_url=settings.gdelt_base_url, timeout=settings.http_timeout_seconds
            ),
            yahoo=YahooPriceProvider() if settings.enable_yfinance else None,
            uploads=UploadStore(),
            history=history,
            model_cache=ModelCache(settings.model_dir, max_age_hours=settings.model_max_age_hours),
        )

    # --- Sources -------------------------------------------------------------

    def active_sources(self) -> list[SourceInfo]:
        sources = [GDELT_SOURCE, USER_UPLOAD_SOURCE]
        if self.evds is not None:
            sources.insert(0, EVDS_SOURCE)
        if self.yahoo is not None:
            sources.append(self.yahoo.source)
        return sources

    # --- Prices & technicals -------------------------------------------------

    def register_prices(self, *, owner: str, symbol: str, raw: bytes) -> dict[str, Any]:
        """Validate an uploaded OHLCV CSV and store it for this owner."""
        instrument = resolve(symbol)
        frame = parse_ohlcv_csv(
            raw,
            symbol=instrument.symbol,
            max_bytes=self.settings.max_upload_bytes,
            max_rows=self.settings.max_csv_rows,
        )
        upload = self.uploads.put(
            owner=owner, kind=PRICE_UPLOAD_KIND, label=instrument.symbol, payload=frame
        )
        return {
            "dataset_id": upload.id,
            "symbol": instrument.symbol,
            "rows": len(frame),
            "start": frame.index[0].date().isoformat(),
            "end": frame.index[-1].date().isoformat(),
            "columns": list(frame.columns),
        }

    def _prices(
        self, instrument: Instrument, *, owner: str, dataset_id: str | None
    ) -> tuple[pd.DataFrame, SourceInfo, bool]:
        """Return (prices, source, cacheable) according to the licensing policy."""
        if dataset_id:
            try:
                upload = self.uploads.get(dataset_id, owner=owner, kind=PRICE_UPLOAD_KIND)
            except UploadNotFoundError:
                raise ProviderError("Price dataset not found or expired") from None
            if upload.label != instrument.symbol:
                raise ProviderError(f"Dataset belongs to {upload.label}, not {instrument.symbol}")
            return upload.payload, USER_UPLOAD_SOURCE, False

        provider: PriceProvider | None = None
        if instrument.evds_series and self.evds is not None:
            provider = EvdsIndexPriceProvider(self.evds)
        elif self.yahoo is not None:
            provider = self.yahoo

        if provider is None:
            if instrument.kind is InstrumentKind.EQUITY:
                raise ProviderError(
                    f"No licensed price source is configured for {instrument.symbol}. "
                    "Upload your own OHLCV CSV and pass its dataset_id."
                )
            raise ProviderError("EVDS is not configured (set EVDS_API_KEY)")

        frame = provider.get_history(instrument.symbol, lookback_days=self.settings.lookback_days)
        return frame, provider.source, True

    def technical(
        self, symbol: str, *, owner: str, dataset_id: str | None = None
    ) -> TechnicalSnapshot:
        instrument = resolve(symbol)
        prices, source, cacheable = self._prices(instrument, owner=owner, dataset_id=dataset_id)
        return analyze(
            prices,
            symbol=instrument.symbol,
            source=source.name,
            cache=self.model_cache if cacheable else None,
        )

    # --- Macro ---------------------------------------------------------------------

    def macro(self) -> dict[str, Any]:
        if self.evds is None:
            raise ProviderError("EVDS is not configured (set EVDS_API_KEY)")
        evds = self.evds
        series = self._macro_memo.get_or_set("macro", lambda: macro_snapshot(evds))
        return {"series": series, "source": _source_dict(EVDS_SOURCE)}

    # --- News ------------------------------------------------------------------------

    def news(self, symbol: str, *, days: int = 7, limit: int = 10) -> dict[str, Any]:
        instrument = resolve(symbol)
        days = max(1, min(days, 30))
        limit = max(1, min(limit, 20))

        def fetch() -> dict[str, Any]:
            return {
                "symbol": instrument.symbol,
                "window_days": days,
                "average_tone": self.gdelt.average_tone(instrument, days=days),
                "headlines": self.gdelt.headlines(instrument, days=days, limit=limit),
                "source": _source_dict(GDELT_SOURCE),
            }

        return self._news_memo.get_or_set(f"{instrument.symbol}:{days}:{limit}", fetch)

    # --- History -------------------------------------------------------------------

    def recent_history(self, symbol: str, *, limit: int = 5) -> list[dict[str, Any]]:
        return self.history.recent(resolve(symbol).symbol, limit=max(1, min(limit, 50)))

    def close(self) -> None:
        if self.evds is not None:
            self.evds.close()
        self.gdelt.close()
