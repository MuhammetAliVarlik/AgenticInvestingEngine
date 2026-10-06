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

from investing_engine.analysis.indicators import compute_features
from investing_engine.analysis.technical import ModelCache, TechnicalSnapshot, analyze
from investing_engine.config import Settings
from investing_engine.guardrails.classifier import (
    GroqPromptGuard,
    InjectionClassifier,
    classify_chunks,
)
from investing_engine.guardrails.injection import new_boundary, quarantine, spotlight
from investing_engine.history import HistoryStore
from investing_engine.ingestion.documents import (
    DocumentError,
    ExtractedDocument,
    ExtractionLimits,
    extract_document,
)
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
DOCUMENT_UPLOAD_KIND = "document"
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


USER_DOCUMENT_ATTRIBUTION = {
    "name": "User-supplied document",
    "url": "",
    "attribution": "Disclosure document supplied by the user.",
}


@dataclass(frozen=True, slots=True)
class StoredDocument:
    """An extracted document plus the guarded text the model is allowed to see."""

    document: ExtractedDocument
    guarded_text: str
    injection_flags: tuple[str, ...]
    truncated: bool


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
        classifier: InjectionClassifier | None = None,
    ) -> None:
        self.settings = settings
        self.evds = evds
        self.gdelt = gdelt
        self.yahoo = yahoo
        self.uploads = uploads
        self.history = history
        self.model_cache = model_cache
        self.classifier = classifier
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
        classifier = None
        if settings.enable_prompt_guard and settings.groq_api_key is not None:
            classifier = GroqPromptGuard(
                api_key=settings.groq_api_key.get_secret_value(),
                model=settings.prompt_guard_model,
                timeout=settings.http_timeout_seconds,
            )
        history = HistoryStore(settings.db_path, salt=settings.telemetry_salt.get_secret_value())
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
            classifier=classifier,
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

    def chart_series(
        self, symbol: str, *, owner: str, dataset_id: str | None = None, bars: int = 120
    ) -> pd.DataFrame:
        """Recent price and indicator series for charts (never sent to the model)."""
        instrument = resolve(symbol)
        prices, _, _ = self._prices(instrument, owner=owner, dataset_id=dataset_id)
        features, _ = compute_features(prices)
        columns = ["Close", "ema34", "ema89", "bb_high", "bb_low", "rsi"]
        return features[columns].tail(bars)

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
            headlines = self.gdelt.headlines(instrument, days=days, limit=limit)
            lines: list[str] = []
            flagged: set[str] = set()
            for item in headlines:
                title, result = quarantine(item["title"])
                flagged.update(result.categories)
                date = (item.get("published_at") or "")[:10]
                lines.append(f"- {date} | {item.get('outlet') or 'unknown'} | {title}")
            return {
                "symbol": instrument.symbol,
                "window_days": days,
                "average_tone": self.gdelt.average_tone(instrument, days=days),
                "headline_count": len(headlines),
                "headlines": spotlight(
                    self._classify("\n".join(lines) or "(no headlines)", flagged),
                    source="GDELT",
                    boundary=new_boundary(),
                ),
                "injection_flags": sorted(flagged),
                "source": _source_dict(GDELT_SOURCE),
            }

        return self._news_memo.get_or_set(f"{instrument.symbol}:{days}:{limit}", fetch)

    # --- Disclosure documents ----------------------------------------------------

    def register_document(
        self, *, owner: str, symbol: str, raw: bytes, filename: str
    ) -> dict[str, Any]:
        """Extract text (with OCR where needed) and store it for this owner."""
        instrument = resolve(symbol)
        limits = ExtractionLimits(
            max_bytes=self.settings.max_document_bytes,
            max_pages=self.settings.max_document_pages,
            ocr_timeout_seconds=self.settings.ocr_timeout_seconds,
        )
        try:
            document = extract_document(raw, filename=filename, limits=limits)
        except DocumentError as exc:
            raise ProviderError(str(exc)) from exc
        if not document.text:
            raise ProviderError("No text could be extracted from the document")

        # Guard once at upload time so the optional classifier never runs twice
        # for the same document.
        limit = self.settings.max_document_chars_for_model
        cleaned, scan = quarantine(document.text[:limit])
        flags = set(scan.categories)
        cleaned = self._classify(cleaned, flags)
        stored = StoredDocument(
            document=document,
            guarded_text=cleaned,
            injection_flags=tuple(sorted(flags)),
            truncated=len(document.text) > limit,
        )
        upload = self.uploads.put(
            owner=owner, kind=DOCUMENT_UPLOAD_KIND, label=instrument.symbol, payload=stored
        )
        return {
            "document_id": upload.id,
            "symbol": instrument.symbol,
            **document.summary(),
            "injection_flags": list(stored.injection_flags),
        }

    def _classify(self, text: str, flags: set[str]) -> str:
        """Run the optional model classifier, recording a flag if it quarantines anything."""
        if self.classifier is None:
            return text
        cleaned, flagged_chunks = classify_chunks(
            self.classifier, text, threshold=self.settings.prompt_guard_threshold
        )
        if flagged_chunks:
            flags.add("classifier")
        return cleaned

    def disclosure(
        self, symbol: str, *, owner: str, document_id: str | None = None
    ) -> dict[str, Any]:
        """Guarded document text for the disclosure analyst."""
        instrument = resolve(symbol)
        if not document_id:
            raise ProviderError(f"No disclosure document was provided for {instrument.symbol}.")
        try:
            upload = self.uploads.get(document_id, owner=owner, kind=DOCUMENT_UPLOAD_KIND)
        except UploadNotFoundError:
            raise ProviderError("Document not found or expired") from None
        if upload.label != instrument.symbol:
            raise ProviderError(f"Document belongs to {upload.label}, not {instrument.symbol}")

        stored: StoredDocument = upload.payload
        return {
            "symbol": instrument.symbol,
            "document": stored.document.summary(),
            "truncated": stored.truncated,
            "content": spotlight(
                stored.guarded_text,
                source=f"uploaded document {instrument.symbol}",
                boundary=new_boundary(),
            ),
            "injection_flags": list(stored.injection_flags),
            "source": USER_DOCUMENT_ATTRIBUTION,
        }

    # --- History -------------------------------------------------------------------

    def recent_history(self, symbol: str, *, viewer: str, limit: int = 5) -> list[dict[str, Any]]:
        return self.history.recent(
            resolve(symbol).symbol, viewer=viewer, limit=max(1, min(limit, 50))
        )

    def close(self) -> None:
        if self.evds is not None:
            self.evds.close()
        self.gdelt.close()
