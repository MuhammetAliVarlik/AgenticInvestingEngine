"""GDELT DOC 2.0 news provider.

Terms: GDELT datasets are available for unlimited and unrestricted use,
including commercial use and redistribution, provided the GDELT Project is
cited with a link to https://www.gdeltproject.org/.

Only article metadata (headline, outlet, URL, timestamp) and GDELT's own
aggregate tone are used - article bodies are never fetched or stored.
Queries are built exclusively from the instrument allowlist, never from raw
user input.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from investing_engine.providers.base import ProviderError, SourceInfo
from investing_engine.universe import Instrument

GDELT_SOURCE = SourceInfo(
    name="The GDELT Project",
    url="https://www.gdeltproject.org/",
    terms="Unlimited and unrestricted use, including commercial use and "
    "redistribution, with citation of the GDELT Project.",
    attribution="News metadata: The GDELT Project (https://www.gdeltproject.org/).",
    deployable=True,
)

_MAX_TITLE_CHARS = 300
_MAX_ARTICLES = 25
# GDELT asks clients to send at most one request every five seconds.
DEFAULT_MIN_INTERVAL_SECONDS = 5.5


class GdeltNewsProvider:
    source = GDELT_SOURCE

    def __init__(
        self,
        *,
        base_url: str,
        timeout: float,
        transport: httpx.BaseTransport | None = None,
        min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
    ) -> None:
        self._min_interval = min_interval_seconds
        self._last_request = 0.0
        self._throttle = threading.Lock()
        # The endpoint is used as a full URL: httpx's base_url joining would
        # append a trailing slash that GDELT does not route.
        self._url = base_url
        self._client = httpx.Client(
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
            headers={"User-Agent": "investing-engine/2.0 (+https://github.com/MuhammetAliVarlik)"},
        )

    def close(self) -> None:
        self._client.close()

    def headlines(
        self, instrument: Instrument, *, days: int = 7, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Most recent headlines mentioning the instrument, newest first."""
        payload = self._get(
            {
                "query": instrument.news_query,
                "mode": "ArtList",
                "format": "json",
                "sort": "DateDesc",
                "timespan": f"{days}d",
                "maxrecords": str(min(limit, _MAX_ARTICLES)),
            }
        )
        articles = payload.get("articles") or []
        return [_to_headline(a) for a in articles if isinstance(a, dict) and a.get("url")][:limit]

    def average_tone(self, instrument: Instrument, *, days: int = 7) -> float | None:
        """Mean GDELT tone across the window (roughly -10 negative .. +10 positive)."""
        payload = self._get(
            {
                "query": instrument.news_query,
                "mode": "TimelineTone",
                "format": "json",
                "timespan": f"{days}d",
            }
        )
        values: list[float] = []
        for series in payload.get("timeline") or []:
            for point in series.get("data") or []:
                value = point.get("value")
                if isinstance(value, (int, float)):
                    values.append(float(value))
        if not values:
            return None
        return round(sum(values) / len(values), 3)

    def _wait_for_slot(self) -> None:
        """Block until the provider-wide request interval has elapsed."""
        with self._throttle:
            delay = self._last_request + self._min_interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last_request = time.monotonic()

    def _get(self, params: dict[str, str]) -> dict[str, Any]:
        self._wait_for_slot()
        try:
            response = self._client.get(self._url, params=params)
        except httpx.HTTPError as exc:
            raise ProviderError(f"GDELT request failed: {type(exc).__name__}") from exc
        if response.status_code == 429:
            raise ProviderError("GDELT rate limit reached; try again shortly")
        if response.status_code != 200:
            raise ProviderError(f"GDELT returned HTTP {response.status_code}")
        if not response.content.strip():
            return {}
        try:
            payload = response.json()
        except ValueError as exc:
            # GDELT answers malformed queries with a plain-text explanation.
            raise ProviderError("GDELT returned a non-JSON response") from exc
        if not isinstance(payload, dict):
            raise ProviderError("GDELT returned an unexpected payload")
        return payload


def _to_headline(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": str(article.get("title", "")).strip()[:_MAX_TITLE_CHARS],
        "outlet": article.get("domain"),
        "url": article.get("url"),
        "language": article.get("language"),
        "published_at": _parse_seendate(article.get("seendate")),
    }


def _parse_seendate(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return parsed.isoformat()
