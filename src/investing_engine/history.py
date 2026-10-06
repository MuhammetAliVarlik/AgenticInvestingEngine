"""SQLite-backed prediction history.

Only derived analysis output is stored (signal, risk score, indicator values
and a short summary) - never third-party content such as headlines or
uploaded documents.

Rows are either shared (``owner`` is NULL: analyses of public data, visible to
every user) or private (analyses built on a caller's own uploads, visible to
that caller only). Owners are stored as salted hashes, never in clear, and
the owner column is never returned to callers. Databases created before the
column existed are migrated in place.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from investing_engine.observability.tracing import hash_principal

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    signal TEXT,
    risk_score REAL,
    ema34 REAL,
    ema89 REAL,
    price_at_prediction REAL,
    rsi_divergence TEXT,
    summary_text TEXT,
    owner TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions(ticker, timestamp);
"""

# Rows a viewer may see (shared or their own), without the owner hash itself.
_SELECT_VISIBLE = """
SELECT id, ticker, timestamp, signal, risk_score, ema34, ema89, price_at_prediction,
       rsi_divergence, summary_text, owner IS NOT NULL AS private
FROM predictions
WHERE ticker = ? AND (owner IS NULL OR owner = ?)
"""

_MAX_SUMMARY_CHARS = 1500


class HistoryStore:
    def __init__(self, db_path: str | Path, *, salt: str) -> None:
        self._db_path = Path(db_path)
        self._salt = salt

    def _owner(self, principal: str) -> str:
        return hash_principal(principal, self._salt)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        with self._connect() as conn:
            existing = {row["name"] for row in conn.execute("PRAGMA table_info(predictions)")}
            if existing and "owner" not in existing:
                conn.execute("ALTER TABLE predictions ADD COLUMN owner TEXT")
            conn.executescript(_SCHEMA)

    def save(
        self,
        *,
        symbol: str,
        signal: str | None,
        risk_score: float | None,
        ema34: float | None,
        ema89: float | None,
        price: float | None,
        divergence: bool | None,
        summary: str | None,
        owner: str | None = None,
    ) -> None:
        """Persist one prediction, shared or private to ``owner``.

        Never raises: history is best-effort and must not break a response
        that has already been produced.
        """
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO predictions
                        (ticker, timestamp, signal, risk_score, ema34, ema89,
                         price_at_prediction, rsi_divergence, summary_text, owner)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol.upper(),
                        datetime.now(timezone.utc).isoformat(),
                        signal,
                        risk_score,
                        ema34,
                        ema89,
                        price,
                        None if divergence is None else ("Yes" if divergence else "No"),
                        (summary or "")[:_MAX_SUMMARY_CHARS] or None,
                        None if owner is None else self._owner(owner),
                    ),
                )
        except sqlite3.Error:
            logger.exception("Failed to save prediction history for %s", symbol)

    def recent(self, symbol: str, *, viewer: str, limit: int = 5) -> list[dict[str, Any]]:
        """Most recent predictions ``viewer`` may see for ``symbol``, newest first."""
        return self._query(
            _SELECT_VISIBLE + "ORDER BY timestamp DESC, id DESC LIMIT ?",
            symbol,
            viewer,
            limit,
        )

    def timeline(self, symbol: str, *, viewer: str, limit: int = 500) -> list[dict[str, Any]]:
        """Capped history ``viewer`` may see for ``symbol``, oldest first (for charting)."""
        return self._query(
            _SELECT_VISIBLE + "ORDER BY timestamp ASC, id ASC LIMIT ?",
            symbol,
            viewer,
            limit,
        )

    def _query(self, sql: str, symbol: str, viewer: str, limit: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(sql, (symbol.upper(), self._owner(viewer), limit)).fetchall()
        return [{**dict(row), "private": bool(row["private"])} for row in rows]
