"""SQLite-backed prediction history.

Only derived analysis output is stored (signal, risk score, indicator values
and a short summary) - never third-party content such as headlines or
uploaded documents. The schema is unchanged from v1 so existing databases
keep working.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    summary_text TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions(ticker, timestamp);
"""

_MAX_SUMMARY_CHARS = 1500


class HistoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)

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
    ) -> None:
        """Persist one prediction. Never raises: history is best-effort and
        must not break a response that has already been produced."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO predictions
                        (ticker, timestamp, signal, risk_score, ema34, ema89,
                         price_at_prediction, rsi_divergence, summary_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    ),
                )
        except sqlite3.Error:
            logger.exception("Failed to save prediction history for %s", symbol)

    def recent(self, symbol: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """Most recent predictions for ``symbol``, newest first."""
        return self._query(
            "SELECT * FROM predictions WHERE ticker = ? ORDER BY timestamp DESC, id DESC LIMIT ?",
            symbol,
            limit,
        )

    def timeline(self, symbol: str, *, limit: int = 500) -> list[dict[str, Any]]:
        """Capped history for ``symbol``, oldest first (for charting)."""
        return self._query(
            "SELECT * FROM predictions WHERE ticker = ? ORDER BY timestamp ASC, id ASC LIMIT ?",
            symbol,
            limit,
        )

    def _query(self, sql: str, symbol: str, limit: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(sql, (symbol.upper(), limit)).fetchall()
        return [dict(row) for row in rows]
