import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "data/predictions.db")

SCHEMA = """
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


@contextmanager
def _connect(db_path: str):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    with _connect(db_path or DB_PATH) as conn:
        conn.executescript(SCHEMA)


def save_prediction(
    ticker: str,
    signal: str | None,
    risk_score: float | None,
    ema34: float | None,
    ema89: float | None,
    price_at_prediction: float | None,
    rsi_divergence: str | None,
    summary_text: str | None,
    db_path: str | None = None,
) -> None:
    """Writes one prediction row. Never raises - a history-write failure
    must not break the API response that already went out to the caller."""
    try:
        with _connect(db_path or DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO predictions
                    (ticker, timestamp, signal, risk_score, ema34, ema89,
                     price_at_prediction, rsi_divergence, summary_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker.upper(),
                    datetime.now(timezone.utc).isoformat(),
                    signal,
                    risk_score,
                    ema34,
                    ema89,
                    price_at_prediction,
                    rsi_divergence,
                    summary_text,
                ),
            )
    except Exception:
        logger.exception("Failed to save prediction history for %s", ticker)


def get_history(ticker: str, limit: int = 5, db_path: str | None = None) -> list[dict]:
    """Most recent `limit` predictions for a ticker, newest first."""
    with _connect(db_path or DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT * FROM predictions
            WHERE ticker = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (ticker.upper(), limit),
        ).fetchall()
        return [dict(row) for row in rows]


def get_all_history(ticker: str, db_path: str | None = None, limit: int = 500) -> list[dict]:
    """Full (capped) history for a ticker, oldest first - convenient for
    plotting a timeline."""
    with _connect(db_path or DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT * FROM predictions
            WHERE ticker = ?
            ORDER BY timestamp ASC, id ASC
            LIMIT ?
            """,
            (ticker.upper(), limit),
        ).fetchall()
        return [dict(row) for row in rows]
