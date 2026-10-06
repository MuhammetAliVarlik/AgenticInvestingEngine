"""Prediction history: shared rows, owner-private rows and in-place migration."""

import sqlite3

from investing_engine.history import HistoryStore


def _store(tmp_path):
    store = HistoryStore(tmp_path / "h.db", salt="test-salt")
    store.init()
    return store


def _save(store, owner=None, risk=4.0):
    store.save(
        symbol="THYAO",
        signal="neutral",
        risk_score=risk,
        ema34=1.0,
        ema89=1.0,
        price=100.0,
        divergence=False,
        summary="summary",
        owner=owner,
    )


def test_private_rows_are_visible_to_their_owner_only(tmp_path):
    store = _store(tmp_path)
    _save(store, risk=1.0)  # shared
    _save(store, owner="alice@example.com", risk=2.0)

    alice = store.timeline("THYAO", viewer="alice@example.com")
    bob = store.timeline("THYAO", viewer="bob@example.com")

    assert [r["risk_score"] for r in alice] == [1.0, 2.0]
    assert [r["private"] for r in alice] == [False, True]
    assert [r["risk_score"] for r in bob] == [1.0]
    assert [r["risk_score"] for r in store.recent("THYAO", viewer="bob@example.com")] == [1.0]


def test_owner_is_stored_hashed_and_never_returned(tmp_path):
    store = _store(tmp_path)
    _save(store, owner="alice@example.com")

    raw = sqlite3.connect(tmp_path / "h.db").execute("SELECT owner FROM predictions").fetchone()[0]
    assert raw
    assert "alice" not in raw
    assert "owner" not in store.timeline("THYAO", viewer="alice@example.com")[0]


def test_existing_database_is_migrated_in_place(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT NOT NULL, "
        "timestamp TEXT NOT NULL, signal TEXT, risk_score REAL, ema34 REAL, ema89 REAL, "
        "price_at_prediction REAL, rsi_divergence TEXT, summary_text TEXT)"
    )
    conn.execute(
        "INSERT INTO predictions (ticker, timestamp, risk_score) VALUES ('XU100', '2026-01-01', 3)"
    )
    conn.commit()
    conn.close()

    store = HistoryStore(path, salt="test-salt")
    store.init()
    rows = store.timeline("XU100", viewer="anyone")
    assert [(r["risk_score"], r["private"]) for r in rows] == [(3.0, False)]
