import sqlite3

import db


def test_init_db_creates_table(db_path):
    db.init_db(db_path)
    assert db.get_history("THYAO.IS", db_path=db_path) == []


def test_save_and_get_history_roundtrip(db_path):
    db.init_db(db_path)
    db.save_prediction(
        ticker="thyao.is", signal="bullish", risk_score=4.0,
        ema34=100.0, ema89=95.0, price_at_prediction=101.5,
        rsi_divergence="No", summary_text="looks good",
        db_path=db_path,
    )
    rows = db.get_history("THYAO.IS", db_path=db_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["ticker"] == "THYAO.IS"  # normalized to uppercase regardless of input case
    assert row["signal"] == "bullish"
    assert row["risk_score"] == 4.0
    assert row["summary_text"] == "looks good"


def test_get_history_orders_newest_first(db_path):
    db.init_db(db_path)
    for i, signal in enumerate(["neutral", "bullish", "bearish"]):
        db.save_prediction(
            ticker="THYAO.IS", signal=signal, risk_score=float(i),
            ema34=None, ema89=None, price_at_prediction=None,
            rsi_divergence=None, summary_text=f"row {i}",
            db_path=db_path,
        )
    rows = db.get_history("THYAO.IS", limit=5, db_path=db_path)
    assert [r["signal"] for r in rows] == ["bearish", "bullish", "neutral"]


def test_get_history_respects_limit(db_path):
    db.init_db(db_path)
    for i in range(5):
        db.save_prediction(
            ticker="THYAO.IS", signal="neutral", risk_score=float(i),
            ema34=None, ema89=None, price_at_prediction=None,
            rsi_divergence=None, summary_text=f"row {i}",
            db_path=db_path,
        )
    rows = db.get_history("THYAO.IS", limit=2, db_path=db_path)
    assert len(rows) == 2


def test_get_all_history_orders_oldest_first(db_path):
    db.init_db(db_path)
    for i, signal in enumerate(["neutral", "bullish", "bearish"]):
        db.save_prediction(
            ticker="THYAO.IS", signal=signal, risk_score=float(i),
            ema34=None, ema89=None, price_at_prediction=None,
            rsi_divergence=None, summary_text=f"row {i}",
            db_path=db_path,
        )
    rows = db.get_all_history("THYAO.IS", db_path=db_path)
    assert [r["signal"] for r in rows] == ["neutral", "bullish", "bearish"]


def test_get_history_empty_ticker_returns_empty_list(db_path):
    db.init_db(db_path)
    assert db.get_history("NOPE.IS", db_path=db_path) == []


def test_history_is_isolated_per_ticker(db_path):
    db.init_db(db_path)
    db.save_prediction(
        ticker="AAA.IS", signal="bullish", risk_score=1.0,
        ema34=None, ema89=None, price_at_prediction=None,
        rsi_divergence=None, summary_text="a",
        db_path=db_path,
    )
    db.save_prediction(
        ticker="BBB.IS", signal="bearish", risk_score=9.0,
        ema34=None, ema89=None, price_at_prediction=None,
        rsi_divergence=None, summary_text="b",
        db_path=db_path,
    )
    assert len(db.get_history("AAA.IS", db_path=db_path)) == 1
    assert len(db.get_history("BBB.IS", db_path=db_path)) == 1


def test_save_prediction_swallows_failure_instead_of_raising(db_path, monkeypatch):
    db.init_db(db_path)

    def raise_error(path):
        raise sqlite3.OperationalError("simulated failure")

    monkeypatch.setattr(db, "_connect", raise_error)

    # Must not raise despite the underlying connection failing.
    db.save_prediction(
        ticker="THYAO.IS", signal="bullish", risk_score=1.0,
        ema34=None, ema89=None, price_at_prediction=None,
        rsi_divergence=None, summary_text="x",
        db_path=db_path,
    )
