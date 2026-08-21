from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import cache
import db
import main


@pytest.fixture(autouse=True)
def _clear_cache():
    cache._cache.clear()
    yield
    cache._cache.clear()


@pytest.fixture
def client(db_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", db_path)
    with TestClient(main.app) as c:
        yield c


def _fake_pipeline_result(output="REPORT TEXT", symbol="THYAO.IS"):
    return {
        "output": output,
        "raw": {
            "technical_results": [
                {"symbol": symbol, "signal": "bullish", "ema34": 100.0, "ema89": 95.0,
                 "price": 105.0, "rsi_fibo_divergence": "No"}
            ],
            "technical_text": "technical text",
            "news_text": "Risk Score: 5/10\nReasoning: calm",
            "pdp_text": "pdp text",
        },
        "stages": {"gathering_seconds": 1.0, "synthesis_seconds": 2.0},
    }


def test_health_check(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"message": "🎉 FastAPI working!"}


def test_stock_price_route_returns_tool_result(client, monkeypatch):
    monkeypatch.setattr(main, "rsi_predictor", lambda ticker: {"symbol": ticker, "signal": "neutral"})
    resp = client.get("/stock_price", params={"ticker": "THYAO.IS"})
    assert resp.status_code == 200
    assert resp.json() == {"symbol": "THYAO.IS", "signal": "neutral"}


def test_get_insights_happy_path(client, monkeypatch):
    async def fake_run_pipeline(ticker_list, tickers=None, history_context=""):
        return _fake_pipeline_result()

    monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)

    resp = client.get("/get_insights", params={"ticker_list": "THYAO.IS"})

    assert resp.status_code == 200
    assert resp.json() == {"output": "REPORT TEXT"}


def test_get_insights_debug_flag_includes_stages(client, monkeypatch):
    async def fake_run_pipeline(ticker_list, tickers=None, history_context=""):
        return _fake_pipeline_result()

    monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)

    resp = client.get("/get_insights", params={"ticker_list": "THYAO.IS", "debug": "true"})

    assert "stages" in resp.json()


def test_get_insights_writes_history_row(client, monkeypatch, db_path):
    async def fake_run_pipeline(ticker_list, tickers=None, history_context=""):
        return _fake_pipeline_result()

    monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)

    client.get("/get_insights", params={"ticker_list": "THYAO.IS"})

    rows = db.get_all_history("THYAO.IS", db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["signal"] == "bullish"
    assert rows[0]["risk_score"] == 5.0
    assert rows[0]["ema34"] == 100.0


def test_get_insights_cache_hit_skips_pipeline_and_duplicate_history_write(client, monkeypatch, db_path):
    call_count = {"n": 0}

    async def counting_pipeline(ticker_list, tickers=None, history_context=""):
        call_count["n"] += 1
        return _fake_pipeline_result(symbol="AAA.IS")

    monkeypatch.setattr(main, "run_pipeline", counting_pipeline)

    r1 = client.get("/get_insights", params={"ticker_list": "AAA.IS"})
    r2 = client.get("/get_insights", params={"ticker_list": "AAA.IS"})

    assert call_count["n"] == 1
    assert r1.json() == r2.json()

    rows = db.get_all_history("AAA.IS", db_path=db_path)
    assert len(rows) == 1


def test_get_insights_cache_key_normalizes_order_and_case(client, monkeypatch):
    call_count = {"n": 0}

    async def counting_pipeline(ticker_list, tickers=None, history_context=""):
        call_count["n"] += 1
        return _fake_pipeline_result()

    monkeypatch.setattr(main, "run_pipeline", counting_pipeline)

    client.get("/get_insights", params={"ticker_list": "aaa.is,bbb.is"})
    client.get("/get_insights", params={"ticker_list": "BBB.IS,AAA.IS"})

    assert call_count["n"] == 1


def test_history_route_returns_saved_rows(client, db_path):
    db.save_prediction(
        ticker="AAA.IS", signal="bullish", risk_score=1.0,
        ema34=None, ema89=None, price_at_prediction=None,
        rsi_divergence=None, summary_text="x", db_path=db_path,
    )

    resp = client.get("/history", params={"ticker": "AAA.IS"})

    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_history_route_empty_for_unknown_ticker(client):
    resp = client.get("/history", params={"ticker": "NOPE.IS"})
    assert resp.status_code == 200
    assert resp.json() == []


# --- _summarize_history / _humanize_days_ago ---------------------------

def test_summarize_history_formats_days_ago_and_risk():
    three_days_ago = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    rows = [{"timestamp": three_days_ago, "signal": "neutral", "risk_score": 4.0}]

    result = main._summarize_history(rows)

    assert "3 days ago" in result
    assert "neutral" in result
    assert "4/10" in result


def test_summarize_history_empty_rows_returns_empty_string():
    assert main._summarize_history([]) == ""


def test_summarize_history_multiple_rows_joined():
    now = datetime.now(timezone.utc)
    rows = [
        {"timestamp": (now - timedelta(days=3)).isoformat(), "signal": "neutral", "risk_score": 4.0},
        {"timestamp": (now - timedelta(days=7)).isoformat(), "signal": "bearish", "risk_score": 6.0},
    ]
    result = main._summarize_history(rows)
    assert "3 days ago" in result and "7 days ago" in result
    assert result.count(";") == 1


def test_get_insights_builds_multi_ticker_history_context(client, monkeypatch, db_path):
    three_days_ago = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    with db._connect(db_path) as conn:
        conn.executescript(db.SCHEMA)
        conn.execute(
            "INSERT INTO predictions (ticker, timestamp, signal, risk_score, ema34, ema89, "
            "price_at_prediction, rsi_divergence, summary_text) VALUES (?,?,?,?,?,?,?,?,?)",
            ("AAA.IS", three_days_ago, "neutral", 4.0, None, None, None, None, "old"),
        )

    captured = {}

    async def capturing_pipeline(ticker_list, tickers=None, history_context=""):
        captured["history_context"] = history_context
        captured["tickers"] = tickers
        return _fake_pipeline_result()

    monkeypatch.setattr(main, "run_pipeline", capturing_pipeline)

    client.get("/get_insights", params={"ticker_list": "AAA.IS,BBB.IS"})

    assert captured["tickers"] == ["AAA.IS", "BBB.IS"]
    assert "AAA.IS: 3 days ago" in captured["history_context"]
    assert "BBB.IS" not in captured["history_context"]  # no history yet for BBB.IS


# --- _split_report_by_ticker --------------------------------------------

def test_split_report_by_ticker_extracts_per_ticker_sections():
    report = (
        "intro\n"
        "📊 Ticker: AAA.IS\nsection A body\n"
        "📊 Ticker: BBB.IS\nsection B body"
    )
    sections = main._split_report_by_ticker(report)
    assert "section A body" in sections["AAA.IS"]
    assert "section B body" in sections["BBB.IS"]
    assert "section B" not in sections["AAA.IS"]


def test_split_report_by_ticker_handles_single_ticker_report():
    report = "📊 Ticker: THYAO.IS\nonly section"
    sections = main._split_report_by_ticker(report)
    assert sections == {"THYAO.IS": "📊 Ticker: THYAO.IS\nonly section"}


def test_split_report_by_ticker_strips_markdown_bold_from_header():
    """Regression test: a real captured run had the model bold the header
    ("**📊 Ticker: THYAO.IS**"), which without stripping produced the key
    "THYAO.IS**" - silently breaking sections.get("THYAO.IS") lookups."""
    report = "**📊 Ticker: THYAO.IS**\n\n**Technical Summary:**\nSignal: Neutral"
    sections = main._split_report_by_ticker(report)
    assert "THYAO.IS" in sections
    assert "THYAO.IS**" not in sections


# --- /get_insights/stream ------------------------------------------------

def _parse_sse(body: str) -> list[dict]:
    import json
    return [
        json.loads(line[len("data: "):])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def test_stream_route_forwards_events_and_yields_final(client, monkeypatch, db_path):
    async def fake_stream_pipeline(ticker_list, tickers=None, history_context=""):
        yield {"type": "tool_call", "agent": "technical", "tool": "rsi_predictor"}
        yield {"type": "token", "agent": "merge", "text": "hi"}
        yield {
            "type": "final",
            "output": "📊 Ticker: THYAO.IS\nSignal: bullish",
            "raw": {
                "technical_results": [{"symbol": "THYAO.IS", "signal": "bullish", "ema34": 1.0,
                                        "ema89": 2.0, "price": 3.0, "rsi_fibo_divergence": "No"}],
                "news_text": "Risk Score: 5/10",
            },
        }

    monkeypatch.setattr(main, "stream_pipeline", fake_stream_pipeline)

    resp = client.get("/get_insights/stream", params={"ticker_list": "THYAO.IS"})

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert events[0] == {"type": "tool_call", "agent": "technical", "tool": "rsi_predictor"}
    assert events[-1]["type"] == "final"
    assert "THYAO.IS" in events[-1]["output"]

    rows = db.get_all_history("THYAO.IS", db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["signal"] == "bullish"


def test_stream_route_cache_hit_returns_single_final_event(client, monkeypatch):
    call_count = {"n": 0}

    async def fake_stream_pipeline(ticker_list, tickers=None, history_context=""):
        call_count["n"] += 1
        yield {"type": "final", "output": "REPORT", "raw": {"technical_results": [], "news_text": ""}}

    monkeypatch.setattr(main, "stream_pipeline", fake_stream_pipeline)

    client.get("/get_insights/stream", params={"ticker_list": "CCC.IS"})
    resp2 = client.get("/get_insights/stream", params={"ticker_list": "CCC.IS"})

    events = _parse_sse(resp2.text)
    assert call_count["n"] == 1
    assert events == [{"type": "final", "output": "REPORT", "cached": True}]
