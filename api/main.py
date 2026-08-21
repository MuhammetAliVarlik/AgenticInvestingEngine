import json
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

import db
from agents import run_pipeline, stream_pipeline
from cache import get_cached, set_cached
from tools.stockPriceAnaliserTool import rsi_predictor

RISK_SCORE_RE = re.compile(r"Risk Score:\s*(\d+(?:\.\d+)?)\s*/\s*10")
TICKER_SECTION_RE = re.compile(r"📊 Ticker:\s*(\S+)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify the origins you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_tickers(ticker_list: str) -> list[str]:
    return [t.strip().upper() for t in ticker_list.split(",") if t.strip()]


def _humanize_days_ago(timestamp_str: str) -> str:
    try:
        ts = datetime.fromisoformat(timestamp_str)
    except ValueError:
        return timestamp_str
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    days = (datetime.now(timezone.utc) - ts).days
    if days <= 0:
        return "today"
    if days == 1:
        return "1 day ago"
    return f"{days} days ago"


def _summarize_history(rows: list[dict]) -> str:
    if not rows:
        return ""
    parts = []
    for row in rows:
        when = _humanize_days_ago(row["timestamp"])
        risk = row.get("risk_score")
        risk_str = f"{risk:g}/10" if risk is not None else "unknown"
        parts.append(f"{when} — {row.get('signal') or 'unknown'}, risk {risk_str}")
    return "; ".join(parts)


def _build_history_context(tickers: list[str]) -> str:
    return "\n".join(
        f"{t}: {summary}"
        for t in tickers
        if (summary := _summarize_history(db.get_history(t, limit=5)))
    )


def _split_report_by_ticker(report_text: str) -> dict[str, str]:
    """Splits the merged report into per-ticker sections keyed by ticker
    symbol, so history rows store the relevant excerpt rather than the
    whole (possibly multi-ticker) report."""
    sections: dict[str, str] = {}
    chunks = re.split(r"(?=📊 Ticker:\s*\S+)", report_text)
    for chunk in chunks:
        match = TICKER_SECTION_RE.match(chunk.strip())
        if match:
            # Strip trailing punctuation/markdown (e.g. "THYAO.IS**" when the
            # model bolds the header) - keep only the ticker itself.
            symbol = re.sub(r"[^A-Z0-9.]+$", "", match.group(1).upper())
            sections[symbol] = chunk.strip()
    return sections


def _persist_predictions(tickers: list[str], output: str, raw: dict) -> None:
    risk_match = RISK_SCORE_RE.search(raw["news_text"])
    risk_score = float(risk_match.group(1)) if risk_match else None

    technical_by_symbol = {
        r["symbol"].upper(): r for r in raw["technical_results"] if r.get("symbol")
    }
    report_sections = _split_report_by_ticker(output)

    for ticker in tickers:
        technical = technical_by_symbol.get(ticker)
        summary_text = report_sections.get(ticker, output)[:1500]
        db.save_prediction(
            ticker=ticker,
            signal=technical.get("signal") if technical else None,
            risk_score=risk_score,
            ema34=technical.get("ema34") if technical else None,
            ema89=technical.get("ema89") if technical else None,
            price_at_prediction=technical.get("price") if technical else None,
            rsi_divergence=technical.get("rsi_fibo_divergence") if technical else None,
            summary_text=summary_text,
        )


@app.get("/")
def read_root():
    return {"message": "🎉 FastAPI working!"}


@app.get("/stock_price")
async def get_stock_price(ticker: str):
    return await run_in_threadpool(rsi_predictor, ticker)


@app.get("/history")
def get_history(ticker: str, limit: int = 50):
    return db.get_all_history(ticker.strip().upper(), limit=limit)


@app.get("/get_insights")
async def get_insights(ticker_list: str, background_tasks: BackgroundTasks, debug: bool = False):
    cached = get_cached(ticker_list)
    if cached is not None:
        return cached

    tickers = _parse_tickers(ticker_list)
    history_context = _build_history_context(tickers)

    result = await run_pipeline(ticker_list, tickers=tickers, history_context=history_context)

    response = {"output": result["output"]}
    if debug:
        response["stages"] = result["stages"]

    background_tasks.add_task(_persist_predictions, tickers, result["output"], result["raw"])

    set_cached(ticker_list, response)
    return response


@app.get("/get_insights/stream")
async def get_insights_stream(ticker_list: str):
    """Server-sent-events variant of /get_insights: streams live progress
    (tool calls, tool results, and reasoning/summary tokens per agent) as
    the pipeline runs, so a UI can show the app is actively working during
    what can be a multi-minute wait, instead of a single blocking call.
    Persists history and populates the cache the same way /get_insights
    does, once the stream completes."""
    cached = get_cached(ticker_list)
    tickers = _parse_tickers(ticker_list)

    async def _event_stream():
        if cached is not None:
            yield f"data: {json.dumps({'type': 'final', 'output': cached['output'], 'cached': True}, ensure_ascii=False)}\n\n"
            return

        history_context = _build_history_context(tickers)
        final_event = None

        async for event in stream_pipeline(ticker_list, tickers=tickers, history_context=history_context):
            if event["type"] == "final":
                final_event = event
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        if final_event:
            _persist_predictions(tickers, final_event["output"], final_event["raw"])
            set_cached(ticker_list, {"output": final_event["output"]})

    return StreamingResponse(_event_stream(), media_type="text/event-stream")
