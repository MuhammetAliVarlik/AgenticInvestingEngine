"""Streamlit front end for the Investing Engine API.

Renders the supervisor's report together with the structured data the API
returns (indicator values, per-symbol risk scores, source attribution), so
nothing on screen depends on parsing free-form model text.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from typing import Any

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")
REQUEST_TIMEOUT_SECONDS = 900.0
REDRAW_INTERVAL_SECONDS = 0.4
MAX_SYMBOLS = 3

SIGNAL_BADGES = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}
AGENTS = {
    "technical_analyst": "📊 Technical",
    "news_analyst": "📰 News risk",
    "macro_analyst": "🏦 Macro",
    "supervisor": "🧠 Report",
}


# --- API client ---------------------------------------------------------------------


@st.cache_data(ttl=3600)
def fetch_instruments() -> list[dict[str, Any]]:
    response = httpx.get(f"{API_BASE_URL}/instruments", timeout=15)
    response.raise_for_status()
    return response.json()


def upload_dataset(symbol: str, file_name: str, content: bytes) -> dict[str, Any]:
    response = httpx.post(
        f"{API_BASE_URL}/datasets",
        data={"symbol": symbol},
        files={"file": (file_name, content, "text/csv")},
        timeout=60,
    )
    if response.status_code == 422:
        raise ValueError(response.json().get("detail", "Invalid file"))
    response.raise_for_status()
    return response.json()


def stream_analysis(symbols: list[str], datasets: dict[str, str]) -> Iterator[dict[str, Any]]:
    with httpx.stream(
        "POST",
        f"{API_BASE_URL}/analyses/stream",
        json={"symbols": symbols, "datasets": datasets},
        timeout=REQUEST_TIMEOUT_SECONDS,
    ) as response:
        if response.status_code == 422:
            response.read()
            raise ValueError(response.json().get("detail", "Invalid request"))
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("data: "):
                yield json.loads(line[6:])


def fetch_history(symbol: str) -> list[dict[str, Any]]:
    response = httpx.get(f"{API_BASE_URL}/history/{symbol}", timeout=30)
    response.raise_for_status()
    return response.json()


# --- Charts -------------------------------------------------------------------------


def risk_gauge(score: float) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            gauge={
                "axis": {"range": [0, 10]},
                "bar": {"color": "#2b2b2b"},
                "steps": [
                    {"range": [0, 3], "color": "#2ecc71"},
                    {"range": [3, 7], "color": "#f1c40f"},
                    {"range": [7, 10], "color": "#e74c3c"},
                ],
            },
            title={"text": "News risk"},
        )
    )
    figure.update_layout(height=220, margin={"l": 20, "r": 20, "t": 40, "b": 10})
    return figure


def history_chart(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["risk_score"],
            mode="lines+markers",
            text=frame["signal"],
            hovertemplate="%{x}<br>Risk: %{y}<br>Signal: %{text}<extra></extra>",
        )
    )
    figure.update_layout(
        yaxis_title="News risk (0-10)", height=260, margin={"l": 20, "r": 20, "t": 10, "b": 20}
    )
    return figure


# --- Rendering ------------------------------------------------------------------------


def render_activity(placeholder: Any, log: list[str], thoughts: dict[str, str]) -> None:
    with placeholder.container():
        if log:
            st.caption("  \n".join(log[-6:]))
        for tab, agent in zip(st.tabs(list(AGENTS.values())), AGENTS, strict=True):
            with tab:
                text = thoughts[agent]
                st.markdown(text + " ▌" if text else "*waiting…*")


def render_indicators(technical: dict[str, Any]) -> None:
    signal = technical.get("signal", "unknown")
    st.markdown(f"#### {SIGNAL_BADGES.get(signal, '⚪')} Signal: **{signal.title()}**")
    rows = {
        "Last close": technical.get("price"),
        "RSI (14)": technical.get("rsi"),
        "Forecast next RSI": technical.get("predicted_next_rsi"),
        "EMA34": technical.get("ema34"),
        "EMA89": technical.get("ema89"),
        "Above EMA34": technical.get("price_above_ema34"),
        "MACD": technical.get("macd"),
        "Bollinger %B": technical.get("bb_pct"),
        "Channel position": technical.get("channel_position"),
        "Momentum divergence": technical.get("momentum_divergence"),
        "As of": technical.get("as_of"),
        "Price source": technical.get("source"),
    }
    table = pd.DataFrame(
        [(k, "-" if v is None else str(v)) for k, v in rows.items()], columns=["Indicator", "Value"]
    )
    st.dataframe(table, hide_index=True, use_container_width=True)


def render_history(symbol: str) -> None:
    try:
        rows = fetch_history(symbol)
    except httpx.HTTPError as exc:
        st.error(f"Could not load history: {exc}")
        return
    if not rows:
        st.info("This is the first analysis of this instrument.")
        return
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    st.plotly_chart(history_chart(frame), use_container_width=True)
    st.dataframe(
        frame[["timestamp", "signal", "risk_score", "price_at_prediction"]].sort_values(
            "timestamp", ascending=False
        ),
        hide_index=True,
        use_container_width=True,
    )


# --- Page -------------------------------------------------------------------------------

st.set_page_config(page_title="Investing Engine", page_icon="📈", layout="wide")
st.title("📈 Investing Engine")
st.caption(
    "Multi-agent research on Borsa Istanbul instruments: technical indicators, headline "
    "risk and the Turkish macro backdrop. For research and education - **not investment advice**."
)

try:
    instruments = {i["symbol"]: i for i in fetch_instruments()}
except httpx.HTTPError as exc:
    st.error(f"Cannot reach the API at {API_BASE_URL}: {exc}")
    st.stop()

selected = st.multiselect(
    "Instruments",
    options=list(instruments),
    default=["XU100"],
    max_selections=MAX_SYMBOLS,
    format_func=lambda s: f"{s} - {instruments[s]['name']}",
)

uploads: dict[str, Any] = {}
needs_prices = [s for s in selected if not instruments[s]["public_prices"]]
if needs_prices:
    st.info(
        "Equity prices are licensed data, so the engine analyses your own export instead. "
        "Upload a daily OHLCV CSV (a Date and Close column at minimum; Turkish brokerage "
        "exports are supported). Files stay in memory for this session only."
    )
    for symbol in needs_prices:
        uploads[symbol] = st.file_uploader(f"{symbol} price history (CSV)", type=["csv"])

ready = bool(selected) and all(uploads.get(s) is not None for s in needs_prices)
if st.button("Analyze", type="primary", disabled=not ready):
    datasets: dict[str, str] = {}
    try:
        for symbol, file in uploads.items():
            datasets[symbol] = upload_dataset(symbol, file.name, file.getvalue())["dataset_id"]
    except (ValueError, httpx.HTTPError) as exc:
        st.error(f"Upload rejected: {exc}")
        st.stop()

    log: list[str] = []
    thoughts = dict.fromkeys(AGENTS, "")
    final: dict[str, Any] | None = None

    st.markdown("#### Live agent activity")
    activity = st.empty()
    render_activity(activity, log, thoughts)
    last_redraw = 0.0

    try:
        for event in stream_analysis(selected, datasets):
            kind, agent = event["type"], event.get("agent", "")
            label = AGENTS.get(agent, agent)
            if kind == "status":
                log.append(f"🧭 {event['text']}")
            elif kind == "tool_call":
                log.append(f"🔧 {label} → `{event['tool']}`")
            elif kind == "tool_result":
                log.append(f"✅ {label} ← `{event['tool']}`")
            elif kind == "error":
                log.append(f"⚠️ {event['message']}")
            elif kind == "token" and agent in thoughts:
                thoughts[agent] += event["text"]
            elif kind == "final":
                final = event
                if event.get("cached"):
                    log.append("⚡ Served from cache")

            now = time.monotonic()
            if kind != "token" or now - last_redraw > REDRAW_INTERVAL_SECONDS:
                last_redraw = now
                render_activity(activity, log, thoughts)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except httpx.HTTPError as exc:
        st.error(f"Connection to the API failed: {exc}")
        st.stop()

    render_activity(activity, log, thoughts)
    if not final or not final.get("report"):
        st.error("The analysis could not be completed. Please try again.")
        st.stop()

    st.success("Analysis complete.")
    report_tab, *symbol_tabs = st.tabs(["📄 Report", *selected])
    with report_tab:
        st.markdown(final["report"])
    for tab, symbol in zip(symbol_tabs, selected, strict=True):
        with tab:
            left, right = st.columns([2, 1])
            with left:
                technical = final["technical"].get(symbol)
                if technical:
                    render_indicators(technical)
                else:
                    st.info("No technical data was produced for this instrument.")
            with right:
                if (score := final["risk"].get(symbol)) is not None:
                    st.plotly_chart(risk_gauge(score), use_container_width=True)
            st.markdown("##### Previous analyses")
            render_history(symbol)

    st.caption(" · ".join(s["attribution"] for s in final.get("sources", [])))
