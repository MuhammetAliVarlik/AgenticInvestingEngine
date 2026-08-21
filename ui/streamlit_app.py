import json
import os
import re
import time

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")
REQUEST_TIMEOUT_SECONDS = 1200.0  # a full multi-agent pipeline run can be slow, especially on modest hardware
THOUGHTS_REFRESH_INTERVAL = 0.4  # seconds between UI redraws while tokens stream in, to avoid flicker/lag

TICKER_SECTION_RE = re.compile(r"📊 Ticker:\s*(\S+)")
SIGNAL_BADGES = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}
AGENT_LABELS = {
    "stock_price_expert": "📊 Technical",
    "stock_news_expert": "📰 News Risk",
    "pdp_expert": "🧾 PDP Filing",
    "merge": "🧠 Merging report",
}
AGENT_ORDER = ("stock_price_expert", "stock_news_expert", "pdp_expert", "merge")


def split_report_by_ticker(report_text: str) -> dict[str, str]:
    """Splits the merged report text into per-ticker sections, mirroring
    api/main.py's _split_report_by_ticker (kept independent since the UI
    talks to the API over HTTP rather than importing its internals)."""
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


def _extract(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def parse_ticker_section(text: str) -> dict:
    """Best-effort field extraction from one ticker's report section. The
    report is LLM-generated free text following a template, not guaranteed
    structured output, so every field can be None - callers must handle
    that (the UI always shows the raw section text as a fallback)."""
    return {
        "signal": _extract(r"Signal:\s*([A-Za-z]+)", text),
        "ema34": _extract(r"EMA34:\s*₺?\s*([\d,.]+)", text),
        "ema89": _extract(r"EMA89:\s*₺?\s*([\d,.]+)", text),
        "price_above_ema34": _extract(r"Price Above EMA34:\s*(\w+)", text),
        "rsi_divergence": _extract(r"RSI[/\- ]?Fibonacci Divergence:\s*(\w+)", text),
        "channel_position": _extract(r"Channel Position:\s*(\w+)", text),
        "risk_score": _extract(r"Risk Score:\s*([\d.]+)\s*/\s*10", text),
        "market_impact": _extract(r"Market Impact:\s*(\w+)", text),
        "recommendation": _extract(r"Recommendation:\s*(.+)", text),
    }


def stream_insights(ticker_list: str):
    """Consumes the SSE /get_insights/stream endpoint, yielding each parsed
    event dict as it arrives."""
    with httpx.stream(
        "GET", f"{API_BASE_URL}/get_insights/stream",
        params={"ticker_list": ticker_list}, timeout=REQUEST_TIMEOUT_SECONDS,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            yield json.loads(line[len("data: "):])


def fetch_history(ticker: str) -> list[dict]:
    resp = httpx.get(f"{API_BASE_URL}/history", params={"ticker": ticker}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def risk_gauge(risk_score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            gauge={
                "axis": {"range": [0, 10]},
                "bar": {"color": "black"},
                "steps": [
                    {"range": [0, 3], "color": "#2ecc71"},
                    {"range": [3, 7], "color": "#f1c40f"},
                    {"range": [7, 10], "color": "#e74c3c"},
                ],
            },
            title={"text": "Risk Score"},
        )
    )
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def history_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["risk_score"], mode="lines+markers",
            text=df["signal"], hovertemplate="%{x}<br>Risk: %{y}<br>Signal: %{text}<extra></extra>",
            name="Risk Score",
        )
    )
    fig.update_layout(yaxis_title="Risk Score (0-10)", height=280, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def render_thoughts(placeholder, status_lines: list[str], thoughts: dict[str, str]) -> None:
    """Renders the live 'inner thoughts' panel: recent tool-call/result
    status lines plus each agent's streamed reasoning/summary text so far -
    visible proof the app is actively working during a long wait."""
    with placeholder.container():
        if status_lines:
            st.caption("  \n".join(status_lines[-6:]))
        tabs = st.tabs([AGENT_LABELS[a] for a in AGENT_ORDER])
        for tab, agent in zip(tabs, AGENT_ORDER):
            with tab:
                text = thoughts.get(agent, "")
                st.markdown(text + " ▌" if text else "*waiting...*")


st.set_page_config(page_title="AgenticInvestingEngine", page_icon="📈", layout="wide")

st.title("📈 AgenticInvestingEngine")
st.caption(
    "AI-powered technical, news-risk, and PDP (KAP) filing analysis for BIST100 stocks. "
    "Output is for personal research only and is **not financial advice**."
)

input_col, button_col = st.columns([4, 1])
with input_col:
    ticker_input = st.text_input(
        "Ticker(s), comma-separated", placeholder="THYAO.IS,ARCLK.IS", label_visibility="collapsed"
    )
with button_col:
    submit = st.button("Analyze", type="primary", use_container_width=True)

if submit and not ticker_input.strip():
    st.warning("Enter at least one ticker symbol.")

if submit and ticker_input.strip():
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    status_lines: list[str] = []
    thoughts: dict[str, str] = {a: "" for a in AGENT_ORDER}
    final_event = None
    error = None

    st.markdown("#### 🧠 Inner thoughts — live agent activity")
    st.caption("Streamed directly from the running agents so you can see the app is working, not frozen.")
    thoughts_placeholder = st.empty()
    render_thoughts(thoughts_placeholder, status_lines, thoughts)

    last_render = 0.0
    try:
        for event in stream_insights(ticker_input):
            etype = event.get("type")
            agent = event.get("agent")

            if etype == "tool_call":
                status_lines.append(f"🔧 [{AGENT_LABELS.get(agent, agent)}] calling `{event['tool']}`...")
            elif etype == "tool_result":
                status_lines.append(f"✅ [{AGENT_LABELS.get(agent, agent)}] `{event['tool']}` returned")
            elif etype == "worker_error":
                status_lines.append(f"⚠️ [{AGENT_LABELS.get(agent, agent)}] error: {event['error']}")
            elif etype == "status":
                status_lines.append(f"🧠 {event['text']}")
            elif etype == "token" and agent in thoughts:
                thoughts[agent] += event["text"]
            elif etype == "final":
                final_event = event
                if event.get("cached"):
                    status_lines.append("⚡ Served from cache (identical request in the last 10 minutes)")

            now = time.monotonic()
            if etype != "token" or now - last_render > THOUGHTS_REFRESH_INTERVAL:
                last_render = now
                render_thoughts(thoughts_placeholder, status_lines, thoughts)
    except httpx.HTTPError as exc:
        error = str(exc)

    render_thoughts(thoughts_placeholder, status_lines, thoughts)

    if error:
        st.error(f"Could not reach the API at {API_BASE_URL}: {error}")
        st.stop()

    if not final_event:
        st.error("Stream ended without a final report.")
        st.stop()

    st.success("Analysis complete.")

    report_text = final_event.get("output", "")
    sections = split_report_by_ticker(report_text)

    if not sections:
        st.warning("Couldn't identify per-ticker sections in the model's output. Showing the raw report below.")
        st.text(report_text)
    else:
        for ticker in tickers:
            section_text = sections.get(ticker)
            st.divider()
            st.subheader(ticker)

            if section_text is None:
                st.info("No section found for this ticker in the report.")
                continue

            fields = parse_ticker_section(section_text)

            report_tab, thoughts_tab, history_tab = st.tabs(["📄 Report", "🧠 Agent reasoning", "📜 History"])

            with report_tab:
                col_summary, col_risk = st.columns([2, 1])
                with col_summary:
                    signal = (fields["signal"] or "unknown").lower()
                    badge = SIGNAL_BADGES.get(signal, "⚪")
                    st.markdown(f"#### {badge} Signal: **{(fields['signal'] or 'Unknown').title()}**")

                    indicator_rows = {
                        "EMA34": fields["ema34"],
                        "EMA89": fields["ema89"],
                        "Price Above EMA34": fields["price_above_ema34"],
                        "RSI/Fibonacci Divergence": fields["rsi_divergence"],
                        "Channel Position": fields["channel_position"],
                    }
                    st.table(
                        pd.DataFrame(indicator_rows.items(), columns=["Indicator", "Value"]).set_index("Indicator")
                    )

                    if fields["market_impact"]:
                        st.markdown(f"**PDP Market Impact:** {fields['market_impact']}")
                    if fields["recommendation"]:
                        st.markdown(f"**Recommendation:** {fields['recommendation']}")

                with col_risk:
                    if fields["risk_score"]:
                        try:
                            st.plotly_chart(risk_gauge(float(fields["risk_score"])), use_container_width=True)
                        except ValueError:
                            st.caption(f"Risk score: {fields['risk_score']}/10")

                with st.expander("Full report section"):
                    st.text(section_text)

            with thoughts_tab:
                st.caption("What each agent actually said while producing this report.")
                for agent in AGENT_ORDER:
                    if thoughts.get(agent):
                        st.markdown(f"**{AGENT_LABELS[agent]}**")
                        st.text(thoughts[agent])

            with history_tab:
                try:
                    history = fetch_history(ticker)
                except httpx.HTTPError as exc:
                    st.error(f"Could not load history: {exc}")
                    history = []

                if not history:
                    st.info("No prior predictions for this ticker yet - this is the first.")
                else:
                    hist_df = pd.DataFrame(history)
                    hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
                    st.plotly_chart(history_chart(hist_df), use_container_width=True)
                    st.dataframe(
                        hist_df[["timestamp", "signal", "risk_score", "summary_text"]]
                        .rename(columns={"summary_text": "summary"})
                        .sort_values("timestamp", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                    )
