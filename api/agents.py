import json
import os
import time

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from tools.yahooFinanceNewsTool import stock_news
from tools.stockPriceAnaliserTool import rsi_predictor
from tools.PDPTool import pdp_news_scraper

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama_server:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.1:latest")

model = ChatOllama(model=MODEL_NAME, base_url=BASE_URL, verbose=True)

stock_price_analiser_agent = create_react_agent(
    model=model,
    tools=[rsi_predictor],
    name="stock_price_expert",
    prompt="""
    You are a technical analysis expert.
    You should analysis data for given stock.
    Use tool to evaluate:
    - signal, ema34, ema89, price_above_ema34
    - rsi_fibo_divergence, channel_position

    Always call the tool with the ticker symbol exactly as given in the user
    message, including any suffix such as '.IS'. Never shorten, translate, or
    otherwise modify the symbol.

    Output a short summary explaining the technical setup and sentiment (bullish, bearish, neutral). Keep it concise and data-driven.
    """
)

stock_news_agent = create_react_agent(
    model=model,
    tools=[stock_news],
    name="stock_news_expert",
    prompt="""
    You are a stock news risk analyst.

    Always call the tool with the ticker symbol exactly as given in the user
    message, including any suffix such as '.IS'. Never shorten, translate, or
    otherwise modify the symbol.

    Analyze recent news and assign a **Risk Score (1–10)** with 2–3 sentence reasoning.

    === Risk Assessment ===
    Risk Score: <score>/10
    Reasoning: <short explanation>
    =======================
    """
)


pdp_agent = create_react_agent(
    model=model,
    tools=[pdp_news_scraper],
    name="pdp_expert",
    prompt="""
    You analyze PDP (KAP) filings for the requested ticker. Always call the
    tool with the ticker symbol exactly as given in the user message,
    including any suffix such as '.IS'. The tool only returns disclosures
    that are actually confirmed to belong to that ticker - if it reports
    no disclosures were found, say that plainly rather than inventing one.

    Provide:
    - Summary of the latest disclosure
    - Market impact (Positive/Negative/Neutral)
    - 2–3 sentence explanation

    === PDP Analysis ===
    Summary: <brief summary>
    Market Impact: <Positive | Negative | Neutral>
    Reasoning: <short explanation>
    ======================
    """
)

SUPERVISOR_PROMPT = """
You are a supervisor agent overseeing a team of three financial analysis experts:

- stock_price_expert: technical analysis (signal, EMA34/EMA89, RSI/Fibonacci
  divergence, channel position).
- stock_news_expert: news-based risk scoring (1-10) with reasoning.
- pdp_expert: PDP/KAP disclosure analysis (summary, market impact).

For the ticker(s) named in the user's request, decide which experts to consult and in
what order to gather everything you need about them, then combine their findings
yourself into one comprehensive investment report. Do not fabricate an expert's
findings - always consult an expert before reporting on their area.

IMPORTANT: produce exactly one report section per ticker requested - no more, no
fewer. Do not invent, substitute, or additionally report on any other company or
ticker, even if one is mentioned in an expert's findings (e.g. an unrelated company
named in a generic news/disclosure feed). If an expert's findings don't contain
usable data for a requested ticker, say so plainly in that ticker's section rather
than fabricating figures.

If the user's request includes recent prediction history for a ticker, compare
today's findings against it and explicitly call out any meaningful shift (e.g. "This
is a shift from the neutral signal three days ago"); if it's broadly consistent with
the history, don't force commentary about it.

For each ticker:
- Combine all expert insights into a well-structured financial report.
- Your goal is to **maximize potential return with calculated risk**.
- Present **all monetary values in Turkish Lira (₺)**.

The final report must include, per ticker:

=======================
📊 Ticker: <SYMBOL>

🔍 Technical Summary:
- Signal: <bullish/bearish/neutral>
- EMA34: ₺<value>
- EMA89: ₺<value>
- Price Above EMA34: <true/false>
- RSI/Fibonacci Divergence: <yes/no>
- Channel Position: <top/middle/bottom>

📅 Next Day Outlook:
- Forecasted Price Range: ₺<low> - ₺<high>
- Decision: <Buy | Hold | Sell>
- Reasoning: <short 2–3 sentence justification based on short-term indicators>

📆 Strategy Recommendations:
- Short-Term (1–7d): <Buy | Hold | Sell> → <reason>
- Mid-Term (1w–1mo): <Buy | Hold | Sell> → <reason>
- Long-Term (1–6mo): <Buy | Hold | Sell> → <reason>

⚠️ Risk Analysis:
=== Risk Assessment ===
Risk Score: <x>/10
Reasoning: <short explanation>
=======================

🧾 PDP Filing Summary:
=== PDP Analysis ===
Summary: <brief summary>
Market Impact: <Positive | Negative | Neutral>
Reasoning: <brief impact analysis>
======================

✅ Final Action Plan:
Recommendation: <e.g., Buy on pullback, Aggressive Hold, Avoid Short-Term>
Suggested Strategy: <e.g., RSI breakout entry, EMA trend-following, Mean-reversion>

=======================

You must make bold but well-reasoned decisions. Use all data to guide the investor with confidence.
"""

HISTORY_INSTRUCTION = """
Recent prediction history for these tickers (most recent first):
{history_context}
"""

WORKER_NODES = ("stock_price_expert", "stock_news_expert", "pdp_expert")
SUPERVISOR_NODE = "supervisor"


def _last_ai_text(response: dict, agent_name: str | None = None) -> str:
    """Return the content of the last AIMessage in a messages list that was
    actually a final answer from `agent_name` - not a handoff-decision
    message (empty content) or an in-flight tool-calling turn. Skipping
    messages that still carry tool_calls is what lets this correctly find an
    expert's real summary even though the shared, full-history message list
    also contains the supervisor's own routing decisions tagged with names
    other than the expert's."""
    for msg in reversed(response.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            if agent_name is None or msg.name == agent_name:
                return msg.content
    return ""


def _extract_tool_results(response: dict, tool_name: str) -> list[dict]:
    """Return every parsed dict returned by the given tool during this agent run.

    create_react_agent's ToolNode JSON-serializes a dict tool return into
    ToolMessage.content, so this reverses that with json.loads instead of
    re-parsing the LLM's prose summary.
    """
    results = []
    for msg in response.get("messages", []):
        if isinstance(msg, ToolMessage) and msg.name == tool_name:
            try:
                parsed = json.loads(msg.content)
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, dict) and "error" not in parsed:
                results.append(parsed)
    return results


def _build_request_message(tickers: list[str], history_context: str = "") -> str:
    message = f"Analyze the following ticker(s): {', '.join(tickers)}."
    if history_context:
        message += "\n\n" + HISTORY_INSTRUCTION.format(history_context=history_context)
    return message


def _build_supervisor_graph():
    """Builds the real decision-making supervisor: langgraph_supervisor wires
    the three expert agents as handoff targets behind tools
    (transfer_to_stock_price_expert, ...) bound to the supervisor's own LLM,
    so the supervisor genuinely decides - via its own tool-calls, one LLM
    turn at a time - which expert(s) to consult and when to stop and write
    the final report, exactly like this project's original architecture.
    add_handoff_back_messages=False keeps the shared message history free of
    synthetic "Transferring back to supervisor" turns, which would otherwise
    sit between an expert's real answer and the supervisor's next decision."""
    workflow = create_supervisor(
        agents=[stock_price_analiser_agent, stock_news_agent, pdp_agent],
        model=model,
        prompt=SUPERVISOR_PROMPT,
        output_mode="full_history",
        add_handoff_back_messages=False,
    )
    return workflow.compile()


supervisor_graph = _build_supervisor_graph()


async def stream_pipeline(ticker_list: str, tickers: list[str] | None = None, history_context: str = ""):
    """Runs the compiled supervisor graph and yields live progress events as
    an async generator - intended for a UI to show real-time "inner
    thoughts" during the (often very slow, especially on modest hardware)
    multi-minute wait.

    The supervisor is a genuine decision-maker, not a fixed pipeline: each
    step, its own LLM call decides whether to hand off to one of the three
    experts next or to stop and write the final report. That decision shows
    up in the stream as a handoff tool-call (transfer_to_<expert>), which
    this translates into a "status" event rather than a raw tool_call, since
    it's supervisor routing, not a real analysis tool.

    Uses stream_mode=["messages", "updates"] with subgraphs=True:
      - "messages" (per subgraph namespace) gives token-by-token/tool-call/
        tool-result events for the live SSE display, tagged with the
        originating expert node.
      - "updates" gives each node's turn as a state-update delta the moment
        it completes. Because create_supervisor runs with
        output_mode="full_history", every one of these deltas carries the
        entire accumulated message list up to that point - so the very last
        one seen (always the supervisor's own closing turn) already
        contains every expert's contribution, and no per-node tracking is
        needed to extract them.

    Event shapes (unchanged from before this refactor):
      {"type": "tool_call", "agent": <label>, "tool": <name>}
      {"type": "tool_result", "agent": <label>, "tool": <name>}
      {"type": "token", "agent": <label or "merge">, "text": <str>}
      {"type": "worker_error", "agent": <label>, "error": <str>}
      {"type": "status", "text": <str>}
      {"type": "final", "output": <str>, "raw": {...}}   # last event
    """
    if tickers is None:
        tickers = [t.strip().upper() for t in ticker_list.split(",") if t.strip()]

    initial_state = {"messages": [HumanMessage(_build_request_message(tickers, history_context))]}

    final_state: dict = {"messages": []}
    status_emitted = False

    try:
        async for ns, mode, item in supervisor_graph.astream(
            initial_state, stream_mode=["messages", "updates"], subgraphs=True
        ):
            if mode == "updates":
                if ns:  # only the root-level updates carry whole-node deltas we care about
                    continue
                for update in item.values():
                    final_state = update
                continue

            # mode == "messages"
            chunk, meta = item
            node = ns[0].split(":")[0] if ns else meta.get("langgraph_node", "")
            agent_label = node if node in WORKER_NODES else "merge"

            if isinstance(chunk, ToolMessage):
                if chunk.name and chunk.name.startswith("transfer_"):
                    continue  # supervisor handoff bookkeeping, not a real tool result
                yield {"type": "tool_result", "agent": agent_label, "tool": chunk.name}
            elif getattr(chunk, "tool_calls", None):
                for call in chunk.tool_calls:
                    name = call.get("name") or ""
                    if name.startswith("transfer_to_"):
                        target = name.removeprefix("transfer_to_")
                        yield {"type": "status", "text": f"Supervisor is consulting {target}..."}
                    elif name and not name.startswith("transfer_"):
                        yield {"type": "tool_call", "agent": agent_label, "tool": name}
            elif getattr(chunk, "content", None):
                if agent_label == "merge" and not status_emitted:
                    yield {"type": "status", "text": "Supervisor is writing the final report..."}
                    status_emitted = True
                yield {"type": "token", "agent": agent_label, "text": chunk.content}
    except Exception as exc:
        yield {"type": "worker_error", "agent": "graph", "error": str(exc)}

    yield {
        "type": "final",
        "output": _last_ai_text(final_state, SUPERVISOR_NODE),
        "raw": {
            "technical_results": _extract_tool_results(final_state, "rsi_predictor"),
            "technical_text": _last_ai_text(final_state, "stock_price_expert"),
            "news_text": _last_ai_text(final_state, "stock_news_expert"),
            "pdp_text": _last_ai_text(final_state, "pdp_expert"),
        },
    }


async def run_pipeline(ticker_list: str, tickers: list[str] | None = None, history_context: str = "") -> dict:
    """Non-streaming entry point (used by /get_insights): drains
    stream_pipeline (the same compiled supervisor graph run) and returns the
    final result plus per-stage timing, derived from the same event stream
    rather than a second graph invocation.

    Returns {"output": str, "raw": {...}, "stages": {...}} where "raw" carries
    structured per-ticker technical fields (for history persistence) and the
    experts' text output, and "stages" carries timing in seconds: the
    supervisor's own routing/gathering decisions are not a separate phase
    from writing the report (it's one sequential decision-by-decision run),
    so "gathering_seconds" times up to the moment the final synthesis starts
    and "synthesis_seconds" times the final report generation itself.
    """
    t0 = time.perf_counter()
    t_gathering_done = None
    stages = {}
    final_event = None

    async for event in stream_pipeline(ticker_list, tickers=tickers, history_context=history_context):
        if event["type"] == "status" and event["text"].startswith("Supervisor is writing") and t_gathering_done is None:
            t_gathering_done = time.perf_counter()
            stages["gathering_seconds"] = round(t_gathering_done - t0, 3)
        elif event["type"] == "final":
            final_event = event

    t_end = time.perf_counter()
    stages["synthesis_seconds"] = round(t_end - (t_gathering_done or t0), 3)

    return {
        "output": final_event["output"] if final_event else "",
        "raw": final_event["raw"] if final_event else {
            "technical_results": [], "technical_text": "", "news_text": "", "pdp_text": "",
        },
        "stages": stages,
    }
