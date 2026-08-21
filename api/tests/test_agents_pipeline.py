import agents
from helpers import (
    FakeToolCallingChatModel,
    make_ai_message,
    make_tool_message,
    patch_supervisor_graph,
    supervisor_turns,
)
from langchain_core.messages import AIMessage


def _worker_response(*messages):
    return {"messages": list(messages)}


# --- _last_ai_text --------------------------------------------------

def test_last_ai_text_returns_final_ai_message_content():
    response = _worker_response(
        make_ai_message(""),
        make_tool_message({"symbol": "THYAO.IS"}, name="rsi_predictor"),
        make_ai_message("final summary text"),
    )
    assert agents._last_ai_text(response) == "final summary text"


def test_last_ai_text_returns_empty_string_when_no_ai_message():
    response = _worker_response(make_tool_message({"a": 1}, name="rsi_predictor"))
    assert agents._last_ai_text(response) == ""


def test_last_ai_text_filters_by_agent_name():
    """Once every expert's turns land in one shared, full-history message
    list (langgraph_supervisor's output_mode="full_history"), _last_ai_text
    must be able to pick out just one agent's contribution."""
    response = _worker_response(
        AIMessage(content="technical text", name="stock_price_expert"),
        AIMessage(content="news text", name="stock_news_expert"),
    )
    assert agents._last_ai_text(response, "stock_price_expert") == "technical text"
    assert agents._last_ai_text(response, "stock_news_expert") == "news text"
    assert agents._last_ai_text(response, "pdp_expert") == ""


def test_last_ai_text_skips_messages_that_still_carry_tool_calls():
    """An expert's node can produce more than one AIMessage tagged with its
    name (its real answer, and - depending on handoff settings - a handoff
    acknowledgement). Only a message with no pending tool_calls is a real
    final answer."""
    response = _worker_response(
        AIMessage(content="the real answer", name="stock_price_expert"),
        AIMessage(
            content="Transferring back to supervisor",
            name="stock_price_expert",
            tool_calls=[{"name": "transfer_back_to_supervisor", "args": {}, "id": "1"}],
        ),
    )
    assert agents._last_ai_text(response, "stock_price_expert") == "the real answer"


# --- _extract_tool_results -------------------------------------------

def test_extract_tool_results_parses_json_tool_messages():
    response = _worker_response(
        make_tool_message({"symbol": "THYAO.IS", "ema34": 100.0}, name="rsi_predictor"),
        make_ai_message("summary"),
    )
    results = agents._extract_tool_results(response, "rsi_predictor")
    assert results == [{"symbol": "THYAO.IS", "ema34": 100.0}]


def test_extract_tool_results_skips_error_dicts():
    response = _worker_response(
        make_tool_message({"error": "No data"}, name="rsi_predictor"),
    )
    assert agents._extract_tool_results(response, "rsi_predictor") == []


def test_extract_tool_results_ignores_other_tool_names():
    response = _worker_response(
        make_tool_message({"Summary": "news"}, name="stock_news"),
    )
    assert agents._extract_tool_results(response, "rsi_predictor") == []


def test_extract_tool_results_multiple_calls_all_collected():
    response = _worker_response(
        make_tool_message({"symbol": "AAA.IS"}, name="rsi_predictor", tool_call_id="1"),
        make_tool_message({"symbol": "BBB.IS"}, name="rsi_predictor", tool_call_id="2"),
    )
    results = agents._extract_tool_results(response, "rsi_predictor")
    assert {r["symbol"] for r in results} == {"AAA.IS", "BBB.IS"}


# --- _build_request_message ---------------------------------------------

def test_build_request_message_lists_requested_tickers():
    message = agents._build_request_message(["THYAO.IS", "ARCLK.IS"])
    assert "THYAO.IS, ARCLK.IS" in message


def test_build_request_message_omits_history_section_when_absent():
    message = agents._build_request_message(["THYAO.IS"], history_context="")
    assert "Recent prediction history" not in message


def test_build_request_message_includes_history_section_when_present():
    message = agents._build_request_message(
        ["THYAO.IS"], history_context="THYAO.IS: 3 days ago — Neutral, risk 4/10",
    )
    assert "Recent prediction history" in message
    assert "3 days ago — Neutral, risk 4/10" in message


# --- supervisor graph structure -----------------------------------------
# Concrete, inspectable evidence this is a genuine decision-making
# supervisor: every expert is reachable only through the supervisor node (a
# hub, wired by langgraph_supervisor's handoff tools), never directly from
# START and never from one another - so which expert(s) actually run, and in
# what order, is the supervisor LLM's own runtime decision, not a fixed
# graph topology.

def test_supervisor_graph_has_worker_and_supervisor_nodes():
    node_names = set(agents.supervisor_graph.get_graph().nodes.keys())
    assert {"stock_price_expert", "stock_news_expert", "pdp_expert", "supervisor"} <= node_names


def test_supervisor_graph_experts_are_only_reachable_through_supervisor():
    edges = {(e.source, e.target) for e in agents.supervisor_graph.get_graph().edges}
    assert ("__start__", "supervisor") in edges
    for worker in agents.WORKER_NODES:
        assert ("__start__", worker) not in edges  # no fixed fan-out from START
        assert (worker, "supervisor") in edges  # every expert reports back to the supervisor
        for other in agents.WORKER_NODES:
            if other != worker:
                assert (worker, other) not in edges  # experts never call each other directly


def test_supervisor_graph_supervisor_can_reach_every_expert():
    edges = {(e.source, e.target) for e in agents.supervisor_graph.get_graph().edges}
    for worker in agents.WORKER_NODES:
        assert ("supervisor", worker) in edges


# --- run_pipeline / stream_pipeline (mocked at the LLM level, real graph) -

async def test_run_pipeline_merges_worker_outputs(monkeypatch):
    patch_supervisor_graph(
        agents, monkeypatch,
        technical_responses=[AIMessage(content="technical summary")],
        news_responses=[AIMessage(content="Risk Score: 4/10\nReasoning: calm")],
        pdp_responses=[AIMessage(content="pdp summary")],
        final_text="MERGED REPORT",
    )

    result = await agents.run_pipeline("THYAO.IS", tickers=["THYAO.IS"])

    assert result["output"].strip() == "MERGED REPORT"
    assert result["raw"]["technical_text"].strip() == "technical summary"
    assert result["raw"]["news_text"].strip() == "Risk Score: 4/10\nReasoning: calm"
    assert result["raw"]["pdp_text"].strip() == "pdp summary"
    assert "gathering_seconds" in result["stages"]
    assert "synthesis_seconds" in result["stages"]


async def test_run_pipeline_output_is_not_a_stringified_state_dict(monkeypatch):
    """Regression test for the original bug where ask() returned
    str(response) (a raw LangGraph state dump) because the state dict has
    no "output" key. run_pipeline must return the supervisor's plain final
    text."""
    patch_supervisor_graph(
        agents, monkeypatch,
        technical_responses=[AIMessage(content="t")],
        news_responses=[AIMessage(content="n")],
        pdp_responses=[AIMessage(content="p")],
        final_text="clean text report",
    )

    result = await agents.run_pipeline("THYAO.IS", tickers=["THYAO.IS"])

    assert result["output"].strip() == "clean text report"
    assert "messages=" not in result["output"]  # not a repr() of a state dict


async def test_run_pipeline_supervisor_can_skip_an_expert(monkeypatch):
    """Direct evidence the supervisor is a genuine decision-maker and not a
    fixed fan-out: when its own model hands off to only two of the three
    experts, the third is never consulted at all."""
    patch_supervisor_graph(
        agents, monkeypatch,
        technical_responses=[AIMessage(content="technical summary")],
        news_responses=[AIMessage(content="Risk Score: 4/10\nReasoning: calm")],
        pdp_responses=[AIMessage(content="should never be produced")],
        supervisor_responses=supervisor_turns(
            "stock_price_expert", "stock_news_expert", final_text="MERGED REPORT",
        ),
    )

    result = await agents.run_pipeline("THYAO.IS", tickers=["THYAO.IS"])

    assert result["output"].strip() == "MERGED REPORT"
    assert result["raw"]["technical_text"].strip() == "technical summary"
    assert result["raw"]["news_text"].strip() == "Risk Score: 4/10\nReasoning: calm"
    assert result["raw"]["pdp_text"] == ""  # supervisor decided not to consult it


async def test_run_pipeline_supervisor_can_reorder_and_repeat_experts(monkeypatch):
    """Further evidence of genuine routing: the supervisor can consult the
    same expert more than once, or in an order different from
    technical/news/pdp - this only makes sense if it's deciding at runtime,
    not following a hardcoded sequence."""
    patch_supervisor_graph(
        agents, monkeypatch,
        technical_responses=[AIMessage(content="first look"), AIMessage(content="second look")],
        news_responses=[AIMessage(content="news summary")],
        pdp_responses=[AIMessage(content="pdp summary")],
        supervisor_responses=supervisor_turns(
            "stock_news_expert", "stock_price_expert", "stock_price_expert",
            final_text="MERGED REPORT",
        ),
    )

    result = await agents.run_pipeline("THYAO.IS", tickers=["THYAO.IS"])

    assert result["output"].strip() == "MERGED REPORT"
    assert result["raw"]["technical_text"].strip() == "second look"
    assert result["raw"]["news_text"].strip() == "news summary"


async def test_run_pipeline_passes_history_context_into_initial_message(monkeypatch):
    captured = {"prompts": []}

    class CapturingModel(FakeToolCallingChatModel):
        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            captured["prompts"].append(messages)
            async for chunk in super()._astream(messages, stop, run_manager, **kwargs):
                yield chunk

    from langgraph.prebuilt import create_react_agent
    monkeypatch.setattr(agents, "stock_price_analiser_agent",
                         create_react_agent(model=FakeToolCallingChatModel(responses=[AIMessage(content="t")]), tools=[], name="stock_price_expert"))
    monkeypatch.setattr(agents, "stock_news_agent",
                         create_react_agent(model=FakeToolCallingChatModel(responses=[AIMessage(content="n")]), tools=[], name="stock_news_expert"))
    monkeypatch.setattr(agents, "pdp_agent",
                         create_react_agent(model=FakeToolCallingChatModel(responses=[AIMessage(content="p")]), tools=[], name="pdp_expert"))
    monkeypatch.setattr(agents, "model", CapturingModel(responses=supervisor_turns(
        "stock_price_expert", "stock_news_expert", "pdp_expert", final_text="ok",
    )))
    monkeypatch.setattr(agents, "supervisor_graph", agents._build_supervisor_graph())

    result = await agents.run_pipeline(
        "THYAO.IS", tickers=["THYAO.IS"],
        history_context="THYAO.IS: 3 days ago — Neutral, risk 4/10",
    )

    assert result["output"].strip() == "ok"
    assert any("3 days ago" in str(p) for p in captured["prompts"])


async def test_stream_pipeline_yields_tool_events_and_final(monkeypatch):
    from langchain_core.tools import tool

    @tool
    def fake_rsi_tool(ticker: str) -> dict:
        """fake tool"""
        return {"symbol": ticker, "signal": "bullish"}

    patch_supervisor_graph(
        agents, monkeypatch,
        technical_responses=[
            AIMessage(content="", tool_calls=[{"name": "fake_rsi_tool", "args": {"ticker": "THYAO.IS"}, "id": "1"}]),
            AIMessage(content="Technical bullish based on tool"),
        ],
        news_responses=[AIMessage(content="News risk 4")],
        pdp_responses=[AIMessage(content="PDP neutral")],
        final_text="FINAL MERGED REPORT",
        technical_tools=[fake_rsi_tool],
    )

    events = []
    async for event in agents.stream_pipeline("THYAO.IS", tickers=["THYAO.IS"]):
        events.append(event)

    types = [e["type"] for e in events]
    assert "tool_call" in types
    assert "tool_result" in types
    assert "status" in types
    assert events[-1]["type"] == "final"
    assert events[-1]["output"].strip() == "FINAL MERGED REPORT"

    tool_call_events = [e for e in events if e["type"] == "tool_call"]
    assert tool_call_events[0]["agent"] == "stock_price_expert"
    assert tool_call_events[0]["tool"] == "fake_rsi_tool"

    merge_tokens = [e for e in events if e["type"] == "token" and e["agent"] == "merge"]
    assert merge_tokens  # supervisor's own final-report call streamed token-by-token too

    # supervisor's handoff decisions surface as status updates, not raw tool_call events
    routing_statuses = [e for e in events if e["type"] == "status" and "consulting" in e["text"]]
    assert routing_statuses
