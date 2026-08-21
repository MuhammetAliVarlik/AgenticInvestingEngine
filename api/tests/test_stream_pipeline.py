from langchain_core.messages import AIMessage

import agents
from helpers import FakeToolCallingChatModel, patch_supervisor_graph, supervisor_turns

KNOWN_EVENT_TYPES = {"tool_call", "tool_result", "token", "status", "worker_error", "final"}


async def test_stream_pipeline_only_yields_known_sse_event_shapes(monkeypatch):
    """The graph's astream() is asked for both "messages" and "updates"
    modes internally (needed to recover the supervisor's final text and each
    expert's tool-results without a second graph invocation) - this checks
    the "updates" events never leak through as-is into the SSE stream, only
    the translated event shapes the UI already understands do."""
    patch_supervisor_graph(
        agents, monkeypatch,
        technical_responses=[AIMessage(content="t")],
        news_responses=[AIMessage(content="n")],
        pdp_responses=[AIMessage(content="p")],
        final_text="merged",
    )

    events = [e async for e in agents.stream_pipeline("THYAO.IS", tickers=["THYAO.IS"])]

    assert events
    assert {e["type"] for e in events} <= KNOWN_EVENT_TYPES


async def test_stream_pipeline_status_emitted_for_each_handoff_and_synthesis(monkeypatch):
    """One status event per supervisor routing decision (one per expert
    consulted), plus exactly one for the start of final-report synthesis -
    concrete, observable evidence of the supervisor's own decisions as they
    happen, not just a single fixed "merging" announcement."""
    patch_supervisor_graph(
        agents, monkeypatch,
        technical_responses=[AIMessage(content="t")],
        news_responses=[AIMessage(content="n")],
        pdp_responses=[AIMessage(content="p")],
        final_text="a longer merged report with several words",
    )

    events = [e async for e in agents.stream_pipeline("THYAO.IS", tickers=["THYAO.IS"])]

    status_events = [e for e in events if e["type"] == "status"]
    routing_statuses = [e for e in status_events if "consulting" in e["text"]]
    synthesis_statuses = [e for e in status_events if e["text"] == "Supervisor is writing the final report..."]
    assert len(routing_statuses) == 3  # one per expert handed off to
    assert len(synthesis_statuses) == 1


async def test_stream_pipeline_yields_error_event_on_graph_failure(monkeypatch):
    class BrokenModel(FakeToolCallingChatModel):
        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            raise RuntimeError("simulated graph failure")
            yield  # pragma: no cover - makes this an async generator

    from langgraph.prebuilt import create_react_agent
    monkeypatch.setattr(
        agents, "stock_price_analiser_agent",
        create_react_agent(model=BrokenModel(responses=[]), tools=[], name="stock_price_expert"),
    )
    monkeypatch.setattr(
        agents, "stock_news_agent",
        create_react_agent(model=FakeToolCallingChatModel(responses=[AIMessage(content="n")]), tools=[], name="stock_news_expert"),
    )
    monkeypatch.setattr(
        agents, "pdp_agent",
        create_react_agent(model=FakeToolCallingChatModel(responses=[AIMessage(content="p")]), tools=[], name="pdp_expert"),
    )
    monkeypatch.setattr(agents, "model", FakeToolCallingChatModel(
        responses=supervisor_turns("stock_price_expert", final_text="merged"),
    ))
    monkeypatch.setattr(agents, "supervisor_graph", agents._build_supervisor_graph())

    events = [e async for e in agents.stream_pipeline("THYAO.IS", tickers=["THYAO.IS"])]

    assert events[-1]["type"] == "final"  # always ends with a final event, even on failure
    assert any(e["type"] == "worker_error" for e in events)


async def test_stream_pipeline_tool_events_tagged_with_correct_worker(monkeypatch):
    from langchain_core.tools import tool

    @tool
    def fake_pdp_tool(ticker: str) -> dict:
        """fake tool"""
        return {"info": "no disclosures"}

    patch_supervisor_graph(
        agents, monkeypatch,
        technical_responses=[AIMessage(content="t")],
        news_responses=[AIMessage(content="n")],
        pdp_responses=[
            AIMessage(content="", tool_calls=[{"name": "fake_pdp_tool", "args": {"ticker": "THYAO.IS"}, "id": "1"}]),
            AIMessage(content="pdp final"),
        ],
        final_text="merged",
        pdp_tools=[fake_pdp_tool],
    )

    events = [e async for e in agents.stream_pipeline("THYAO.IS", tickers=["THYAO.IS"])]

    pdp_tool_calls = [e for e in events if e["type"] == "tool_call" and e["agent"] == "pdp_expert"]
    assert len(pdp_tool_calls) == 1
    assert pdp_tool_calls[0]["tool"] == "fake_pdp_tool"
    # never mislabeled as coming from a different worker
    assert not [e for e in events if e["type"] == "tool_call" and e["tool"] == "fake_pdp_tool" and e["agent"] != "pdp_expert"]


async def test_stream_pipeline_handoff_tool_calls_never_leak_as_raw_tool_events(monkeypatch):
    """The supervisor's own transfer_to_*/transfer_back_to_* bookkeeping
    must never show up as tool_call/tool_result events - those are internal
    routing plumbing, not a real analysis tool a UI should show as one."""
    patch_supervisor_graph(
        agents, monkeypatch,
        technical_responses=[AIMessage(content="t")],
        news_responses=[AIMessage(content="n")],
        pdp_responses=[AIMessage(content="p")],
        final_text="merged",
    )

    events = [e async for e in agents.stream_pipeline("THYAO.IS", tickers=["THYAO.IS"])]

    leaked = [
        e for e in events
        if e["type"] in ("tool_call", "tool_result") and e.get("tool", "").startswith("transfer_")
    ]
    assert leaked == []
