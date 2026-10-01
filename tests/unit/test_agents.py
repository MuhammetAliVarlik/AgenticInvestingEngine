"""Supervisor graph tests: real LangGraph routing over real MCP tools, scripted LLM."""

import pytest

from investing_engine.agents.graph import (
    SPECIALISTS,
    build_graph,
    risk_scores,
    run_analysis,
    stream_analysis,
)
from investing_engine.agents.session import engine_tools
from investing_engine.mcp_server.server import build_server
from investing_engine.services import MarketData
from investing_engine.universe import resolve
from tests.factories import synthetic_prices
from tests.fakes import ScriptedChatModel, call, handoff, say
from tests.unit.test_services import StubEvds, StubGdelt

NEWS_BLOCK = "=== News Risk: {symbol} ===\nRisk Score: {score}/10\nReasoning: steady coverage."


@pytest.fixture
def market(settings):
    service = MarketData.from_settings(settings)
    service.gdelt = StubGdelt()
    service.evds = StubEvds(synthetic_prices(400, ohlcv=False))
    return service


@pytest.fixture
def server(market):
    return build_server(market)


def _csv() -> bytes:
    frame = synthetic_prices(300).round(4)
    frame.index.name = "Date"
    return frame.to_csv().encode()


def full_script(symbol: str, risk: int = 4) -> list:
    return [
        call("prediction_history", symbol=symbol),
        handoff("technical_analyst"),
        call("technical_snapshot", symbol=symbol),
        say(f"{symbol} trades above its EMA34."),
        handoff("news_analyst"),
        call("news_headlines", symbol=symbol),
        say(NEWS_BLOCK.format(symbol=symbol, score=risk)),
        handoff("macro_analyst"),
        call("macro_snapshot"),
        say("Macro data unavailable."),
        say(f"## {symbol} report\n**Disclaimer:** not investment advice."),
    ]


async def test_agents_discover_tools_over_mcp_with_least_privilege(server):
    async with engine_tools(server, principal="alice") as tools:
        assert "upload_price_csv" not in tools
        schema = tools["technical_snapshot"].args_schema
        assert "dataset_id" not in schema["properties"]
        assert schema["required"] == ["symbol"]

        graph = build_graph(ScriptedChatModel(script=[]), tools)
    assert set(SPECIALISTS) <= set(graph.nodes)


async def test_full_analysis_routes_through_every_specialist(server):
    model = ScriptedChatModel(script=full_script("XU100", risk=3))
    async with engine_tools(server, principal="alice") as tools:
        result = await run_analysis(build_graph(model, tools), [resolve("XU100")])

    assert model.calls == len(model.script)
    assert result.report.startswith("## XU100 report")
    assert result.technical["XU100"]["source"] == "TCMB EVDS"
    assert result.risk == {"XU100": 3.0}
    assert result.specialist_text["technical_analyst"].strip() == "XU100 trades above its EMA34."
    assert result.errors == []


async def test_supervisor_may_skip_specialists(server):
    script = [
        call("prediction_history", symbol="XU100"),
        handoff("news_analyst"),
        call("news_headlines", symbol="XU100"),
        say(NEWS_BLOCK.format(symbol="XU100", score=7)),
        say("## XU100 news-only report"),
    ]
    model = ScriptedChatModel(script=script)
    async with engine_tools(server, principal="alice") as tools:
        result = await run_analysis(build_graph(model, tools), [resolve("XU100")])

    assert result.technical == {}
    assert result.risk == {"XU100": 7.0}
    assert result.specialist_text["technical_analyst"] == ""


async def test_uploaded_dataset_is_injected_for_its_owner_only(server, market):
    dataset = market.register_prices(owner="alice", symbol="THYAO", raw=_csv())["dataset_id"]

    model = ScriptedChatModel(script=full_script("THYAO"))
    async with engine_tools(server, principal="alice", datasets={"THYAO": dataset}) as tools:
        result = await run_analysis(build_graph(model, tools), [resolve("THYAO")])
    assert result.technical["THYAO"]["source"] == "User-supplied file"

    # Same dataset id, different principal: the server refuses it.
    model = ScriptedChatModel(script=full_script("THYAO"))
    async with engine_tools(server, principal="mallory", datasets={"THYAO": dataset}) as tools:
        result = await run_analysis(build_graph(model, tools), [resolve("THYAO")])
    assert result.technical == {}


async def test_model_supplied_dataset_id_is_ignored(server, market):
    dataset = market.register_prices(owner="alice", symbol="THYAO", raw=_csv())["dataset_id"]
    script = full_script("THYAO")
    script[2] = call("technical_snapshot", symbol="THYAO", dataset_id=dataset)

    model = ScriptedChatModel(script=script)
    async with engine_tools(server, principal="alice") as tools:  # no datasets granted
        result = await run_analysis(build_graph(model, tools), [resolve("THYAO")])
    assert result.technical == {}


async def test_stream_emits_progress_then_final(server):
    model = ScriptedChatModel(script=full_script("XU100"))
    async with engine_tools(server, principal="alice") as tools:
        events = [e async for e in stream_analysis(build_graph(model, tools), [resolve("XU100")])]

    kinds = [e["type"] for e in events]
    assert kinds[-1] == "final"
    assert {"text": "Consulting technical_analyst", "type": "status"} in events
    assert {
        "type": "tool_call",
        "agent": "technical_analyst",
        "tool": "technical_snapshot",
    } in events
    assert {"type": "tool_result", "agent": "news_analyst", "tool": "news_headlines"} in events
    assert {"type": "status", "text": "Writing the report"} in events
    supervisor_tokens = [e for e in events if e["type"] == "token" and e["agent"] == "supervisor"]
    assert "".join(e["text"] for e in supervisor_tokens).startswith("## XU100")
    assert events[-1]["result"].timings["total_seconds"] >= 0


async def test_graph_failure_yields_error_and_final(server):
    model = ScriptedChatModel(script=[])  # first LLM call raises
    async with engine_tools(server, principal="alice") as tools:
        events = [e async for e in stream_analysis(build_graph(model, tools), [resolve("XU100")])]

    assert events[-2] == {"type": "error", "message": "The analysis could not be completed."}
    assert events[-1]["result"].errors == ["AssertionError"]


def test_risk_scores_are_parsed_per_symbol():
    text = (
        NEWS_BLOCK.format(symbol="THYAO", score=2)
        + "\n"
        + NEWS_BLOCK.format(symbol="TUPRS", score=8.5)
    )
    assert risk_scores(text) == {"THYAO": 2.0, "TUPRS": 8.5}
