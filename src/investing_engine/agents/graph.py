"""LangGraph supervisor over MCP-provided tools.

The supervisor is a genuine decision-maker: each turn its own LLM call
chooses (through handoff tool calls) which specialist to consult next, or
writes the final report. Each specialist receives only the MCP tools it
needs (least privilege).
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel
from langgraph_supervisor import create_supervisor

from investing_engine.agents.prompts import (
    MACRO_ANALYST_PROMPT,
    NEWS_ANALYST_PROMPT,
    SUPERVISOR_PROMPT,
    TECHNICAL_ANALYST_PROMPT,
)
from investing_engine.universe import Instrument

logger = logging.getLogger(__name__)

SUPERVISOR = "supervisor"
SPECIALIST_TOOLS: dict[str, tuple[str, ...]] = {
    "technical_analyst": ("technical_snapshot",),
    "news_analyst": ("news_headlines",),
    "macro_analyst": ("macro_snapshot",),
}
SUPERVISOR_TOOLS: tuple[str, ...] = ("prediction_history",)
SPECIALISTS = tuple(SPECIALIST_TOOLS)

_SPECIALIST_PROMPTS = {
    "technical_analyst": TECHNICAL_ANALYST_PROMPT,
    "news_analyst": NEWS_ANALYST_PROMPT,
    "macro_analyst": MACRO_ANALYST_PROMPT,
}
_RISK_BLOCK = re.compile(
    r"News Risk:\s*(?P<symbol>[A-Z0-9.]+)\s*=*\s*Risk Score:\s*(?P<score>\d+(?:\.\d+)?)\s*/\s*10",
    re.IGNORECASE,
)


def _select(tools: Mapping[str, BaseTool], names: Sequence[str]) -> list[BaseTool]:
    missing = [n for n in names if n not in tools]
    if missing:
        raise RuntimeError(f"MCP server did not expose required tools: {missing}")
    return [tools[n] for n in names]


def build_graph(model: BaseChatModel, tools: Mapping[str, BaseTool]) -> CompiledStateGraph[Any]:
    specialists: list[Pregel[Any, None, Any, Any]] = [
        create_agent(
            model,
            _select(tools, SPECIALIST_TOOLS[name]),
            system_prompt=_SPECIALIST_PROMPTS[name],
            name=name,
        )
        for name in SPECIALISTS
    ]
    workflow = create_supervisor(
        specialists,
        model=model,
        tools=list[BaseTool | Callable[..., Any]](_select(tools, SUPERVISOR_TOOLS)),
        prompt=SUPERVISOR_PROMPT,
        output_mode="full_history",
        add_handoff_back_messages=False,
        supervisor_name=SUPERVISOR,
    )
    return workflow.compile()


def build_request(instruments: Sequence[Instrument]) -> str:
    listed = ", ".join(f"{i.symbol} ({i.name})" for i in instruments)
    return f"Prepare the research report for: {listed}."


# --- Result extraction ---------------------------------------------------------------


def final_text(messages: Sequence[BaseMessage], agent: str | None = None) -> str:
    """Content of the last finished answer (no pending tool calls) from ``agent``."""
    for message in reversed(messages):
        if (
            isinstance(message, AIMessage)
            and message.content
            and not message.tool_calls
            and (agent is None or message.name == agent)
        ):
            return message.text
    return ""


def tool_results(messages: Sequence[BaseMessage], tool_name: str) -> list[dict[str, Any]]:
    """Structured payloads returned by ``tool_name`` during the run (errors skipped)."""
    results: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name != tool_name:
            continue
        if message.status == "error":
            continue
        artifact = message.artifact
        if isinstance(artifact, dict) and isinstance(artifact.get("structured_content"), dict):
            results.append(artifact["structured_content"])
            continue
        try:
            parsed = json.loads(message.text)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            results.append(parsed)
    return results


def risk_scores(text: str) -> dict[str, float]:
    return {m["symbol"].upper(): float(m["score"]) for m in _RISK_BLOCK.finditer(text)}


@dataclass
class AnalysisResult:
    report: str = ""
    technical: dict[str, dict[str, Any]] = field(default_factory=dict)
    risk: dict[str, float] = field(default_factory=dict)
    specialist_text: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @classmethod
    def from_messages(cls, messages: Sequence[BaseMessage]) -> AnalysisResult:
        technical = {
            str(r["symbol"]): r
            for r in tool_results(messages, "technical_snapshot")
            if "symbol" in r
        }
        texts = {name: final_text(messages, name) for name in SPECIALISTS}
        return cls(
            report=final_text(messages, SUPERVISOR),
            technical=technical,
            risk=risk_scores(texts["news_analyst"]),
            specialist_text=texts,
        )


# --- Streaming -------------------------------------------------------------------------


async def stream_analysis(
    graph: CompiledStateGraph[Any], instruments: Sequence[Instrument]
) -> AsyncIterator[dict[str, Any]]:
    """Run the graph and yield UI events as they happen.

    Event types: ``status``, ``tool_call``, ``tool_result``, ``token``,
    ``error`` and finally ``final`` (always emitted, carrying an
    :class:`AnalysisResult`).
    """
    started = time.perf_counter()
    writing_started: float | None = None
    messages: Sequence[BaseMessage] = []
    errors: list[str] = []
    state = {"messages": [HumanMessage(build_request(instruments))]}

    try:
        stream = cast(
            "AsyncIterator[tuple[tuple[str, ...], str, Any]]",
            graph.astream(state, stream_mode=["messages", "updates"], subgraphs=True),
        )
        async for namespace, mode, item in stream:
            if mode == "updates":
                if not namespace:  # root-level deltas carry the full history
                    for update in item.values():
                        if isinstance(update, dict) and "messages" in update:
                            messages = update["messages"]
                continue

            chunk, metadata = item
            node = namespace[0].split(":")[0] if namespace else metadata.get("langgraph_node", "")
            agent = node if node in SPECIALISTS else SUPERVISOR

            if isinstance(chunk, ToolMessage):
                if not (chunk.name or "").startswith("transfer_"):
                    yield {"type": "tool_result", "agent": agent, "tool": chunk.name}
            elif getattr(chunk, "tool_calls", None):
                for call in chunk.tool_calls:
                    name = call.get("name") or ""
                    if name.startswith("transfer_to_"):
                        target = name.removeprefix("transfer_to_")
                        yield {"type": "status", "text": f"Consulting {target}"}
                    elif name and not name.startswith("transfer_"):
                        yield {"type": "tool_call", "agent": agent, "tool": name}
            elif text := getattr(chunk, "text", ""):
                if agent == SUPERVISOR and writing_started is None:
                    writing_started = time.perf_counter()
                    yield {"type": "status", "text": "Writing the report"}
                yield {"type": "token", "agent": agent, "text": text}
    except Exception as exc:
        logger.exception("Analysis graph failed")
        errors.append(type(exc).__name__)
        yield {"type": "error", "message": "The analysis could not be completed."}

    result = AnalysisResult.from_messages(messages)
    result.errors = errors
    finished = time.perf_counter()
    result.timings = {
        "total_seconds": round(finished - started, 3),
        "gathering_seconds": round((writing_started or finished) - started, 3),
    }
    yield {"type": "final", "result": result}


async def run_analysis(
    graph: CompiledStateGraph[Any], instruments: Sequence[Instrument]
) -> AnalysisResult:
    """Non-streaming convenience wrapper over :func:`stream_analysis`."""
    result = AnalysisResult()
    async for event in stream_analysis(graph, instruments):
        if event["type"] == "final":
            result = event["result"]
    return result
