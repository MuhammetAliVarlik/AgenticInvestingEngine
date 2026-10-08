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
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel
from langgraph_supervisor import create_supervisor

from investing_engine.agents.prompts import (
    DISCLOSURE_ANALYST_PROMPT,
    MACRO_ANALYST_PROMPT,
    NEWS_ANALYST_PROMPT,
    SUPERVISOR_PROMPT,
    TECHNICAL_ANALYST_PROMPT,
)
from investing_engine.guardrails.report import check_report, collect_facts
from investing_engine.universe import Instrument

logger = logging.getLogger(__name__)

SUPERVISOR = "supervisor"
SPECIALIST_TOOLS: dict[str, tuple[str, ...]] = {
    "technical_analyst": ("technical_snapshot",),
    "news_analyst": ("news_headlines",),
    "macro_analyst": ("macro_snapshot",),
    "disclosure_analyst": ("disclosure_document",),
}
SUPERVISOR_TOOLS: tuple[str, ...] = ("prediction_history",)
SPECIALISTS = tuple(SPECIALIST_TOOLS)

_SPECIALIST_PROMPTS = {
    "technical_analyst": TECHNICAL_ANALYST_PROMPT,
    "news_analyst": NEWS_ANALYST_PROMPT,
    "macro_analyst": MACRO_ANALYST_PROMPT,
    "disclosure_analyst": DISCLOSURE_ANALYST_PROMPT,
}
# One block per symbol: "News Risk: <SYMBOL>" followed (before the next block)
# by "Risk Score: <n>/10". Models often add Markdown (**bold**, headings) or
# small variations ("Risk score (1-10): 4", "4 out of 10"); markup is removed
# before matching and the score must be within 0-10.
_RISK_BLOCK = re.compile(
    r"News\s+Risk\s*[:\-\u2013]\s*(?P<symbol>[A-Z0-9.]{2,10})\b"
    r"(?:(?!News\s+Risk).){0,400}?"
    r"Risk\s+Score\s*(?:\([^)]{0,20}\))?\s*[:=]\s*(?P<score>\d{1,2}(?:[.,]\d+)?)",
    re.IGNORECASE | re.DOTALL,
)
_MARKUP = re.compile(r"[*_`#>]+")


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


def build_request(instruments: Sequence[Instrument], *, documents_for: Sequence[str] = ()) -> str:
    listed = ", ".join(f"{i.symbol} ({i.name})" for i in instruments)
    request = f"Prepare the research report for: {listed}."
    if documents_for:
        request += f" Disclosure documents were provided for: {', '.join(documents_for)}."
    else:
        request += " No disclosure documents were provided."
    return request


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
    """Risk score per symbol from the news analyst's answer (0-10, else ignored)."""
    scores: dict[str, float] = {}
    for match in _RISK_BLOCK.finditer(_MARKUP.sub("", text)):
        score = float(match["score"].replace(",", "."))
        if 0 <= score <= 10:
            scores[match["symbol"].upper().removesuffix(".IS")] = score
    if text and not scores and "unavailable" not in text.lower():
        # Content is not logged (it can quote third-party headlines); length is enough
        # to tell "no answer" from "answer in an unexpected format".
        logger.warning(
            "News analyst answered without a readable risk score", extra={"chars": len(text)}
        )
    return scores


@dataclass
class AnalysisResult:
    report: str = ""
    technical: dict[str, dict[str, Any]] = field(default_factory=dict)
    risk: dict[str, float] = field(default_factory=dict)
    specialist_text: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    usage: dict[str, dict[str, int]] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)
    injection_flags: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @classmethod
    def from_messages(
        cls, messages: Sequence[BaseMessage], *, requested: Sequence[str] = ()
    ) -> AnalysisResult:
        technical = {
            str(r["symbol"]): r
            for r in tool_results(messages, "technical_snapshot")
            if "symbol" in r
        }
        texts = {name: final_text(messages, name) for name in SPECIALISTS}
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

        flags: set[str] = set()
        for name in ("news_headlines", "disclosure_document"):
            for payload in tool_results(messages, name):
                flags.update(payload.get("injection_flags") or [])

        report, check = check_report(
            final_text(messages, SUPERVISOR),
            requested=requested,
            facts=collect_facts(
                [m.artifact for m in tool_messages] + [m.text for m in tool_messages]
            ),
        )
        return cls(
            report=report,
            technical=technical,
            risk=risk_scores(texts["news_analyst"]),
            specialist_text=texts,
            usage=token_usage(messages),
            checks=check.as_dict() if report else {},
            injection_flags=sorted(flags),
        )


def token_usage(messages: Sequence[BaseMessage]) -> dict[str, dict[str, int]]:
    """Input/output tokens per agent, from the provider's usage metadata."""
    usage: dict[str, dict[str, int]] = {}
    for message in messages:
        if not isinstance(message, AIMessage) or not message.usage_metadata:
            continue
        agent = message.name if message.name in SPECIALISTS else SUPERVISOR
        totals = usage.setdefault(agent, {"input_tokens": 0, "output_tokens": 0, "calls": 0})
        totals["input_tokens"] += message.usage_metadata.get("input_tokens", 0)
        totals["output_tokens"] += message.usage_metadata.get("output_tokens", 0)
        totals["calls"] += 1
    return usage


# --- Streaming -------------------------------------------------------------------------


async def stream_analysis(
    graph: CompiledStateGraph[Any],
    instruments: Sequence[Instrument],
    *,
    documents_for: Sequence[str] = (),
    config: RunnableConfig | None = None,
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
    state = {"messages": [HumanMessage(build_request(instruments, documents_for=documents_for))]}

    try:
        stream = cast(
            "AsyncIterator[tuple[tuple[str, ...], str, Any]]",
            graph.astream(state, config, stream_mode=["messages", "updates"], subgraphs=True),
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

    result = AnalysisResult.from_messages(messages, requested=[i.symbol for i in instruments])
    result.errors = errors
    finished = time.perf_counter()
    result.timings = {
        "total_seconds": round(finished - started, 3),
        "gathering_seconds": round((writing_started or finished) - started, 3),
    }
    yield {"type": "final", "result": result}


async def run_analysis(
    graph: CompiledStateGraph[Any],
    instruments: Sequence[Instrument],
    *,
    documents_for: Sequence[str] = (),
) -> AnalysisResult:
    """Non-streaming convenience wrapper over :func:`stream_analysis`."""
    result = AnalysisResult()
    async for event in stream_analysis(graph, instruments, documents_for=documents_for):
        if event["type"] == "final":
            result = event["result"]
    return result
