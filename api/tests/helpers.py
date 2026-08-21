import asyncio
import json as _json
from typing import List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


def make_ai_message(content):
    return AIMessage(content=content)


def make_tool_message(content, name, tool_call_id="1"):
    payload = content if isinstance(content, str) else _json.dumps(content)
    return ToolMessage(content=payload, name=name, tool_call_id=tool_call_id)


class FakeToolCallingChatModel(BaseChatModel):
    """A minimal fake chat model that supports bind_tools() and both
    sync/async generation and streaming (word-by-word, then any tool
    calls) - used to exercise the real create_react_agent/create_supervisor
    machinery in tests without a live LLM. Each call to generate/stream
    consumes the next AIMessage from `responses`, so pass one response per
    expected model turn (e.g. a tool-call message, then a final-answer
    message for a worker that's expected to call a tool once)."""

    responses: List[AIMessage]
    _idx: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self.responses[self._idx]
        self._idx += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.responses[self._idx]
        self._idx += 1
        if msg.content:
            for word in msg.content.split(" "):
                yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))
                await asyncio.sleep(0)
        if msg.tool_calls:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_calls=msg.tool_calls))

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"


def make_handoff(target: str, call_id: str):
    """An AIMessage shaped like a real supervisor LLM turn that decides to
    hand off to `target` - matches the transfer_to_<agent name> tool-call
    langgraph_supervisor binds to the supervisor's model."""
    return AIMessage(content="", tool_calls=[{"name": f"transfer_to_{target}", "args": {}, "id": call_id}])


def supervisor_turns(*targets: str, final_text: str) -> List[AIMessage]:
    """Builds the sequence of turns a fake supervisor LLM emits to hand off
    to each of `targets` in order, then close with `final_text` as the
    synthesized report - this is what a test passes as `supervisor_responses`
    to exercise the real genuine-decision-making handoff graph."""
    turns = [make_handoff(target, call_id=f"handoff-{i}") for i, target in enumerate(targets)]
    turns.append(AIMessage(content=final_text))
    return turns


def patch_supervisor_graph(
    agents_module,
    monkeypatch,
    technical_responses,
    news_responses,
    pdp_responses,
    final_text: str | None = None,
    supervisor_responses: List[AIMessage] | None = None,
    technical_tools=(),
    news_tools=(),
    pdp_tools=(),
):
    """Replaces the three expert agents and the supervisor's own model with
    fresh FakeToolCallingChatModel-backed agents, then rebuilds the compiled
    supervisor graph from them - this is the "mock at the node/LLM level"
    pattern: the real create_react_agent and create_supervisor orchestration
    code all still runs (including the genuine handoff-decision machinery),
    only the LLM calls are faked. No tools are bound by default (tests that
    need to exercise tool-call/tool-result events pass in a small fake
    @tool, never the real network-calling tools).

    By default the fake supervisor model hands off to all three experts (in
    stock_price_expert, stock_news_expert, pdp_expert order) then closes
    with `final_text` - pass `supervisor_responses` explicitly (built with
    `supervisor_turns()`) for tests that need the supervisor to decide on a
    different order, skip an expert, or exercise multi-step routing.

    (Patching .astream/.ainvoke directly on an already-compiled subgraph
    node breaks LangGraph's own update-delta computation for that node -
    verified while building this - so tests must replace the whole agent
    object and rebuild the graph instead.)"""
    from langgraph.prebuilt import create_react_agent

    if supervisor_responses is None:
        if final_text is None:
            raise ValueError("patch_supervisor_graph requires final_text or supervisor_responses")
        supervisor_responses = supervisor_turns(
            "stock_price_expert", "stock_news_expert", "pdp_expert", final_text=final_text,
        )

    monkeypatch.setattr(
        agents_module, "stock_price_analiser_agent",
        create_react_agent(
            model=FakeToolCallingChatModel(responses=technical_responses),
            tools=list(technical_tools), name="stock_price_expert",
        ),
    )
    monkeypatch.setattr(
        agents_module, "stock_news_agent",
        create_react_agent(
            model=FakeToolCallingChatModel(responses=news_responses),
            tools=list(news_tools), name="stock_news_expert",
        ),
    )
    monkeypatch.setattr(
        agents_module, "pdp_agent",
        create_react_agent(
            model=FakeToolCallingChatModel(responses=pdp_responses),
            tools=list(pdp_tools), name="pdp_expert",
        ),
    )
    monkeypatch.setattr(agents_module, "model", FakeToolCallingChatModel(responses=supervisor_responses))
    monkeypatch.setattr(agents_module, "supervisor_graph", agents_module._build_supervisor_graph())
