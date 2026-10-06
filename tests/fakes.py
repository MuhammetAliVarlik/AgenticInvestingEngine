"""Scripted chat model for exercising real agent graphs without an LLM."""

from __future__ import annotations

import asyncio
import itertools
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

_ids = itertools.count()


class ScriptedChatModel(BaseChatModel):
    """Returns the next scripted AIMessage on every call, in order.

    A single instance can be shared by the supervisor and all specialists
    because the supervisor graph runs its LLM turns sequentially. Streaming
    yields the content word by word, then any tool calls.
    """

    script: list[AIMessage]
    calls: int = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def _next(self) -> AIMessage:
        if self.calls >= len(self.script):
            raise AssertionError(
                f"Model called {self.calls + 1} times; script has {len(self.script)}"
            )
        message = self.script[self.calls]
        self.calls += 1
        return message

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        message = self._next()
        if message.content:
            for word in str(message.content).split(" "):
                yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))
                await asyncio.sleep(0)
        if message.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="", tool_calls=message.tool_calls)
            )

    @property
    def _llm_type(self) -> str:
        return "scripted"


def say(text: str) -> AIMessage:
    return AIMessage(content=text)


def call(tool: str, **args: Any) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": tool, "args": args, "id": f"call-{next(_ids)}"}]
    )


def handoff(agent: str) -> AIMessage:
    return call(f"transfer_to_{agent}")
