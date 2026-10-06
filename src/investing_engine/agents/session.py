"""MCP client side: load the engine's tools for one analysis request.

The agents never import tool functions directly - they discover them over
the Model Context Protocol, exactly as an external client would. Inside the
API process the transport is an in-memory stream pair (no sockets, no
subprocess), and the caller's identity is propagated through the MCP SDK's
own auth context so the server scopes uploads to that user.

Dataset ids are deliberately kept out of the model's view: the
``dataset_id`` argument is removed from the tool schema the LLM sees and is
injected by a tool-call interceptor from server-side request state. The
model therefore cannot mistype, invent or swap another user's dataset id.
"""

from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, TextContent

from investing_engine.universe import normalize_symbol

AGENT_HIDDEN_ARGUMENTS = frozenset({"dataset_id", "document_id"})
# Tools the agents are never given: uploads happen through the API, not the LLM.
AGENT_EXCLUDED_TOOLS = frozenset({"upload_price_csv", "upload_document"})

# Marker credential for the in-memory transport; it is never checked or sent anywhere.
_IN_PROCESS_TOKEN = "in-process"  # noqa: S105

ToolHandler = Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]]


class UploadInjector:
    """Interceptor that attaches the caller's uploads to the tools that read them.

    Any id the model tries to pass is discarded; only ids granted for this
    request (and owned by this caller, as the server re-checks) are used.
    """

    def __init__(self, datasets: Mapping[str, str], documents: Mapping[str, str]) -> None:
        self._grants = {
            "technical_snapshot": ("dataset_id", _by_symbol(datasets)),
            "disclosure_document": ("document_id", _by_symbol(documents)),
        }

    async def __call__(
        self, request: MCPToolCallRequest, handler: ToolHandler
    ) -> MCPToolCallResult:
        args = {k: v for k, v in request.args.items() if k not in AGENT_HIDDEN_ARGUMENTS}
        if request.name in self._grants:
            argument, grants = self._grants[request.name]
            granted = grants.get(normalize_symbol(str(args.get("symbol", ""))))
            if granted:
                args[argument] = granted
        return await handler(request.override(args=args))


async def fill_empty_content(
    request: MCPToolCallRequest, handler: ToolHandler
) -> MCPToolCallResult:
    """Interceptor that guarantees every tool result carries at least one text block.

    FastMCP serialises an empty list return value as zero content blocks, which
    becomes an empty tool message; providers such as Groq reject those. The
    structured payload is rendered as JSON text instead.
    """
    result = await handler(request)
    if isinstance(result, CallToolResult) and not result.content:
        payload = result.structuredContent
        if isinstance(payload, dict) and set(payload) == {"result"}:
            payload = payload["result"]
        text = json.dumps(payload) if payload is not None else "No content."
        result = result.model_copy(update={"content": [TextContent(type="text", text=text)]})
    return result


def _by_symbol(mapping: Mapping[str, str]) -> dict[str, str]:
    return {normalize_symbol(k): v for k, v in mapping.items()}


def _hide_arguments(tool: BaseTool) -> BaseTool:
    schema = tool.args_schema
    if not isinstance(schema, dict):
        return tool
    properties = schema.get("properties", {})
    if not AGENT_HIDDEN_ARGUMENTS & properties.keys():
        return tool
    trimmed: dict[str, Any] = copy.deepcopy(schema)
    for name in AGENT_HIDDEN_ARGUMENTS:
        trimmed["properties"].pop(name, None)
    trimmed["required"] = [
        r for r in trimmed.get("required", []) if r not in AGENT_HIDDEN_ARGUMENTS
    ]
    return tool.model_copy(update={"args_schema": trimmed})


@asynccontextmanager
async def engine_tools(
    server: FastMCP,
    *,
    principal: str,
    datasets: Mapping[str, str] | None = None,
    documents: Mapping[str, str] | None = None,
) -> AsyncIterator[dict[str, BaseTool]]:
    """Open an MCP session as ``principal`` and yield agent-ready tools by name."""
    identity = AuthenticatedUser(
        AccessToken(
            token=_IN_PROCESS_TOKEN,
            client_id="investing-engine-api",
            scopes=[],
            subject=principal,
        )
    )
    reset_token = auth_context_var.set(identity)
    try:
        async with create_connected_server_and_client_session(server) as session:
            tools = await load_mcp_tools(
                session,
                tool_interceptors=[
                    UploadInjector(datasets or {}, documents or {}),
                    fill_empty_content,
                ],
                server_name="investing-engine",
            )
            yield {
                tool.name: _hide_arguments(tool)
                for tool in tools
                if tool.name not in AGENT_EXCLUDED_TOOLS
            }
    finally:
        auth_context_var.reset(reset_token)
