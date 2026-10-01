"""LLM tracing with Langfuse (OpenTelemetry-based), privacy-first.

Tracing is active only when Langfuse keys are configured and the optional
``observability`` extra is installed; otherwise every call is a no-op.

What leaves the process:
* the caller's identity only as a salted SHA-256 hash;
* prompts and tool payloads with long strings truncated and anything that
  looks like a credential redacted (``mask``), so uploaded documents are
  never shipped to the tracing backend in full;
* per-analysis scores: numeric grounding, scope check, injection flags.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.runnables import RunnableConfig

from investing_engine.config import Settings

if TYPE_CHECKING:
    from investing_engine.agents.graph import AnalysisResult

logger = logging.getLogger(__name__)

MAX_TRACED_CHARS = 2000
_SECRET_PATTERN = re.compile(
    r"\b(gsk_[A-Za-z0-9]{8,}|sk-[A-Za-z0-9-]{8,}|pk-lf-[A-Za-z0-9-]+|sk-lf-[A-Za-z0-9-]+)\b"
    r"|(?i:\b(api[_-]?key|token|secret|password)\b\s*[:=]\s*\S+)"
)


def hash_principal(principal: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{principal}".encode()).hexdigest()[:16]


def mask(data: Any, **_: Any) -> Any:
    """Redact credentials and truncate long strings in traced payloads."""
    if isinstance(data, str):
        redacted = _SECRET_PATTERN.sub("[redacted]", data)
        if len(redacted) > MAX_TRACED_CHARS:
            omitted = len(redacted) - MAX_TRACED_CHARS
            return f"{redacted[:MAX_TRACED_CHARS]}… [{omitted} chars omitted]"
        return redacted
    if isinstance(data, dict):
        return {k: mask(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [mask(v) for v in data]
    return data


@dataclass
class TraceHandle:
    config: RunnableConfig
    handler: Any | None = None


class Tracer:
    def __init__(self, settings: Settings) -> None:
        self._salt = settings.telemetry_salt.get_secret_value()
        self._client: Any | None = None
        if settings.langfuse_public_key and settings.langfuse_secret_key:
            try:
                from langfuse import Langfuse
            except ImportError:
                logger.warning("Langfuse keys set but the observability extra is not installed")
                return
            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key.get_secret_value(),
                base_url=settings.langfuse_base_url,
                environment=settings.environment,
                mask=mask,
            )
            self._public_key = settings.langfuse_public_key

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def start(
        self, *, principal: str, request_id: str, symbols: list[str], recursion_limit: int
    ) -> TraceHandle:
        """Build the LangGraph run config, attaching the Langfuse handler when enabled."""
        config: RunnableConfig = {
            "recursion_limit": recursion_limit,
            "run_name": "analysis",
            "metadata": {
                "langfuse_user_id": hash_principal(principal, self._salt),
                "langfuse_session_id": request_id,
                "langfuse_tags": ["analysis", *symbols],
            },
        }
        if not self.enabled:
            return TraceHandle(config)
        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler(public_key=self._public_key)
        config["callbacks"] = [handler]
        return TraceHandle(config, handler)

    def finish(self, handle: TraceHandle, result: AnalysisResult) -> None:
        """Attach quality scores to the trace and flush. Never raises."""
        if self._client is None or handle.handler is None:
            return
        trace_id = getattr(handle.handler, "last_trace_id", None)
        try:
            if trace_id and result.checks:
                self._client.create_score(
                    trace_id=trace_id,
                    name="grounding",
                    value=float(result.checks["grounding_score"]),
                    comment=", ".join(result.checks["ungrounded_figures"]) or None,
                )
                self._client.create_score(
                    trace_id=trace_id,
                    name="scope_ok",
                    value=0.0 if result.checks["unexpected_symbols"] else 1.0,
                )
            if trace_id:
                self._client.create_score(
                    trace_id=trace_id,
                    name="injection_flags",
                    value=float(len(result.injection_flags)),
                    comment=", ".join(result.injection_flags) or None,
                )
            self._client.flush()
        except Exception:
            logger.warning("Failed to record trace scores", exc_info=True)

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.shutdown()
