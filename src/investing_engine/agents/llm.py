"""Chat-model factory: one switch between local Ollama and hosted Groq."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from investing_engine.config import Settings


class LLMConfigurationError(RuntimeError):
    pass


def _is_reasoning_model(model: str) -> bool:
    return "gpt-oss" in model or model.startswith("qwen/qwen3")


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Return the chat model selected by ``LLM_PROVIDER``.

    Imports are local so a deployment only needs the client library it uses.
    """
    if settings.llm_provider == "groq":
        if settings.groq_api_key is None:
            raise LLMConfigurationError("LLM_PROVIDER=groq requires GROQ_API_KEY")
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.max_output_tokens,
            max_retries=settings.llm_max_retries,
            # Only reasoning models accept this parameter; others reject it.
            reasoning_effort=(
                settings.llm_reasoning_effort if _is_reasoning_model(settings.groq_model) else None
            ),
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        num_predict=settings.max_output_tokens,
    )
