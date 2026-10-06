"""Centralised, typed application settings.

Every value is read from the environment (or a local ``.env`` file), so the
same container image runs unchanged on a laptop, Azure Container Apps or
Hugging Face Spaces. Secrets are typed as :class:`SecretStr` so they never
appear in logs, reprs or tracebacks.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["ollama", "groq"]
AuthMode = Literal["none", "trusted-proxy"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "development"
    log_level: str = "INFO"
    json_logs: bool = True

    # --- Access control ----------------------------------------------------------
    # "none" for local use; "trusted-proxy" when deployed behind the UI, which
    # authenticates users and calls the API with INTERNAL_API_TOKEN.
    auth_mode: AuthMode = "none"
    internal_api_token: SecretStr | None = None
    # Comma-separated identities, e.g. "you@example.com,github:octocat".
    allowed_users: str = ""
    requests_per_minute: int = Field(default=60, ge=1)

    # --- LLM -----------------------------------------------------------------
    llm_provider: LLMProvider = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:latest"
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_output_tokens: int = Field(default=1024, ge=128, le=8192)
    # Retries after a provider rate limit. The Groq client waits as long as the
    # Retry-After header asks, so a short per-minute token limit (free tier)
    # slows an analysis down instead of failing it.
    llm_max_retries: int = Field(default=6, ge=0, le=10)
    max_graph_steps: int = Field(
        default=40, ge=10, le=200, description="LangGraph recursion limit per analysis."
    )

    # --- Data providers --------------------------------------------------------
    evds_api_key: SecretStr | None = None
    evds_base_url: str = "https://evds3.tcmb.gov.tr/igmevdsms-dis"
    gdelt_base_url: str = "https://api.gdeltproject.org/api/v2/doc/doc"
    # Yahoo Finance is personal-use only; it must stay disabled in any
    # deployed build. See DATA_SOURCES.md.
    enable_yfinance: bool = False
    http_timeout_seconds: float = Field(default=15.0, gt=0)

    # --- Analysis ----------------------------------------------------------------
    lookback_days: int = Field(default=400, ge=120, le=2000)
    model_dir: str = "models"
    model_max_age_hours: float = Field(default=24.0, gt=0)

    # --- Persistence & caching -----------------------------------------------------
    db_path: str = "data/predictions.db"
    cache_ttl_seconds: int = Field(default=600, ge=0)

    # --- Request limits --------------------------------------------------------------
    max_symbols_per_request: int = Field(default=3, ge=1, le=10)
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, gt=0)
    max_csv_rows: int = Field(default=5000, gt=0)
    max_document_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_document_pages: int = Field(default=30, ge=1, le=200)
    ocr_timeout_seconds: float = Field(default=30.0, gt=0)
    max_document_chars_for_model: int = Field(default=12_000, ge=1000)

    # --- Guardrails --------------------------------------------------------------------
    # Model-based injection classifier on Groq, layered on the heuristic scanner.
    enable_prompt_guard: bool = False
    prompt_guard_model: str = "meta-llama/llama-prompt-guard-2-86m"
    prompt_guard_threshold: float = Field(default=0.5, gt=0, lt=1)

    # --- Observability -----------------------------------------------------------------
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    # Salt for hashing user identities in traces and usage counters. Set a
    # random value in every deployment.
    telemetry_salt: SecretStr = SecretStr("local-development-salt")
    # Send prompt and tool text to the tracing backend. Turn off for public
    # deployments: traces then keep structure, timings, token counts and scores
    # but no content, so nothing a user typed or uploaded leaves the service.
    trace_content: bool = True

    # --- Budgets -----------------------------------------------------------------------
    daily_analyses_per_user: int = Field(default=20, ge=1)
    daily_tokens_per_user: int = Field(default=200_000, ge=1000)
    rate_limit_cooldown_seconds: float = Field(default=60.0, ge=1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance (cached after first read)."""
    return Settings()
