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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM -----------------------------------------------------------------
    llm_provider: LLMProvider = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:latest"
    groq_api_key: SecretStr | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)

    # --- Data providers --------------------------------------------------------
    evds_api_key: SecretStr | None = None
    evds_base_url: str = "https://evds3.tcmb.gov.tr/service/evds"
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance (cached after first read)."""
    return Settings()
