import sys
import types

import pandas as pd
import pytest
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from investing_engine.agents.llm import LLMConfigurationError, build_chat_model
from investing_engine.config import Settings
from investing_engine.providers.base import ProviderError
from investing_engine.providers.yahoo import YahooPriceProvider
from tests.factories import synthetic_prices


def test_ollama_is_the_default_provider():
    model = build_chat_model(Settings(_env_file=None))
    assert isinstance(model, ChatOllama)


def test_groq_requires_a_key():
    with pytest.raises(LLMConfigurationError):
        build_chat_model(Settings(_env_file=None, llm_provider="groq"))


def test_groq_model_keeps_the_key_secret():
    settings = Settings(_env_file=None, llm_provider="groq", groq_api_key="gsk_test_secret")
    model = build_chat_model(settings)
    assert isinstance(model, ChatGroq)
    assert "gsk_test_secret" not in repr(model)
    assert "gsk_test_secret" not in repr(settings)


def _fake_yfinance(monkeypatch, frame):
    module = types.ModuleType("yfinance")
    module.download = lambda *args, **kwargs: frame
    monkeypatch.setitem(sys.modules, "yfinance", module)


def test_yahoo_provider_appends_exchange_suffix(monkeypatch):
    captured = {}
    module = types.ModuleType("yfinance")

    def download(ticker, **kwargs):
        captured["ticker"] = ticker
        return synthetic_prices(120)

    module.download = download
    monkeypatch.setitem(sys.modules, "yfinance", module)

    frame = YahooPriceProvider().get_history("THYAO", lookback_days=200)
    assert captured["ticker"] == "THYAO.IS"
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_yahoo_provider_is_never_deployable():
    assert YahooPriceProvider.source.deployable is False


def test_yahoo_empty_result_is_a_provider_error(monkeypatch):
    _fake_yfinance(monkeypatch, pd.DataFrame())
    with pytest.raises(ProviderError, match="no data"):
        YahooPriceProvider().get_history("THYAO", lookback_days=200)


def test_reasoning_effort_is_sent_to_reasoning_models_only():
    reasoning = build_chat_model(
        Settings(
            _env_file=None, llm_provider="groq", groq_api_key="k", groq_model="openai/gpt-oss-120b"
        )
    )
    plain = build_chat_model(
        Settings(
            _env_file=None, llm_provider="groq", groq_api_key="k", groq_model="llama-3.1-8b-instant"
        )
    )
    assert reasoning.reasoning_effort == "low"
    assert plain.reasoning_effort is None
