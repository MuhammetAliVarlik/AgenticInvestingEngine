import json
import logging
import sys
import types

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from pydantic import SecretStr

from investing_engine.agents.graph import AnalysisResult, token_usage
from investing_engine.api.app import create_app
from investing_engine.observability.budget import (
    BudgetExceededError,
    CircuitBreaker,
    UsageStore,
    total_tokens,
)
from investing_engine.observability.logging import JsonFormatter, request_id_var
from investing_engine.observability.tracing import MAX_TRACED_CHARS, Tracer, hash_principal, mask
from investing_engine.services import MarketData
from tests.factories import synthetic_prices
from tests.fakes import ScriptedChatModel
from tests.unit.test_agents import full_script
from tests.unit.test_services import StubEvds, StubGdelt


def test_mask_redacts_credentials_and_truncates():
    payload = {
        "prompt": "use api_key=abc123 and gsk_ABCDEFGHIJKLMNOP",
        "document": "x" * (MAX_TRACED_CHARS + 500),
        "nested": [{"token": "token: s3cr3t"}],
        "number": 4.2,
    }
    masked = mask(payload)

    assert "abc123" not in masked["prompt"]
    assert "gsk_ABCDEFGHIJKLMNOP" not in masked["prompt"]
    assert masked["document"].endswith("[500 chars omitted]")
    assert "s3cr3t" not in masked["nested"][0]["token"]
    assert masked["number"] == 4.2


def test_principal_hash_is_salted_and_stable():
    assert hash_principal("alice@example.com", "s1") == hash_principal("alice@example.com", "s1")
    assert hash_principal("alice@example.com", "s1") != hash_principal("alice@example.com", "s2")
    assert "alice" not in hash_principal("alice@example.com", "s1")


def test_tracer_is_a_noop_without_keys(settings):
    tracer = Tracer(settings)
    handle = tracer.start(principal="alice", request_id="r1", symbols=["XU100"], recursion_limit=30)

    assert not tracer.enabled
    assert handle.handler is None
    assert handle.config["recursion_limit"] == 30
    assert handle.config["metadata"]["langfuse_user_id"] != "alice"
    tracer.finish(handle, AnalysisResult())  # must not raise


def test_tracer_attaches_handler_and_scores(settings, monkeypatch):
    scores, created = [], {}

    class FakeLangfuse:
        def __init__(self, **kwargs):
            created.update(kwargs)

        def create_score(self, **kwargs):
            scores.append(kwargs)

        def flush(self):
            pass

        def shutdown(self):
            pass

    class FakeHandler:
        def __init__(self, public_key):
            self.last_trace_id = "trace-1"

    monkeypatch.setitem(sys.modules, "langfuse", types.SimpleNamespace(Langfuse=FakeLangfuse))
    monkeypatch.setitem(
        sys.modules, "langfuse.langchain", types.SimpleNamespace(CallbackHandler=FakeHandler)
    )
    configured = settings.model_copy(
        update={"langfuse_public_key": "pk-lf-1", "langfuse_secret_key": SecretStr("sk-lf-1")}
    )
    tracer = Tracer(configured)
    handle = tracer.start(principal="alice", request_id="r1", symbols=["XU100"], recursion_limit=30)
    result = AnalysisResult(
        checks={"grounding_score": 0.5, "ungrounded_figures": ["1.23"], "unexpected_symbols": []},
        injection_flags=["override"],
    )
    tracer.finish(handle, result)

    assert created["mask"] is mask
    assert isinstance(handle.config["callbacks"][0], FakeHandler)
    assert {s["name"]: s["value"] for s in scores} == {
        "grounding": 0.5,
        "scope_ok": 1.0,
        "injection_flags": 1.0,
    }


def test_usage_store_enforces_daily_limits(tmp_path):
    store = UsageStore(tmp_path / "u.db", salt="s", max_analyses=2, max_tokens=1000)
    store.check("alice")
    store.record("alice", tokens=400)
    store.record("alice", tokens=100)

    assert store.get("alice").analyses == 2
    assert store.get("alice").tokens == 500
    with pytest.raises(BudgetExceededError, match="Daily limit") as info:
        store.check("alice")
    assert 0 < info.value.retry_after <= 86_400
    store.check("bob")  # independent per principal


def test_usage_store_token_budget(tmp_path):
    store = UsageStore(tmp_path / "u.db", salt="s", max_analyses=100, max_tokens=1000)
    store.record("alice", tokens=1200)
    with pytest.raises(BudgetExceededError, match="token budget"):
        store.check("alice")


def test_circuit_breaker_opens_and_closes(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(
        "investing_engine.observability.budget.time.monotonic", lambda: clock["now"]
    )
    breaker = CircuitBreaker(cooldown=60)
    breaker.check()
    breaker.trip()
    with pytest.raises(BudgetExceededError) as info:
        breaker.check()
    assert info.value.retry_after == 61
    clock["now"] += 61
    breaker.check()


def test_token_usage_is_aggregated_per_agent():
    def turn(name: str, inp: int, out: int) -> AIMessage:
        usage = {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}
        return AIMessage(content="x", name=name, usage_metadata=usage)

    messages = [
        turn("supervisor", 100, 10),
        turn("news_analyst", 50, 20),
        turn("supervisor", 200, 30),
    ]
    usage = token_usage(messages)
    assert usage["supervisor"] == {"input_tokens": 300, "output_tokens": 40, "calls": 2}
    assert total_tokens(usage) == 410


def test_json_log_lines_carry_request_id():
    record = logging.makeLogRecord({"msg": "hello", "levelname": "INFO", "name": "t", "path": "/x"})
    token = request_id_var.set("req42")
    try:
        line = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert line["request_id"] == "req42"
    assert line["path"] == "/x"


@pytest.fixture
def market(settings):
    service = MarketData.from_settings(settings)
    service.gdelt = StubGdelt()
    service.evds = StubEvds(synthetic_prices(400, ohlcv=False))
    return service


def test_api_enforces_quota_and_reports_usage(settings, market):
    limited = settings.model_copy(update={"daily_analyses_per_user": 1, "cache_ttl_seconds": 0})
    scripts = [ScriptedChatModel(script=full_script("XU100"))]
    app = create_app(limited, market=market, model_factory=lambda: scripts.pop(0))
    with TestClient(app) as client:
        first = client.post("/analyses", json={"symbols": ["XU100"]})
        assert first.status_code == 200
        assert first.headers["X-Request-ID"]

        client.app.state.cache.clear()  # cache hits are free; force a fresh analysis
        second = client.post("/analyses", json={"symbols": ["XU100"]})
        assert second.status_code == 429
        assert int(second.headers["Retry-After"]) > 0
        assert client.post("/analyses/stream", json={"symbols": ["XU100"]}).status_code == 429
        assert client.get("/usage").json()["analyses"] == 1


def test_rate_limited_provider_trips_the_breaker(settings, market):
    class RateLimitError(Exception):
        pass

    class RateLimitedModel(ScriptedChatModel):
        def _next(self):
            raise RateLimitError("429")

    models = [RateLimitedModel(script=[])]
    app = create_app(settings, market=market, model_factory=lambda: models.pop(0))
    with TestClient(app) as client:
        assert client.post("/analyses", json={"symbols": ["XU100"]}).status_code == 502
        blocked = client.post("/analyses", json={"symbols": ["XU100"]})
    assert blocked.status_code == 429
    assert "rate-limited" in blocked.json()["detail"]
