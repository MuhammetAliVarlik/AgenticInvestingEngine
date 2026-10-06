import json

import pytest
from fastapi.testclient import TestClient

from investing_engine.api.app import _section_for, create_app
from investing_engine.services import MarketData
from tests.factories import synthetic_prices
from tests.fakes import ScriptedChatModel
from tests.unit.test_agents import full_script
from tests.unit.test_services import StubEvds, StubGdelt


@pytest.fixture
def market(settings):
    service = MarketData.from_settings(settings)
    service.gdelt = StubGdelt()
    service.evds = StubEvds(synthetic_prices(400, ohlcv=False))
    return service


@pytest.fixture
def models():
    """Scripts queued for successive analyses; one model is built per request."""
    return []


@pytest.fixture
def client(settings, market, models):
    app = create_app(settings, market=market, model_factory=lambda: models.pop(0))
    with TestClient(app) as test_client:
        yield test_client


def _csv() -> bytes:
    frame = synthetic_prices(300).round(4)
    frame.index.name = "Date"
    return frame.to_csv().encode()


def _sse(response) -> list[dict]:
    return [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]


def test_reference_endpoints(client):
    assert client.get("/healthz").json()["status"] == "ok"
    symbols = {i["symbol"] for i in client.get("/instruments").json()}
    assert {"XU100", "THYAO"} <= symbols
    assert "TCMB EVDS" in {s["name"] for s in client.get("/sources").json()}


def test_analysis_returns_structured_result_and_persists_history(client, models):
    models.append(ScriptedChatModel(script=full_script("XU100", risk=3)))
    body = client.post("/analyses", json={"symbols": ["xu100"]}).json()

    assert body["report"].startswith("## XU100")
    assert body["technical"]["XU100"]["source"] == "TCMB EVDS"
    assert body["risk"] == {"XU100": 3.0}
    assert body["cached"] is False

    history = client.get("/history/XU100").json()
    assert len(history) == 1
    assert history[0]["risk_score"] == 3.0
    assert history[0]["signal"] == body["technical"]["XU100"]["signal"]


def test_identical_request_is_served_from_cache(client, models):
    models.append(ScriptedChatModel(script=full_script("XU100")))
    client.post("/analyses", json={"symbols": ["XU100"]})
    cached = client.post("/analyses", json={"symbols": ["xu100.is"]}).json()

    assert cached["cached"] is True
    assert not models  # no second model was requested
    assert len(client.get("/history/XU100").json()) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"symbols": []},
        {"symbols": ["ZZZZ"]},
        {"symbols": ["THYAO", "TUPRS", "ASELS", "AKBNK"]},
        {"symbols": ["XU100; drop table"]},
    ],
)
def test_invalid_requests_are_rejected_before_any_llm_call(client, payload):
    assert client.post("/analyses", json=payload).status_code == 422
    assert client.post("/analyses/stream", json=payload).status_code == 422


def test_upload_then_analyse_equity(client, models):
    upload = client.post("/datasets", data={"symbol": "THYAO"}, files={"file": ("p.csv", _csv())})
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]

    models.append(ScriptedChatModel(script=full_script("THYAO")))
    body = client.post(
        "/analyses", json={"symbols": ["THYAO"], "datasets": {"THYAO": dataset_id}}
    ).json()
    assert body["technical"]["THYAO"]["source"] == "User-supplied file"
    # Output derived from a private upload is recorded for its owner only.
    rows = client.get("/history/THYAO").json()
    assert len(rows) == 1
    assert rows[0]["private"] is True
    assert "owner" not in rows[0]


def test_upload_rejects_bad_files(client):
    response = client.post(
        "/datasets", data={"symbol": "THYAO"}, files={"file": ("x.csv", b"\x00\x01binary")}
    )
    assert response.status_code == 422


def test_technical_endpoint_needs_no_llm(client):
    body = client.get("/technical/XU100").json()
    assert body["feature_set"] == "close"
    assert client.get("/technical/THYAO").status_code == 422


def test_stream_emits_progress_and_final(client, models):
    models.append(ScriptedChatModel(script=full_script("XU100")))
    with client.stream("POST", "/analyses/stream", json={"symbols": ["XU100"]}) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _sse(response)

    assert events[-1]["type"] == "final"
    assert events[-1]["technical"]["XU100"]["symbol"] == "XU100"
    assert any(e["type"] == "tool_call" for e in events)


def test_failed_analysis_returns_502_and_writes_nothing(client, models):
    models.append(ScriptedChatModel(script=[]))
    assert client.post("/analyses", json={"symbols": ["XU100"]}).status_code == 502
    assert client.get("/history/XU100").json() == []


def test_section_extraction():
    report = "## THYAO - THY\nbody one\n## TUPRS - Tüpraş\nbody two\n**Sources:** x"
    assert _section_for(report, "THYAO") == "## THYAO - THY\nbody one"
    assert _section_for(report, "TUPRS").startswith("## TUPRS")
    assert _section_for(report, "XU100") == report
