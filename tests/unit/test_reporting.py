import io

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from investing_engine.analysis.indicators import compute_features
from investing_engine.analysis.technical import analyze
from investing_engine.api.app import create_app, principal
from investing_engine.reporting.markdown import to_html
from investing_engine.reporting.pdf import (
    BlockedFetchError,
    InstrumentSection,
    refuse_fetch,
    render_html,
    render_pdf,
)
from investing_engine.services import MarketData
from tests.factories import synthetic_prices
from tests.fakes import ScriptedChatModel
from tests.unit.test_agents import full_script
from tests.unit.test_services import StubEvds, StubGdelt

HOSTILE_REPORT = (
    "## THYAO - THY\n"
    "**Technical view:** neutral <script>alert('x')</script>\n"
    '<img src="file:///etc/passwd"> <link rel="stylesheet" href="http://attacker.example/x.css">\n'
    "- see [details](http://attacker.example/steal)\n"
    "![tracker](http://attacker.example/pixel.png)"
)


def _section(with_series: bool = True) -> InstrumentSection:
    prices = synthetic_prices(300)
    features, _ = compute_features(prices)
    technical = analyze(prices, symbol="THYAO", source="User-supplied file").model_dump(mode="json")
    return InstrumentSection(
        symbol="THYAO",
        name="Türk Hava Yolları",
        technical=technical,
        risk=4.0,
        series=features[["Close", "ema34", "ema89", "bb_high", "bb_low", "rsi"]].tail(120)
        if with_series
        else None,
        history=[
            {"timestamp": "2026-09-01T00:00:00+00:00", "risk_score": 3},
            {"timestamp": "2026-09-08T00:00:00+00:00", "risk_score": 5},
        ],
    )


def _render(**overrides):
    kwargs = {
        "report": HOSTILE_REPORT,
        "sections": [_section()],
        "checks": {"grounding_score": 0.75, "ungrounded_figures": ["99.99"]},
        "injection_flags": ["override"],
        "usage": {"supervisor": {"input_tokens": 1000, "output_tokens": 200}},
        "sources": [{"attribution": "News metadata: The GDELT Project."}],
        "reference": "ref123",
    }
    kwargs.update(overrides)
    return kwargs


def test_markdown_converter_escapes_everything_but_its_own_tags():
    html = to_html(HOSTILE_REPORT)

    assert "<script" not in html
    assert "<img" not in html
    assert "<link" not in html
    assert 'href="' not in html  # raw markup survives only as inert, escaped text
    assert 'src="' not in html
    assert "attacker.example/steal" not in html  # Markdown link targets are dropped
    assert "pixel.png" not in html
    assert "&lt;script&gt;" in html
    assert "<h3>THYAO - THY</h3>" in html
    assert "<strong>Technical view:</strong>" in html
    assert "<li>see details</li>" in html


def test_html_contains_charts_tiles_and_warnings():
    html = render_html(**_render())

    assert html.count("<svg") == 3
    assert "Neutral" in html or "Bullish" in html or "Bearish" in html
    assert "99.99" in html  # ungrounded figure called out
    assert "override" in html
    assert "1,200" in html  # tokens tile
    assert "not investment advice" in html


def test_pdf_renders_and_contains_expected_text():
    pdf = render_pdf(**_render())
    reader = PdfReader(io.BytesIO(pdf))
    text = "\n".join(page.extract_text() for page in reader.pages)

    assert pdf.startswith(b"%PDF")
    assert len(reader.pages) >= 2
    assert "Market research report" in text
    assert "THYAO" in text
    assert "Türk Hava Yolları" in text
    assert "alert('x')" in text  # rendered as inert text, not executed or dropped silently


def test_renderer_never_fetches_resources(monkeypatch):
    requested = []

    def spy(url, *args, **kwargs):
        requested.append(url)
        return refuse_fetch(url)

    monkeypatch.setattr("investing_engine.reporting.pdf.refuse_fetch", spy)
    render_pdf(**_render())
    assert requested == []  # escaped markup produces no fetchable references


def test_fetcher_refuses_everything():
    for url in ("file:///etc/passwd", "http://169.254.169.254/latest/meta-data", "data:,x"):
        with pytest.raises(BlockedFetchError):
            refuse_fetch(url)


def test_report_without_series_still_renders():
    html = render_html(**_render(sections=[_section(with_series=False)]))
    assert html.count("<svg") == 1  # history chart only


@pytest.fixture
def client(settings):
    market = MarketData.from_settings(settings)
    market.gdelt = StubGdelt()
    market.evds = StubEvds(synthetic_prices(400, ohlcv=False))
    models = [ScriptedChatModel(script=full_script("XU100"))]
    app = create_app(settings, market=market, model_factory=lambda: models.pop(0))
    with TestClient(app) as test_client:
        yield test_client


def test_pdf_download_is_owner_scoped(client):
    body = client.post("/analyses", json={"symbols": ["XU100"]}).json()
    url = f"/analyses/{body['analysis_id']}/report.pdf"

    response = client.get(url)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "attachment" in response.headers["content-disposition"]
    assert response.content.startswith(b"%PDF")

    client.app.dependency_overrides[principal] = lambda: "mallory"
    assert client.get(url).status_code == 404
    client.app.dependency_overrides.clear()
    assert client.get("/analyses/not-a-real-id/report.pdf").status_code == 404


def test_cached_responses_also_get_a_report(client):
    first = client.post("/analyses", json={"symbols": ["XU100"]}).json()
    second = client.post("/analyses", json={"symbols": ["XU100"]}).json()

    assert second["cached"] is True
    assert second["analysis_id"] != first["analysis_id"]
    assert client.get(f"/analyses/{second['analysis_id']}/report.pdf").status_code == 200
