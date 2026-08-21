from tools import PDPTool as pdp


LISTING_HTML = """
<html><body><main>
<div>
  <div></div>
  <div>
    <div></div>
    <div></div>
    <div>
      <div>
        <div>
          <div></div>
          <div><a href="/borsa/hisse/balatacilar-bala/kap-haberi/1"></a></div>
          <div><a href="/borsa/hisse/rainbow-polikarb/kap-haberi/2"></a></div>
        </div>
      </div>
    </div>
  </div>
</div>
</main></body></html>
"""


def _detail_html(heading: str, table_text: str = "Some disclosure text") -> str:
    return f"""
<html><body><main>
<div>
  <div></div>
  <div>
    <div>
      <article>
        <div>
          <div><h1>{heading}</h1></div>
        </div>
        <div>
          <div>
            <table><tbody><tr><td>{table_text}</td></tr></tbody></table>
          </div>
        </div>
      </article>
    </div>
  </div>
</div>
</main></body></html>
"""


BALAT_DETAIL_HTML = _detail_html("BALAT/BALATACILAR BALATACILIK SANAYI VE TICARET A.S.")
RNPOL_DETAIL_HTML = _detail_html("RNPOL/RAINBOW POLIKARBONAT SANAYI TICARET A.S.", "Rainbow disclosure text")
EMPTY_TABLE_DETAIL_HTML = _detail_html("BALAT/BALATACILAR BALATACILIK SANAYI VE TICARET A.S.", table_text="")


class _FakeResponse:
    def __init__(self, html):
        self.content = html.encode("utf-8")


def _fake_get(details: dict):
    """details maps detail-page URL substrings to their HTML."""
    def _get(url, headers=None, timeout=None):
        if "kap-haberleri" in url:
            return _FakeResponse(LISTING_HTML)
        for key, html_body in details.items():
            if key in url:
                return _FakeResponse(html_body)
        return _FakeResponse(_detail_html("ZZZZ/UNKNOWN COMPANY A.S."))
    return _get


def test_pdp_scraper_returns_only_matching_ticker_disclosure(monkeypatch):
    monkeypatch.setattr(pdp.requests, "get", _fake_get({
        "balatacilar-bala": BALAT_DETAIL_HTML,
        "rainbow-polikarb": RNPOL_DETAIL_HTML,
    }))

    result = pdp.pdp_news_scraper.invoke({"ticker": "BALAT.IS", "pages_to_scan": 1})

    assert len(result) == 1
    assert "Some disclosure text" in result[0]["content"]
    assert "balatacilar-bala" in result[0]["url"]


def test_pdp_scraper_ticker_match_is_case_and_suffix_insensitive(monkeypatch):
    monkeypatch.setattr(pdp.requests, "get", _fake_get({
        "rainbow-polikarb": RNPOL_DETAIL_HTML,
        "balatacilar-bala": BALAT_DETAIL_HTML,
    }))

    result = pdp.pdp_news_scraper.invoke({"ticker": "rnpol.is", "pages_to_scan": 1})

    assert len(result) == 1
    assert "Rainbow disclosure text" in result[0]["content"]


def test_pdp_scraper_returns_explicit_no_match_info_when_ticker_not_found(monkeypatch):
    monkeypatch.setattr(pdp.requests, "get", _fake_get({
        "balatacilar-bala": BALAT_DETAIL_HTML,
        "rainbow-polikarb": RNPOL_DETAIL_HTML,
    }))

    result = pdp.pdp_news_scraper.invoke({"ticker": "THYAO.IS", "pages_to_scan": 1})

    assert len(result) == 1
    assert "info" in result[0]
    assert "THYAO.IS" in result[0]["info"]
    assert "error" not in result[0]


def test_pdp_scraper_never_returns_unrelated_disclosure_as_a_match(monkeypatch):
    """Regression test: an unrelated company's disclosure must never be
    silently returned for a different requested ticker."""
    monkeypatch.setattr(pdp.requests, "get", _fake_get({
        "balatacilar-bala": BALAT_DETAIL_HTML,
    }))

    result = pdp.pdp_news_scraper.invoke({"ticker": "THYAO.IS", "pages_to_scan": 1})

    urls = [r.get("url") for r in result]
    assert "balatacilar-bala" not in " ".join(str(u) for u in urls)


def test_pdp_scraper_empty_content_uses_placeholder(monkeypatch):
    monkeypatch.setattr(pdp.requests, "get", _fake_get({
        "balatacilar-bala": EMPTY_TABLE_DETAIL_HTML,
    }))

    result = pdp.pdp_news_scraper.invoke({"ticker": "BALAT.IS", "pages_to_scan": 1})

    assert all("bulunamadı" in entry["content"] for entry in result)


def test_pdp_scraper_scans_multiple_pages(monkeypatch):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        if "kap-haberleri" in url:
            return _FakeResponse(LISTING_HTML)
        return _FakeResponse(BALAT_DETAIL_HTML)

    monkeypatch.setattr(pdp.requests, "get", fake_get)

    pdp.pdp_news_scraper.invoke({"ticker": "BALAT.IS", "pages_to_scan": 3})

    listing_calls = [u for u in calls if "kap-haberleri" in u]
    assert len(listing_calls) == 3


def test_pdp_scraper_returns_error_dict_on_request_failure(monkeypatch):
    def raise_error(url, headers=None, timeout=None):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(pdp.requests, "get", raise_error)

    result = pdp.pdp_news_scraper.invoke({"ticker": "THYAO.IS"})

    assert result == [{"error": "simulated network failure"}]


def test_normalize_ticker_strips_suffix_and_uppercases():
    assert pdp._normalize_ticker("thyao.is") == "THYAO"
    assert pdp._normalize_ticker("THYAO.IS") == "THYAO"
    assert pdp._normalize_ticker("thyao") == "THYAO"
