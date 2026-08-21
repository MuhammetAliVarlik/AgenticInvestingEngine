from tools import yahooFinanceNewsTool as ynt


class _FakeTicker:
    def __init__(self, news):
        self._news = news

    def get_news(self):
        return self._news


def test_stock_news_extracts_summary(monkeypatch):
    fake_news = [
        {"content": {"summary": "First article summary"}},
        {"content": {"summary": "Second article summary"}},
    ]
    monkeypatch.setattr(ynt.yf, "Ticker", lambda ticker: _FakeTicker(fake_news))

    result = ynt.stock_news.invoke({"ticker": "THYAO.IS"})

    assert result == [
        {"Summary": "First article summary"},
        {"Summary": "Second article summary"},
    ]


def test_stock_news_limits_to_five_articles(monkeypatch):
    fake_news = [{"content": {"summary": f"Article {i}"}} for i in range(10)]
    monkeypatch.setattr(ynt.yf, "Ticker", lambda ticker: _FakeTicker(fake_news))

    result = ynt.stock_news.invoke({"ticker": "THYAO.IS"})

    assert len(result) == 5


def test_stock_news_handles_missing_content_gracefully(monkeypatch):
    fake_news = [{"id": "1"}]  # no "content" key at all
    monkeypatch.setattr(ynt.yf, "Ticker", lambda ticker: _FakeTicker(fake_news))

    result = ynt.stock_news.invoke({"ticker": "THYAO.IS"})

    assert result == [{"Summary": "Summary not available"}]


def test_stock_news_handles_missing_summary_field(monkeypatch):
    fake_news = [{"content": {"title": "no summary here"}}]
    monkeypatch.setattr(ynt.yf, "Ticker", lambda ticker: _FakeTicker(fake_news))

    result = ynt.stock_news.invoke({"ticker": "THYAO.IS"})

    assert result == [{"Summary": "Summary not available"}]


def test_stock_news_returns_empty_list_when_news_not_a_list(monkeypatch):
    monkeypatch.setattr(ynt.yf, "Ticker", lambda ticker: _FakeTicker(None))

    result = ynt.stock_news.invoke({"ticker": "THYAO.IS"})

    assert result == []
