import pytest

from investing_engine.providers.base import ProviderError
from investing_engine.services import MarketData
from investing_engine.uploads import UploadNotFoundError, UploadStore
from tests.factories import synthetic_prices


class StubEvds:
    def __init__(self, frame):
        self.frame = frame
        self.calls = 0

    def fetch_series(self, codes, *, start, end):
        self.calls += 1
        return self.frame.rename(columns={"Close": codes[0]})[[codes[0]]]

    def close(self):
        pass


class StubGdelt:
    def __init__(self):
        self.calls = 0

    def headlines(self, instrument, *, days, limit):
        self.calls += 1
        return [{"title": f"{instrument.name} news", "url": "https://x"}]

    def average_tone(self, instrument, *, days):
        return -1.5

    def close(self):
        pass


@pytest.fixture
def market(settings):
    service = MarketData.from_settings(settings)
    service.gdelt = StubGdelt()
    return service


def _csv(rows=300) -> bytes:
    frame = synthetic_prices(rows).round(4)
    frame.index.name = "Date"
    return frame.to_csv().encode()


def test_equity_without_upload_explains_how_to_proceed(market):
    with pytest.raises(ProviderError, match="Upload your own OHLCV CSV"):
        market.technical("THYAO", owner="alice")


def test_uploaded_prices_drive_the_snapshot_and_are_not_cached(market, settings, tmp_path):
    meta = market.register_prices(owner="alice", symbol="thyao.is", raw=_csv())
    snapshot = market.technical("THYAO", owner="alice", dataset_id=meta["dataset_id"])

    assert meta["symbol"] == "THYAO"
    assert meta["rows"] == 300
    assert snapshot.source == "User-supplied file"
    assert not list((tmp_path / "models").glob("*"))  # user data never persisted


def test_dataset_is_isolated_per_owner(market):
    meta = market.register_prices(owner="alice", symbol="THYAO", raw=_csv())
    with pytest.raises(ProviderError, match="not found"):
        market.technical("THYAO", owner="mallory", dataset_id=meta["dataset_id"])


def test_dataset_cannot_be_reused_for_another_symbol(market):
    meta = market.register_prices(owner="alice", symbol="THYAO", raw=_csv())
    with pytest.raises(ProviderError, match="belongs to THYAO"):
        market.technical("TUPRS", owner="alice", dataset_id=meta["dataset_id"])


def test_index_uses_evds_and_caches_the_model(market, tmp_path):
    market.evds = StubEvds(synthetic_prices(400, ohlcv=False))
    snapshot = market.technical("XU100", owner="alice")

    assert snapshot.source == "TCMB EVDS"
    assert len(list((tmp_path / "models").glob("*.joblib"))) == 1


def test_index_without_evds_key_reports_configuration(market):
    with pytest.raises(ProviderError, match="EVDS_API_KEY"):
        market.technical("XU100", owner="alice")


def test_news_is_memoised_and_carries_attribution(market):
    first = market.news("THYAO", days=7)
    second = market.news("thyao", days=7)

    assert first is second
    assert market.gdelt.calls == 1
    assert first["average_tone"] == -1.5
    assert "GDELT" in first["source"]["attribution"]


def test_news_clamps_window(market):
    assert market.news("THYAO", days=999)["window_days"] == 30


def test_active_sources_never_include_yahoo_by_default(market):
    names = [s.name for s in market.active_sources()]
    assert not any("Yahoo" in n for n in names)
    assert all(s.deployable for s in market.active_sources())


def test_upload_store_owner_isolation_and_kind_check():
    store = UploadStore()
    upload = store.put(owner="alice", kind="prices", label="THYAO", payload=1)

    assert store.get(upload.id, owner="alice", kind="prices").payload == 1
    for owner, kind in [("bob", "prices"), ("alice", "document")]:
        with pytest.raises(UploadNotFoundError):
            store.get(upload.id, owner=owner, kind=kind)

    store.delete(upload.id, owner="bob")
    assert store.get(upload.id, owner="alice", kind="prices")
    store.delete(upload.id, owner="alice")
    with pytest.raises(UploadNotFoundError):
        store.get(upload.id, owner="alice", kind="prices")


class FlakyGdelt(StubGdelt):
    """Headlines work until ``down`` is set; the tone endpoint always fails."""

    def __init__(self):
        super().__init__()
        self.down = False

    def headlines(self, instrument, *, days, limit):
        if self.down:
            raise ProviderError("GDELT rate limit reached; try again shortly")
        return super().headlines(instrument, days=days, limit=limit)

    def average_tone(self, instrument, *, days):
        raise ProviderError("GDELT rate limit reached; try again shortly")


def test_news_survives_a_failing_tone_request(settings):
    service = MarketData.from_settings(settings)
    service.gdelt = FlakyGdelt()
    news = service.news("THYAO")
    assert news["headline_count"] == 1
    assert news["average_tone"] is None


def test_last_good_news_is_served_as_stale_when_gdelt_fails(settings):
    service = MarketData.from_settings(settings)
    service.gdelt = FlakyGdelt()
    fresh = service.news("THYAO")
    service._news_memo._cache.clear()  # the short cache expired
    service.gdelt.down = True

    stale = service.news("THYAO")
    assert stale["stale"] is True
    assert "not available" in stale["note"]
    assert stale["headlines"] == fresh["headlines"]


def test_news_error_without_earlier_result_is_raised(settings):
    service = MarketData.from_settings(settings)
    service.gdelt = FlakyGdelt()
    service.gdelt.down = True
    with pytest.raises(ProviderError):
        service.news("THYAO")
