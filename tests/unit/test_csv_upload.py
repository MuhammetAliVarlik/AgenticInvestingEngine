import pytest

from investing_engine.providers.base import ProviderError
from investing_engine.providers.csv_upload import parse_ohlcv_csv
from tests.factories import synthetic_prices

LIMITS = {"max_bytes": 1_000_000, "max_rows": 5000}


def _international_csv(rows: int = 60) -> bytes:
    frame = synthetic_prices(rows).round(2)
    frame.index.name = "Date"
    return frame.to_csv().encode()


def _turkish_csv(rows: int = 60) -> bytes:
    frame = synthetic_prices(rows).round(2)
    lines = ["Tarih;Açılış;En Yüksek;En Düşük;Kapanış;Hacim"]
    for ts, row in frame.iterrows():
        values = [f"{row[c]:.2f}".replace(".", ",") for c in ("Open", "High", "Low", "Close")]
        volume = f"{int(row['Volume']):,}".replace(",", ".")
        lines.append(";".join([ts.strftime("%d.%m.%Y"), *values, volume]))
    return "\n".join(lines).encode("utf-8")


def test_parses_international_export():
    frame = parse_ohlcv_csv(_international_csv(), symbol="THYAO", **LIMITS)
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(frame) == 60


def test_parses_turkish_export_with_decimal_commas():
    expected = synthetic_prices(60).round(2)
    frame = parse_ohlcv_csv(_turkish_csv(), symbol="THYAO", **LIMITS)

    assert frame["Close"].iloc[0] == pytest.approx(expected["Close"].iloc[0], abs=0.01)
    assert frame["Volume"].iloc[0] == int(expected["Volume"].iloc[0])
    assert frame.index[0].day == expected.index[0].day  # day-first date parsing


def test_close_only_file_is_accepted():
    raw = b"date,close\n" + b"\n".join(f"2026-01-{d:02d},{100 + d}".encode() for d in range(1, 32))
    frame = parse_ohlcv_csv(raw, symbol="THYAO", **LIMITS)
    assert list(frame.columns) == ["Close"]


def test_rejects_missing_close_column():
    with pytest.raises(ProviderError, match="'Date' and 'Close'"):
        parse_ohlcv_csv(b"date,open\n2026-01-01,1\n", symbol="THYAO", **LIMITS)


def test_rejects_oversized_file():
    with pytest.raises(ProviderError, match="upload limit"):
        parse_ohlcv_csv(b"x" * 2048, symbol="THYAO", max_bytes=1024, max_rows=10)


def test_rejects_too_many_rows():
    with pytest.raises(ProviderError, match="row limit"):
        parse_ohlcv_csv(_international_csv(60), symbol="THYAO", max_bytes=1_000_000, max_rows=50)


@pytest.mark.parametrize("raw", [b"\x89PNG\r\n\x1a\n\x00\x00", b"\xff\xfe\x00b\x00a"])
def test_rejects_binary_content(raw):
    with pytest.raises(ProviderError):
        parse_ohlcv_csv(raw, symbol="THYAO", **LIMITS)


def test_rejects_too_little_history():
    with pytest.raises(ProviderError, match="Not enough price history"):
        parse_ohlcv_csv(_international_csv(10), symbol="THYAO", **LIMITS)
