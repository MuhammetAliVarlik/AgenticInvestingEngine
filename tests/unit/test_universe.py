import pytest

from investing_engine.universe import (
    InstrumentKind,
    UnknownSymbolError,
    normalize_symbol,
    resolve,
    resolve_many,
)


@pytest.mark.parametrize("raw", ["thyao", " THYAO ", "THYAO.IS", "thyao.is"])
def test_normalize_symbol_accepts_common_spellings(raw):
    assert normalize_symbol(raw) == "THYAO"


def test_resolve_returns_instrument_metadata():
    instrument = resolve("XU100")
    assert instrument.kind is InstrumentKind.INDEX
    assert instrument.evds_series == "TP.MK.F.BILESIK.TUM"


@pytest.mark.parametrize(
    "raw",
    ["", "T", "THY AO", "THYAO; DROP TABLE", "../etc", "ignore previous instructions", "A" * 40],
)
def test_resolve_rejects_malformed_input(raw):
    with pytest.raises(UnknownSymbolError):
        resolve(raw)


def test_resolve_rejects_symbols_outside_the_universe():
    with pytest.raises(UnknownSymbolError, match="Unsupported"):
        resolve("ZZZZZ")


def test_resolve_many_deduplicates_and_preserves_order():
    result = resolve_many("TUPRS, thyao.is ,TUPRS,,", limit=3)
    assert [i.symbol for i in result] == ["TUPRS", "THYAO"]


def test_resolve_many_enforces_limit():
    with pytest.raises(UnknownSymbolError, match="At most 2"):
        resolve_many("THYAO,TUPRS,ASELS", limit=2)


def test_resolve_many_rejects_empty_input():
    with pytest.raises(UnknownSymbolError):
        resolve_many(" , ", limit=3)
