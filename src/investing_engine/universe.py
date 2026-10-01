"""The set of instruments the engine is allowed to analyse.

The universe is an explicit allowlist: every symbol that reaches a provider,
a tool or a prompt has been validated here first. This keeps arbitrary user
text out of outbound queries and bounds the work a single request can cause.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{2,6}$")


class InstrumentKind(str, Enum):
    INDEX = "index"
    EQUITY = "equity"


@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: str
    name: str
    kind: InstrumentKind
    news_query: str
    """GDELT full-text query used to find coverage of this instrument."""
    evds_series: str | None = None
    """EVDS series code for instruments whose prices are published by TCMB."""


class UnknownSymbolError(ValueError):
    """Raised when a symbol is malformed or outside the supported universe."""


_INSTRUMENTS: tuple[Instrument, ...] = (
    Instrument(
        "XU100",
        "BIST 100 Index",
        InstrumentKind.INDEX,
        '("Borsa Istanbul" OR "BIST 100" OR "BIST100")',
        evds_series="TP.MK.F.BILESIK.TUM",
    ),
    Instrument("AKBNK", "Akbank", InstrumentKind.EQUITY, '"Akbank"'),
    Instrument("ARCLK", "Arçelik", InstrumentKind.EQUITY, '("Arcelik" OR "Arçelik")'),
    Instrument("ASELS", "Aselsan", InstrumentKind.EQUITY, '"Aselsan"'),
    Instrument(
        "BIMAS", "BİM Birleşik Mağazalar", InstrumentKind.EQUITY, '"BIM Birlesik Magazalar"'
    ),
    Instrument(
        "EREGL", "Ereğli Demir Çelik", InstrumentKind.EQUITY, '("Erdemir" OR "Eregli Demir")'
    ),
    Instrument("FROTO", "Ford Otosan", InstrumentKind.EQUITY, '"Ford Otosan"'),
    Instrument("GARAN", "Garanti BBVA", InstrumentKind.EQUITY, '"Garanti BBVA"'),
    Instrument("ISCTR", "Türkiye İş Bankası", InstrumentKind.EQUITY, '("Isbank" OR "Is Bankasi")'),
    Instrument("KCHOL", "Koç Holding", InstrumentKind.EQUITY, '("Koc Holding" OR "Koç Holding")'),
    Instrument("PGSUS", "Pegasus Hava Taşımacılığı", InstrumentKind.EQUITY, '"Pegasus Airlines"'),
    Instrument("SAHOL", "Hacı Ömer Sabancı Holding", InstrumentKind.EQUITY, '"Sabanci Holding"'),
    Instrument("SISE", "Türkiye Şişe ve Cam", InstrumentKind.EQUITY, '("Sisecam" OR "Şişecam")'),
    Instrument("TCELL", "Turkcell", InstrumentKind.EQUITY, '"Turkcell"'),
    Instrument("THYAO", "Türk Hava Yolları", InstrumentKind.EQUITY, '"Turkish Airlines"'),
    Instrument("TOASO", "Tofaş Türk Otomobil", InstrumentKind.EQUITY, '("Tofas" OR "Tofaş")'),
    Instrument("TTKOM", "Türk Telekom", InstrumentKind.EQUITY, '"Turk Telekom"'),
    Instrument("TUPRS", "Tüpraş", InstrumentKind.EQUITY, '("Tupras" OR "Tüpraş")'),
    Instrument("YKBNK", "Yapı Kredi", InstrumentKind.EQUITY, '"Yapi Kredi"'),
)

UNIVERSE: dict[str, Instrument] = {i.symbol: i for i in _INSTRUMENTS}


def normalize_symbol(raw: str) -> str:
    """Canonicalise user input: upper-case, trimmed, Yahoo ``.IS`` suffix removed."""
    symbol = raw.strip().upper()
    return symbol.removesuffix(".IS")


def resolve(raw: str) -> Instrument:
    """Validate ``raw`` and return the matching instrument.

    Raises:
        UnknownSymbolError: if the symbol is malformed or not supported.
    """
    symbol = normalize_symbol(raw)
    if not _SYMBOL_PATTERN.fullmatch(symbol):
        raise UnknownSymbolError(f"Malformed symbol: {raw[:16]!r}")
    try:
        return UNIVERSE[symbol]
    except KeyError:
        raise UnknownSymbolError(f"Unsupported symbol: {symbol}") from None


def resolve_many(raw_list: str, *, limit: int) -> list[Instrument]:
    """Parse a comma-separated symbol list, de-duplicated and order-preserving.

    Raises:
        UnknownSymbolError: on any invalid symbol, an empty list, or more
            than ``limit`` distinct symbols.
    """
    seen: dict[str, Instrument] = {}
    for part in raw_list.split(","):
        if part.strip():
            instrument = resolve(part)
            seen.setdefault(instrument.symbol, instrument)
    if not seen:
        raise UnknownSymbolError("No symbols supplied")
    if len(seen) > limit:
        raise UnknownSymbolError(f"At most {limit} symbols per request (got {len(seen)})")
    return list(seen.values())
