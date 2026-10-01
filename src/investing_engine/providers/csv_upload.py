"""Parse user-supplied OHLCV CSV files.

Equity prices on Borsa Istanbul are licensed data, so the deployed engine
never fetches them itself: users analyse their own exports (for example from
their brokerage) instead. Both international (``,`` separated, ``.``
decimals) and Turkish (``;`` separated, ``,`` decimals, Turkish headers)
exports are accepted.
"""

from __future__ import annotations

import csv
import io
import unicodedata

import pandas as pd

from investing_engine.providers.base import ProviderError, SourceInfo, validate_price_frame

USER_UPLOAD_SOURCE = SourceInfo(
    name="User-supplied file",
    url="",
    terms="Provided by the user from their own licensed source; processed in memory "
    "for this session and never redistributed.",
    attribution="Price data supplied by the user.",
    deployable=True,
)

_HEADER_ALIASES: dict[str, str] = {
    "date": "Date",
    "tarih": "Date",
    "open": "Open",
    "acilis": "Open",
    "high": "High",
    "yuksek": "High",
    "en yuksek": "High",
    "low": "Low",
    "dusuk": "Low",
    "en dusuk": "Low",
    "close": "Close",
    "kapanis": "Close",
    "adj close": "Adj Close",
    "volume": "Volume",
    "hacim": "Volume",
}


def _fold(text: str) -> str:
    """Lower-case and strip diacritics so Turkish and ASCII header spellings compare equal."""
    text = text.strip().lower().replace("ı", "i")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def parse_ohlcv_csv(raw: bytes, *, symbol: str, max_bytes: int, max_rows: int) -> pd.DataFrame:
    """Validate and parse an uploaded CSV into a standard price frame.

    Raises:
        ProviderError: if the file is too large, not valid UTF-8 text, has no
            recognisable date/close columns, or contains too few rows.
    """
    if len(raw) > max_bytes:
        raise ProviderError(f"File exceeds the {max_bytes // 1024} KiB upload limit")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ProviderError("File must be UTF-8 encoded CSV text") from exc
    if "\x00" in text:
        raise ProviderError("File appears to be binary, not CSV")

    sample = text[:4096]
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        delimiter = ","
    decimal = "," if delimiter == ";" else "."

    try:
        # Read everything as text; numbers are converted below with the
        # locale-specific decimal separator.
        frame = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            nrows=max_rows + 1,
            dtype=str,
            engine="python",
        )
    except (pd.errors.ParserError, ValueError) as exc:
        raise ProviderError("Could not parse the CSV file") from exc

    if len(frame) > max_rows:
        raise ProviderError(f"File exceeds the {max_rows}-row limit")

    renamed = {c: _HEADER_ALIASES.get(_fold(str(c))) for c in frame.columns}
    frame = frame.rename(columns={k: v for k, v in renamed.items() if v})
    if "Close" not in frame.columns and "Adj Close" in frame.columns:
        frame = frame.rename(columns={"Adj Close": "Close"})
    if "Date" not in frame.columns or "Close" not in frame.columns:
        raise ProviderError("CSV must contain at least 'Date' and 'Close' columns")

    for column in ("Open", "High", "Low", "Close", "Volume"):
        if column in frame.columns:
            frame[column] = _to_number(frame[column], decimal=decimal)

    frame["Date"] = pd.to_datetime(frame["Date"], dayfirst=decimal == ",", errors="coerce")
    frame = frame.dropna(subset=["Date"]).set_index("Date")
    return validate_price_frame(frame, symbol=symbol)


def _to_number(column: pd.Series, *, decimal: str) -> pd.Series:
    values = column.astype(str).str.strip()
    if decimal == ",":
        values = values.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        values = values.str.replace(",", "", regex=False)
    return pd.to_numeric(values, errors="coerce")
