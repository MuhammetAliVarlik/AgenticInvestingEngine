import os

from cachetools import TTLCache

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "600"))

_cache: TTLCache = TTLCache(maxsize=256, ttl=CACHE_TTL_SECONDS)


def normalize_ticker_list(ticker_list: str) -> str:
    tickers = sorted({t.strip().upper() for t in ticker_list.split(",") if t.strip()})
    return ",".join(tickers)


def get_cached(ticker_list: str):
    return _cache.get(normalize_ticker_list(ticker_list))


def set_cached(ticker_list: str, value) -> None:
    _cache[normalize_ticker_list(ticker_list)] = value
