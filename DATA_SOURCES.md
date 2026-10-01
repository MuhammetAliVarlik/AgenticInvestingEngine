# Data Sources & Licensing

Investing Engine only ships data sources whose terms allow use in a hosted,
multi-user application. Every source is declared in code with its terms,
attribution text and a `deployable` flag
([`providers/base.py`](src/investing_engine/providers/base.py)), and the
attribution of every source used is returned with each analysis and shown in
the UI.

## Sources in the deployed build

| Source | Used for | Terms | Attribution shown |
|---|---|---|---|
| [TCMB EVDS](https://evds3.tcmb.gov.tr) — Central Bank of the Republic of Türkiye | BIST 100 index closes, USD/TRY, EUR/TRY, CBRT funding rate, CPI | Data may be used and republished by third parties provided the source is cited. Requires a free API key. | *Source: Central Bank of the Republic of Türkiye (TCMB), EVDS.* |
| [The GDELT Project](https://www.gdeltproject.org/) — DOC 2.0 API | Headline metadata (title, outlet, URL, time) and aggregate tone | Unlimited and unrestricted use, including commercial use and redistribution, with a citation of and link to the GDELT Project. | *News metadata: The GDELT Project (https://www.gdeltproject.org/).* |
| User-supplied files | Daily OHLCV for individual equities | Supplied by the user from their own licensed source (e.g. a brokerage export). | *Price data supplied by the user.* |

### How each source is used

- **EVDS** — called with the API key in a request header (never in the URL),
  through a typed client with timeouts and no redirects. Series codes are
  verified against the live service with
  [`scripts/verify_evds_series.py`](scripts/verify_evds_series.py).
- **GDELT** — only headline metadata and GDELT's own tone metric are used;
  article bodies are never fetched. Queries are built from a fixed instrument
  allowlist, never from user input, and requests are throttled to GDELT's
  guidance of one request every five seconds.
- **User uploads** — validated (size, row count, encoding, required columns),
  held in memory for at most two hours, visible only to the uploader, and never
  written to disk. Models trained on uploaded data are not persisted.

## Local-only source

| Source | Status |
|---|---|
| Yahoo Finance via `yfinance` | Yahoo's terms limit this data to personal, non-commercial use and prohibit redistribution. The provider is marked `deployable=False`, is disabled unless `ENABLE_YFINANCE=true`, and is installed only with the optional `local` extra. It powers the local research script `scripts/backtest.py`. |

## Sources deliberately not used

| Source | Reason |
|---|---|
| Automated KAP (Public Disclosure Platform) retrieval | Production use of KAP's data distribution service requires a data distribution agreement with Borsa İstanbul. Disclosures will instead be analysed from documents the user uploads. |
| Scraping news or market-data websites | Publisher content is copyrighted and site terms generally prohibit automated collection. |
| Real-time or delayed BIST equity prices | Licensed by Borsa İstanbul; redistribution requires a vendor agreement. |

## What is stored

The prediction-history database stores only the engine's own derived output
per analysis: timestamp, signal, news-risk score, EMA values, price at
analysis time, a divergence flag and the report section. No headlines,
uploaded files or third-party text are persisted.

---

Nothing produced by this project is investment advice.
