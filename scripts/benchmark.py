#!/usr/bin/env python3
"""Benchmarks /get_insights against a running API instance.

Runs N requests and reports mean/median/p95 latency, plus per-stage timing
if the API returns it (pass --debug to request that). By default rotates
through a small pool of tickers so repeated requests don't just measure
cache-hit latency (see api/cache.py) - use --same-ticker to deliberately
measure cache-hit latency instead.

Usage:
    python scripts/benchmark.py --ticker THYAO.IS --n 5 --url http://localhost:8080
    python scripts/benchmark.py --tickers THYAO.IS,ARCLK.IS,VESTL.IS --n 3
"""
import argparse
import statistics
import sys
import time

import httpx


def run_benchmark(url: str, tickers: list[str], n: int, debug: bool, timeout: float) -> None:
    latencies = []
    stage_totals: dict[str, list[float]] = {}

    for i in range(n):
        ticker = tickers[i % len(tickers)]
        params = {"ticker_list": ticker}
        if debug:
            params["debug"] = "true"

        print(f"[{i + 1}/{n}] requesting {ticker} ...", file=sys.stderr)
        start = time.perf_counter()
        try:
            resp = httpx.get(f"{url}/get_insights", params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            print(f"  request failed: {exc}", file=sys.stderr)
            continue
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        print(f"  {elapsed:.2f}s", file=sys.stderr)

        for stage, seconds in data.get("stages", {}).items():
            stage_totals.setdefault(stage, []).append(seconds)

    if not latencies:
        print("No successful requests - nothing to report.", file=sys.stderr)
        sys.exit(1)

    print("\n--- Results ---")
    print(f"Requests: {len(latencies)}")
    print(f"Mean:     {statistics.mean(latencies):.2f}s")
    print(f"Median:   {statistics.median(latencies):.2f}s")
    if len(latencies) > 1:
        print(f"P95:      {statistics.quantiles(latencies, n=20)[18]:.2f}s")
    print(f"Min/Max:  {min(latencies):.2f}s / {max(latencies):.2f}s")

    for stage, values in stage_totals.items():
        print(f"  stage '{stage}': mean {statistics.mean(values):.2f}s over {len(values)} samples")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", default="THYAO.IS", help="single ticker to use for every request")
    parser.add_argument("--tickers", default=None, help="comma-separated tickers to rotate through instead of --ticker")
    parser.add_argument("--same-ticker", action="store_true", help="reuse the same ticker every request (measures cache-hit latency after the first request)")
    parser.add_argument("--n", type=int, default=5, help="number of requests to run")
    parser.add_argument("--url", default="http://localhost:8080", help="base URL of the running API")
    parser.add_argument("--debug", action="store_true", help="request per-stage timing breakdown")
    parser.add_argument("--timeout", type=float, default=1200.0, help="per-request timeout in seconds (a full pipeline run can be slow, especially on modest hardware)")
    args = parser.parse_args()

    if args.same_ticker:
        tickers = [args.ticker]
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = [args.ticker]

    run_benchmark(args.url, tickers, args.n, args.debug, args.timeout)


if __name__ == "__main__":
    main()
