#!/usr/bin/env python3
"""Latency benchmark for ``POST /analyses`` against a running API.

Reports mean, median and p95 end-to-end latency plus the server-side
gathering/total timings returned with every analysis. Requests rotate
through ``--symbols``; cached responses are counted separately so they do
not distort the cold-path numbers.

Usage:
    python scripts/benchmark.py --symbols XU100 --n 5 --url http://localhost:8080
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--symbols", default="XU100", help="comma-separated symbols to rotate")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    latencies: list[float] = []
    server: dict[str, list[float]] = {}
    cached = 0

    with httpx.Client(base_url=args.url, timeout=args.timeout) as client:
        for i in range(args.n):
            symbol = symbols[i % len(symbols)]
            print(f"[{i + 1}/{args.n}] {symbol}", file=sys.stderr)
            started = time.perf_counter()
            try:
                response = client.post("/analyses", json={"symbols": [symbol]})
                response.raise_for_status()
            except httpx.HTTPError as exc:
                print(f"  failed: {exc}", file=sys.stderr)
                continue
            elapsed = time.perf_counter() - started
            body = response.json()
            if body.get("cached"):
                cached += 1
                continue
            latencies.append(elapsed)
            for name, seconds in body.get("timings", {}).items():
                server.setdefault(name, []).append(seconds)

    if not latencies:
        sys.exit("No uncached successful requests to report.")

    print(f"\nUncached requests: {len(latencies)} (cache hits skipped: {cached})")
    print(f"Mean:   {statistics.mean(latencies):.2f}s")
    print(f"Median: {statistics.median(latencies):.2f}s")
    if len(latencies) >= 2:
        print(f"P95:    {statistics.quantiles(latencies, n=20)[18]:.2f}s")
    for name, values in server.items():
        print(f"server {name}: mean {statistics.mean(values):.2f}s")


if __name__ == "__main__":
    main()
