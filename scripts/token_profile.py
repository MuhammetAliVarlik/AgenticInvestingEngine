#!/usr/bin/env python3
"""Token profile and capacity plan for the configured LLM.

Runs N full analyses in-process (no cache), records the provider-reported
token usage per agent, and derives how many analyses per day and per minute
the free tier allows.

Usage:
    python scripts/token_profile.py --runs 5 --symbol XU100 --rpd 1000 --tpm 12000 --rpm 30

Defaults for --rpd/--tpm/--rpm match Groq's free tier for
openai/gpt-oss-120b at the time of writing; check
https://console.groq.com/settings/limits for your organisation's values.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import tempfile
from collections import defaultdict

from investing_engine.agents.graph import AnalysisResult, build_graph, run_analysis
from investing_engine.agents.llm import build_chat_model
from investing_engine.agents.session import engine_tools
from investing_engine.config import get_settings
from investing_engine.mcp_server.server import build_server
from investing_engine.services import MarketData
from investing_engine.universe import resolve


def percentile(values: list[int], q: float) -> float:
    if len(values) == 1:
        return float(values[0])
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


async def profile(runs: int, symbol: str) -> list[AnalysisResult]:
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        settings = get_settings().model_copy(update={"db_path": f"{tmp}/profile.db"})
        market = MarketData.from_settings(settings)
        server = build_server(market)
        for run in range(runs):
            async with engine_tools(server, principal="profiler") as tools:
                result = await run_analysis(
                    build_graph(build_chat_model(settings), tools), [resolve(symbol)]
                )
            tokens = sum(u["input_tokens"] + u["output_tokens"] for u in result.usage.values())
            print(f"run {run + 1}/{runs}: {tokens} tokens, {result.timings}")
            results.append(result)
        market.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--symbol", default="XU100")
    parser.add_argument("--rpd", type=int, default=1000, help="requests per day limit")
    parser.add_argument("--rpm", type=int, default=30, help="requests per minute limit")
    parser.add_argument("--tpm", type=int, default=12000, help="tokens per minute limit")
    args = parser.parse_args()

    results = [r for r in asyncio.run(profile(args.runs, args.symbol)) if r.usage]
    if not results:
        raise SystemExit("No run reported token usage; check the LLM configuration.")

    per_agent: dict[str, list[int]] = defaultdict(list)
    totals, calls = [], []
    for result in results:
        for agent, usage in result.usage.items():
            per_agent[agent].append(usage["input_tokens"] + usage["output_tokens"])
        totals.append(sum(u["input_tokens"] + u["output_tokens"] for u in result.usage.values()))
        calls.append(sum(u["calls"] for u in result.usage.values()))

    print(f"\n{'Agent':<22}{'p50 tokens':>12}{'p95 tokens':>12}")
    for agent, values in sorted(per_agent.items()):
        print(f"{agent:<22}{percentile(values, 50):>12.0f}{percentile(values, 95):>12.0f}")
    p95_tokens, p95_calls = percentile(totals, 95), percentile(calls, 95)
    print(f"{'total per analysis':<22}{percentile(totals, 50):>12.0f}{p95_tokens:>12.0f}")
    print(f"LLM calls per analysis: p50 {percentile(calls, 50):.0f}, p95 {p95_calls:.0f}")

    per_day = args.rpd / p95_calls
    per_minute = min(args.rpm / p95_calls, args.tpm / p95_tokens)
    print("\nCapacity plan (p95, free tier):")
    print(f"  analyses per day    ~{per_day:.0f}  (RPD {args.rpd} / {p95_calls:.0f} calls)")
    print(
        f"  analyses per minute ~{per_minute:.2f}  (bounded by RPM {args.rpm} and TPM {args.tpm})"
    )


if __name__ == "__main__":
    main()
