"""HTTP API: uploads, analyses (blocking and streamed) and history.

Every endpoint that triggers work or changes state is a POST with a typed
JSON body, so browser-originated requests can be protected by CSRF checks.
The caller's identity comes from the ``principal`` dependency, which the
authentication layer overrides in deployed builds.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Annotated, Any

from cachetools import TTLCache
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from investing_engine import __version__
from investing_engine.agents.graph import AnalysisResult, build_graph, stream_analysis
from investing_engine.agents.llm import build_chat_model
from investing_engine.agents.session import engine_tools
from investing_engine.config import Settings, get_settings
from investing_engine.mcp_server.server import LOCAL_PRINCIPAL, build_server
from investing_engine.providers.base import ProviderError
from investing_engine.services import MarketData
from investing_engine.universe import (
    Instrument,
    UnknownSymbolError,
    describe_universe,
    resolve,
    resolve_many,
)

logger = logging.getLogger(__name__)

ModelFactory = Callable[[], BaseChatModel]


class AnalysisRequest(BaseModel):
    symbols: list[str] = Field(min_length=1, max_length=10, examples=[["XU100"]])
    datasets: dict[str, str] = Field(
        default_factory=dict,
        description="Optional symbol -> dataset_id map from POST /datasets.",
    )


class AnalysisResponse(BaseModel):
    report: str
    technical: dict[str, dict[str, Any]]
    risk: dict[str, float]
    sources: list[dict[str, str]]
    timings: dict[str, float]
    cached: bool = False


def principal() -> str:
    """Identity of the caller. Replaced by the authentication layer when deployed."""
    return LOCAL_PRINCIPAL


Principal = Annotated[str, Depends(principal)]


def _market(request: Request) -> MarketData:
    market: MarketData = request.app.state.market
    return market


Market = Annotated[MarketData, Depends(_market)]


def _instruments(body: AnalysisRequest, settings: Settings) -> list[Instrument]:
    try:
        return resolve_many(",".join(body.symbols), limit=settings.max_symbols_per_request)
    except UnknownSymbolError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _section_for(report: str, symbol: str) -> str:
    """The report section headed ``## <symbol>``, or the whole report as fallback."""
    marker = f"## {symbol}"
    start = report.find(marker)
    if start == -1:
        return report
    end = report.find("\n## ", start + len(marker))
    return report[start : end if end != -1 else None].strip()


def _persist(market: MarketData, instruments: Sequence[Instrument], result: AnalysisResult) -> None:
    for instrument in instruments:
        technical = result.technical.get(instrument.symbol, {})
        market.history.save(
            symbol=instrument.symbol,
            signal=technical.get("signal"),
            risk_score=result.risk.get(instrument.symbol),
            ema34=technical.get("ema34"),
            ema89=technical.get("ema89"),
            price=technical.get("price"),
            divergence=technical.get("momentum_divergence"),
            summary=_section_for(result.report, instrument.symbol),
        )


def _response(market: MarketData, result: AnalysisResult) -> AnalysisResponse:
    return AnalysisResponse(
        report=result.report,
        technical=result.technical,
        risk=result.risk,
        sources=[{"name": s.name, "attribution": s.attribution} for s in market.active_sources()],
        timings=result.timings,
    )


def create_app(
    settings: Settings | None = None,
    *,
    market: MarketData | None = None,
    model_factory: ModelFactory | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = market is None
        app.state.market = market or MarketData.from_settings(settings)
        app.state.mcp = build_server(app.state.market)
        app.state.model_factory = model_factory or (lambda: build_chat_model(settings))
        app.state.cache = TTLCache(maxsize=256, ttl=max(settings.cache_ttl_seconds, 1))
        yield
        if owned:
            app.state.market.close()

    app = FastAPI(
        title="Investing Engine API",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )

    async def _run_events(
        request: Request, caller: str, body: AnalysisRequest
    ) -> AsyncIterator[dict[str, Any]]:
        state = request.app.state
        instruments = _instruments(body, settings)
        cache_key = ",".join(sorted(i.symbol for i in instruments))
        if not body.datasets and (cached := state.cache.get(cache_key)) is not None:
            yield {"type": "final", **cached.model_dump(), "cached": True}
            return

        model = state.model_factory()
        async with engine_tools(state.mcp, principal=caller, datasets=body.datasets) as tools:
            async for event in stream_analysis(build_graph(model, tools), instruments):
                if event["type"] != "final":
                    yield event
                    continue
                result: AnalysisResult = event["result"]
                response = _response(state.market, result)
                if result.report and not result.errors:
                    await run_in_threadpool(_persist, state.market, instruments, result)
                    if not body.datasets:
                        state.cache[cache_key] = response
                yield {"type": "final", **response.model_dump()}

    @app.get("/healthz", tags=["meta"])
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/instruments", tags=["reference"])
    def instruments() -> list[dict[str, Any]]:
        return describe_universe()

    @app.get("/sources", tags=["reference"])
    def sources(market: Market) -> list[dict[str, str]]:
        return [
            {"name": s.name, "url": s.url, "terms": s.terms, "attribution": s.attribution}
            for s in market.active_sources()
        ]

    @app.post("/datasets", tags=["data"], status_code=201)
    async def upload_dataset(
        market: Market,
        caller: Principal,
        symbol: Annotated[str, Form(max_length=16)],
        file: Annotated[UploadFile, File(description="Daily OHLCV CSV")],
    ) -> dict[str, Any]:
        raw = await file.read(settings.max_upload_bytes + 1)
        try:
            return await run_in_threadpool(
                market.register_prices, owner=caller, symbol=symbol, raw=raw
            )
        except (UnknownSymbolError, ProviderError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/technical/{symbol}", tags=["analysis"])
    async def technical(
        symbol: str, market: Market, caller: Principal, dataset_id: str | None = None
    ) -> dict[str, Any]:
        """Indicators and forecast only - no LLM involved."""
        try:
            snapshot = await run_in_threadpool(
                market.technical, symbol, owner=caller, dataset_id=dataset_id
            )
        except (UnknownSymbolError, ProviderError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return snapshot.model_dump(mode="json")

    @app.post("/analyses", tags=["analysis"])
    async def analyse(
        body: AnalysisRequest, request: Request, caller: Principal
    ) -> AnalysisResponse:
        final: dict[str, Any] | None = None
        async for event in _run_events(request, caller, body):
            if event["type"] == "final":
                final = event
        if final is None or not final.get("report"):
            raise HTTPException(status_code=502, detail="The analysis could not be completed.")
        final.pop("type")
        return AnalysisResponse(**final)

    @app.post("/analyses/stream", tags=["analysis"])
    async def analyse_stream(
        body: AnalysisRequest, request: Request, caller: Principal
    ) -> StreamingResponse:
        """Server-sent events: ``status``, ``tool_call``, ``tool_result``, ``token``,
        ``error`` and a terminal ``final`` event with the structured result."""
        _instruments(body, settings)  # validate before the stream starts (422, not 200)

        async def events() -> AsyncIterator[str]:
            async for event in _run_events(request, caller, body):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.get("/history/{symbol}", tags=["analysis"])
    def history(symbol: str, market: Market, limit: int = 100) -> list[dict[str, Any]]:
        try:
            instrument = resolve(symbol)
        except UnknownSymbolError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return market.history.timeline(instrument.symbol, limit=max(1, min(limit, 500)))

    return app
