"""HTTP API: uploads, analyses (blocking and streamed) and history.

Every endpoint that triggers work or changes state is a POST with a typed
JSON body, so browser-originated requests can be protected by CSRF checks.
The caller's identity comes from the ``principal`` dependency, which the
authentication layer overrides in deployed builds.
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Annotated, Any

from cachetools import TTLCache
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from investing_engine import __version__
from investing_engine.agents.graph import AnalysisResult, build_graph, stream_analysis
from investing_engine.agents.llm import build_chat_model
from investing_engine.agents.session import engine_tools
from investing_engine.config import Settings, get_settings
from investing_engine.mcp_server.server import LOCAL_PRINCIPAL, build_server
from investing_engine.observability.budget import (
    RATE_LIMIT_ERRORS,
    BudgetExceededError,
    CircuitBreaker,
    UsageStore,
    total_tokens,
)
from investing_engine.observability.logging import request_id_var
from investing_engine.observability.tracing import Tracer
from investing_engine.providers.base import ProviderError
from investing_engine.reporting.pdf import InstrumentSection, render_pdf
from investing_engine.services import MarketData
from investing_engine.universe import (
    Instrument,
    UnknownSymbolError,
    describe_universe,
    resolve,
    resolve_many,
)
from investing_engine.uploads import UploadNotFoundError

logger = logging.getLogger(__name__)

ModelFactory = Callable[[], BaseChatModel]


class AnalysisRequest(BaseModel):
    symbols: list[str] = Field(min_length=1, max_length=10, examples=[["XU100"]])
    datasets: dict[str, str] = Field(
        default_factory=dict,
        description="Optional symbol -> dataset_id map from POST /datasets.",
    )
    documents: dict[str, str] = Field(
        default_factory=dict,
        description="Optional symbol -> document_id map from POST /documents.",
    )

    @property
    def personalised(self) -> bool:
        return bool(self.datasets or self.documents)


class AnalysisResponse(BaseModel):
    report: str
    technical: dict[str, dict[str, Any]]
    risk: dict[str, float]
    sources: list[dict[str, str]]
    timings: dict[str, float]
    usage: dict[str, dict[str, int]] = Field(default_factory=dict)
    checks: dict[str, Any] = Field(default_factory=dict)
    injection_flags: list[str] = Field(default_factory=list)
    cached: bool = False
    analysis_id: str | None = Field(
        default=None, description="Use with GET /analyses/{analysis_id}/report.pdf."
    )


ANALYSIS_KIND = "analysis"


def _remember(
    market: MarketData,
    caller: str,
    body: AnalysisRequest,
    instruments: Sequence[Instrument],
    response: AnalysisResponse,
) -> str:
    """Keep the result (owner-scoped, expiring) so its PDF can be downloaded."""
    symbols = [i.symbol for i in instruments]
    record = {"response": response, "symbols": symbols, "body": body}
    upload = market.uploads.put(
        owner=caller, kind=ANALYSIS_KIND, label=",".join(symbols), payload=record
    )
    return upload.id


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


def _upper_keys(mapping: dict[str, str]) -> set[str]:
    return {k.strip().upper().removesuffix(".IS") for k in mapping}


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
        usage=result.usage,
        checks=result.checks,
        injection_flags=result.injection_flags,
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
        app.state.tracer = Tracer(settings)
        app.state.usage = UsageStore(
            settings.db_path,
            salt=settings.telemetry_salt.get_secret_value(),
            max_analyses=settings.daily_analyses_per_user,
            max_tokens=settings.daily_tokens_per_user,
        )
        app.state.breaker = CircuitBreaker(settings.rate_limit_cooldown_seconds)
        yield
        app.state.tracer.shutdown()
        if owned:
            app.state.market.close()

    app = FastAPI(
        title="Investing Engine API",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )

    @app.middleware("http")
    async def correlate(request: Request, call_next: Any) -> Any:
        """Attach a request id to logs, traces and the response; log timing."""
        incoming = request.headers.get("x-request-id", "")
        request_id = (
            incoming if incoming.isalnum() and len(incoming) <= 64 else secrets.token_hex(8)
        )
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": round((time.perf_counter() - started) * 1000),
                "request_id": request_id,
            },
        )
        return response

    @app.exception_handler(BudgetExceededError)
    async def budget_exceeded(_: Request, exc: BudgetExceededError) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={"detail": str(exc)},
            headers={"Retry-After": str(exc.retry_after)},
        )

    def _prepare(
        request: Request, caller: str, body: AnalysisRequest
    ) -> tuple[list[Instrument], str, AnalysisResponse | None]:
        """Validate, look up the cache and enforce budgets before any work starts."""
        state = request.app.state
        instruments = _instruments(body, settings)
        cache_key = ",".join(sorted(i.symbol for i in instruments))
        if not body.personalised and (cached := state.cache.get(cache_key)) is not None:
            return instruments, cache_key, cached
        state.breaker.check()
        state.usage.check(caller)
        return instruments, cache_key, None

    async def _run_events(
        request: Request, caller: str, body: AnalysisRequest
    ) -> AsyncIterator[dict[str, Any]]:
        state = request.app.state
        instruments, cache_key, cached = _prepare(request, caller, body)
        if cached is not None:
            analysis_id = _remember(state.market, caller, body, instruments, cached)
            yield {
                "type": "final",
                **cached.model_dump(),
                "cached": True,
                "analysis_id": analysis_id,
            }
            return

        trace = state.tracer.start(
            principal=caller,
            request_id=request_id_var.get(),
            symbols=[i.symbol for i in instruments],
            recursion_limit=settings.max_graph_steps,
        )
        model = state.model_factory()
        documents_for = [i.symbol for i in instruments if i.symbol in _upper_keys(body.documents)]
        async with engine_tools(
            state.mcp, principal=caller, datasets=body.datasets, documents=body.documents
        ) as tools:
            async for event in stream_analysis(
                build_graph(model, tools),
                instruments,
                documents_for=documents_for,
                config=trace.config,
            ):
                if event["type"] != "final":
                    yield event
                    continue
                result: AnalysisResult = event["result"]
                await run_in_threadpool(state.tracer.finish, trace, result)
                await run_in_threadpool(
                    state.usage.record, caller, tokens=total_tokens(result.usage)
                )
                if RATE_LIMIT_ERRORS & set(result.errors):
                    state.breaker.trip()
                response = _response(state.market, result)
                if result.report and not result.errors:
                    await run_in_threadpool(_persist, state.market, instruments, result)
                    if not body.personalised:
                        state.cache[cache_key] = response
                    response = response.model_copy(
                        update={
                            "analysis_id": _remember(
                                state.market, caller, body, instruments, response
                            )
                        }
                    )
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

    @app.post("/documents", tags=["data"], status_code=201)
    async def upload_document(
        market: Market,
        caller: Principal,
        symbol: Annotated[str, Form(max_length=16)],
        file: Annotated[UploadFile, File(description="Disclosure document: PDF, PNG or JPEG")],
    ) -> dict[str, Any]:
        """Register a disclosure document; scanned pages are read with OCR."""
        raw = await file.read(settings.max_document_bytes + 1)
        try:
            return await run_in_threadpool(
                market.register_document,
                owner=caller,
                symbol=symbol,
                raw=raw,
                filename=file.filename or "document",
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
        # Validate and enforce budgets before the stream starts, so errors are
        # proper 422/429 responses rather than a 200 stream that fails.
        _prepare(request, caller, body)

        async def events() -> AsyncIterator[str]:
            async for event in _run_events(request, caller, body):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.get(
        "/analyses/{analysis_id}/report.pdf",
        tags=["analysis"],
        response_class=Response,
        responses={200: {"content": {"application/pdf": {}}}},
    )
    async def report_pdf(analysis_id: str, market: Market, caller: Principal) -> Response:
        """Download a completed analysis as a PDF report with charts."""
        try:
            record = market.uploads.get(analysis_id, owner=caller, kind=ANALYSIS_KIND).payload
        except UploadNotFoundError:
            raise HTTPException(status_code=404, detail="Analysis not found or expired") from None
        response: AnalysisResponse = record["response"]
        body: AnalysisRequest = record["body"]
        datasets = {k.strip().upper().removesuffix(".IS"): v for k, v in body.datasets.items()}

        def build() -> bytes:
            sections = []
            for symbol in record["symbols"]:
                instrument = resolve(symbol)
                try:
                    series = market.chart_series(
                        symbol, owner=caller, dataset_id=datasets.get(symbol)
                    )
                except ProviderError:
                    series = None  # e.g. the uploaded dataset has expired
                sections.append(
                    InstrumentSection(
                        symbol=symbol,
                        name=instrument.name,
                        technical=response.technical.get(symbol),
                        risk=response.risk.get(symbol),
                        series=series,
                        history=market.history.timeline(symbol, limit=60),
                    )
                )
            return render_pdf(
                report=response.report,
                sections=sections,
                checks=response.checks,
                injection_flags=response.injection_flags,
                usage=response.usage,
                sources=response.sources,
                reference=analysis_id[:8],
            )

        pdf = await run_in_threadpool(build)
        filename = f"investing-engine-{'-'.join(record['symbols']).lower()}.pdf"
        return Response(
            content=pdf,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/usage", tags=["meta"])
    def usage(request: Request, caller: Principal) -> dict[str, int]:
        """The caller's consumption against today's budget."""
        store: UsageStore = request.app.state.usage
        today = store.get(caller)
        return {
            "analyses": today.analyses,
            "analyses_limit": store.max_analyses,
            "tokens": today.tokens,
            "tokens_limit": store.max_tokens,
        }

    @app.get("/history/{symbol}", tags=["analysis"])
    def history(symbol: str, market: Market, limit: int = 100) -> list[dict[str, Any]]:
        try:
            instrument = resolve(symbol)
        except UnknownSymbolError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return market.history.timeline(instrument.symbol, limit=max(1, min(limit, 500)))

    return app
