"""LLM service: the only component that talks to model providers.

Internal API (Bearer INTERNAL_SERVICE_TOKEN):
  POST /v1/generate  -> ChatResponse
  POST /v1/stream    -> text/event-stream: token {delta} | done {model, finish_reason, usage, latency_ms} | error {code, message, retryable}
  GET  /health (liveness), GET /ready (provider reachable + authenticated), GET /metrics
"""
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from advancerag_common.observability import METRICS, RequestContextMiddleware, configure_logging
from advancerag_common.service import ServiceError, install_error_handlers, internal_auth, metrics_endpoint
from llm_service.config import settings
from llm_service.providers import close_provider, get_provider
from llm_service.providers.base import GenerationRequest, LLMError, LLMProvider
from llm_service.schemas import ChatRequest, ChatResponse, UsageOut

log = logging.getLogger("advancerag.llm")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging("llm", settings.LOG_LEVEL)
    log.info("startup", extra={"environment": settings.ENVIRONMENT, "provider": settings.LLM_PROVIDER, "default_model": settings.FREELLMAPI_MODEL})
    yield
    await close_provider()


app = FastAPI(title="AdvancedRAG LLM Service", lifespan=lifespan, docs_url=None if settings.ENVIRONMENT == "production" else "/docs")
app.add_middleware(RequestContextMiddleware)
install_error_handlers(app)
metrics_endpoint(app, settings.internal_token)
auth = Depends(internal_auth(settings.internal_token))


def _to_request(body: ChatRequest) -> GenerationRequest:
    return GenerationRequest(
        messages=[m.model_dump(exclude_none=True) for m in body.messages],
        model=body.model,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        response_format=body.response_format,
        tools=body.tools,
        tool_choice=body.tool_choice,
    )


def _record(outcome: str, started: float, kind: str, model: str = "") -> float:
    ms = round((time.perf_counter() - started) * 1000, 2)
    # Histogram count per outcome = request/success/error counts; `model` is the routed model id.
    METRICS.observe_ms("llm_call", ms, kind=kind, outcome=outcome, model=model or "unknown")
    log.info("llm_call", extra={"kind": kind, "outcome": outcome, "duration_ms": ms, "model": model})
    return ms


@app.post("/v1/generate", response_model=ChatResponse, dependencies=[auth])
async def generate(body: ChatRequest, provider: LLMProvider = Depends(get_provider)):
    started = time.perf_counter()
    try:
        result = await provider.generate(_to_request(body))
    except LLMError as exc:
        _record(exc.code, started, "generate")
        raise ServiceError(exc.code, exc.message, exc.status_code, exc.retryable)
    ms = _record("ok", started, "generate", result.model or "")
    return ChatResponse(
        content=result.content,
        model=result.model,
        finish_reason=result.finish_reason,
        usage=UsageOut(**asdict(result.usage)),
        tool_calls=result.tool_calls,
        latency_ms=ms,
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/v1/stream", dependencies=[auth])
async def stream(body: ChatRequest, provider: LLMProvider = Depends(get_provider)):
    request = _to_request(body)

    async def events():
        started = time.perf_counter()
        first_token_ms = None
        finished = False
        try:
            async for piece in provider.stream(request):
                if piece.text is not None:
                    if first_token_ms is None:
                        first_token_ms = round((time.perf_counter() - started) * 1000, 2)
                    yield _sse("token", {"delta": piece.text})
                elif piece.final is not None:
                    finished = True
                    ms = _record("ok", started, "stream", piece.final.model or "")
                    yield _sse(
                        "done",
                        {
                            "model": piece.final.model,
                            "finish_reason": piece.final.finish_reason,
                            "usage": asdict(piece.final.usage),
                            "latency_ms": ms,
                            "first_token_ms": first_token_ms,
                        },
                    )
        except LLMError as exc:
            finished = True
            _record(exc.code, started, "stream")
            yield _sse("error", {"code": exc.code, "message": exc.message, "retryable": exc.retryable})
        finally:
            if not finished:  # caller disconnected mid-stream (or shutdown): record it as cancelled
                _record("cancelled", started, "stream")

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready(provider: LLMProvider = Depends(get_provider)):
    """Lightweight readiness (safe for deploy health probes): FreeLLMAPI is configured,
    reachable, accepts our key, answers /models with OpenAI-compatible JSON and offers the
    configured model. Never runs a chat completion (use `python -m llm_service.healthcheck
    --chat` for that). No URLs or secrets in the response."""
    status = await provider.health_check(deep=False)
    body = {
        "status": "ready" if status.ok else "not_ready",
        "provider": provider.name,
        "default_model": provider.default_model,
        "checks": {
            "configured": status.configured,
            "reachable": status.reachable,
            "authenticated": status.authenticated,
            "models_response_valid": status.valid_response,
            "model_available": status.model_available,
        },
        "detail": status.detail,
    }
    return JSONResponse(status_code=200 if status.ok else 503, content=body)
