"""Agent Core service (internal; Bearer INTERNAL_SERVICE_TOKEN). Stateless orchestration layer
between the Backend, RAG Core and the LLM service — see docs/agent-core-architecture.md.

  POST /api/v1/agent/invoke -> {content, citations[], metadata{}}
  POST /api/v1/agent/stream -> text/event-stream (see orchestrator module docstring)
  GET  /health (liveness), /ready (RAG Core + LLM service ready), /metrics (token)
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from advancerag_common.observability import RequestContextMiddleware, configure_logging
from advancerag_common.service import ServiceError, install_error_handlers, internal_auth, metrics_endpoint
from advancerag_common.sse import format_event
from agent_core.clients import LLMClient, RAGClient
from agent_core.config import settings
from agent_core.errors import AgentError
from agent_core.orchestrator import Orchestrator
from agent_core.schemas import AgentRequest

log = logging.getLogger("advancerag.agent")
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator(settings, RAGClient(settings), LLMClient(settings))
    return _orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging("agent-core", settings.LOG_LEVEL)
    log.info("startup", extra={"environment": settings.ENVIRONMENT})
    wake_tasks = []
    if settings.WAKE_DEPENDENCIES_ON_START:
        from advancerag_common.wake import wake_dependencies

        wake_tasks = wake_dependencies([settings.RAG_CORE_URL, settings.LLM_SERVICE_URL])
    yield
    for task in wake_tasks:
        task.cancel()
    if _orchestrator:
        await _orchestrator.rag.aclose()
        await _orchestrator.llm.aclose()


app = FastAPI(title="AdvancedRAG Agent Core", lifespan=lifespan, docs_url=None if settings.ENVIRONMENT == "production" else "/docs")
app.add_middleware(RequestContextMiddleware)
install_error_handlers(app)
metrics_endpoint(app, settings.internal_token)
auth = Depends(internal_auth(settings.internal_token))


@app.post("/api/v1/agent/invoke", dependencies=[auth])
async def invoke(body: AgentRequest, orch: Orchestrator = Depends(get_orchestrator)):
    try:
        return await orch.invoke(body)
    except AgentError as exc:
        raise ServiceError(exc.code, exc.message, 503, exc.retryable)


@app.post("/api/v1/agent/stream", dependencies=[auth])
async def stream(body: AgentRequest, orch: Orchestrator = Depends(get_orchestrator)):
    async def events():
        async for name, data in orch.stream(body):
            yield format_event(name, data)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready(orch: Orchestrator = Depends(get_orchestrator)):
    """Lightweight: RAG Core and the LLM service report ready (no retrieval or generation)."""
    rag_ok, llm_ok = await asyncio.gather(orch.rag.ready(), orch.llm.ready())
    ok = rag_ok and llm_ok
    return JSONResponse(status_code=200 if ok else 503, content={"status": "ready" if ok else "not_ready", "checks": {"rag_core": rag_ok, "llm": llm_ok}})
