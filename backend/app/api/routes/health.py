import asyncio
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.config import settings
from app.db.session import session_scope
from app.integrations.http import ping
from app.integrations.storage import get_storage

router = APIRouter()
logger = logging.getLogger("advancerag.health")


async def _database_ok() -> bool:
    try:
        async with session_scope() as db:
            await asyncio.wait_for(db.execute(text("SELECT 1")), timeout=3.0)
        return True
    except Exception:
        logger.warning("readiness_db_check_failed")
        return False


async def _storage_ok() -> bool:
    try:
        return await asyncio.wait_for(get_storage().healthy(), timeout=5.0)
    except Exception:
        return False


@router.get("/health")
async def health():
    """Liveness: the process is up. No dependency checks."""
    return {"status": "ok"}


@router.get("/health/live")
async def live():
    return {"status": "alive"}


@router.get("/ready")
@router.get("/health/ready")
async def ready():
    """Readiness. Database and storage are always required. Agent Core and RAG Core (which in
    turn check the LLM layer, embeddings and vector store via their own /ready) are required
    when READINESS_REQUIRE_SERVICES is on (default outside development). Booleans only."""
    db_ok, storage_ok, agent_ok, rag_ok = await asyncio.gather(
        _database_ok(),
        _storage_ok(),
        ping("agent_core", settings.AGENT_CORE_URL, "/ready"),
        ping("rag_core", settings.RAG_CORE_URL, "/ready"),
    )
    required = [db_ok, storage_ok] + ([agent_ok, rag_ok] if settings.readiness_requires_services else [])
    ok = all(required)
    body = {
        "status": "ready" if ok else "not_ready",
        "checks": {"database": db_ok, "storage": storage_ok, "agent_core": agent_ok, "rag_core": rag_ok},
        "services_required": settings.readiness_requires_services,
    }
    return JSONResponse(status_code=200 if ok else 503, content=body)
