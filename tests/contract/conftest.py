"""Cross-service contract tests: the REAL backend clients call the REAL Agent Core, RAG Core and
LLM service apps (in-process, over ASGI), and the LLM service calls a mock FreeLLMAPI over real
HTTP. Only model output and embeddings are fake.

Run from the repository root:  .venv/Scripts/python -m pytest tests/contract
"""
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("backend", "agent-core", "rag-core", "llm", "scripts"):
    sys.path.insert(0, str(ROOT / sub))

MOCK_KEY = "contract-test-key"
os.environ.update(
    {
        "ADVANCERAG_ENV_FILE": "none",  # never read a developer .env in tests
        "ENVIRONMENT": "test",
        "DATABASE_URL": "sqlite+aiosqlite://",
        "JWT_SECRET": "contract-access-secret-0123456789abcdefghijkl",
        "JWT_REFRESH_SECRET": "contract-refresh-secret-0123456789abcdefghij",
        "INTERNAL_SERVICE_TOKEN": "contract-internal-token-0123456789abcdefghijkl",
        "WORKER_ENABLED": "false",
        "RATE_LIMIT_ENABLED": "false",
        "EMAIL_PROVIDER": "console",
        "LOG_JSON": "false",
        "LOG_LEVEL": "WARNING",
        "EMBEDDING_PROVIDER": "fake",
        "EMBEDDING_DIMENSIONS": "64",
        "VECTOR_STORE": "local",
        "RERANKER": "none",
        "CHUNK_SIZE": "400",
        "CHUNK_OVERLAP": "50",
        "FREELLMAPI_API_KEY": MOCK_KEY,
        "LLM_MAX_RETRIES": "0",
        "INGEST_POLL_INTERVAL_SECONDS": "0.02",
    }
)

import socket  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402
import uvicorn  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def freellmapi_url():
    from mock_freellmapi import build_app

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(build_app(MOCK_KEY), host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
async def stack(freellmapi_url, tmp_path):
    """Wire backend -> agent-core -> {rag-core, llm} -> mock FreeLLMAPI."""
    from agent_core import main as agent_main
    from agent_core.clients import LLMClient, RAGClient
    from agent_core.config import settings as agent_settings
    from agent_core.orchestrator import Orchestrator
    from app.db import session as backend_db
    from app.db.base import Base as BackendBase
    from app.integrations import http as backend_http
    from llm_service import main as llm_main
    from llm_service.config import settings as llm_settings
    from llm_service.providers import build_provider, get_provider
    from rag_core import db as rag_db
    from rag_core import main as rag_main

    # Databases (backend + rag core), in-memory.
    bengine = backend_db.configure_engine("sqlite+aiosqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    async with bengine.begin() as c:
        import app.db.models  # noqa: F401
        await c.run_sync(BackendBase.metadata.create_all)
    # RAG Core indexes in a background task while other requests poll it: like PostgreSQL in
    # production, every session needs its own connection (a shared in-memory connection would let
    # one session's rollback undo another's uncommitted writes).
    from sqlalchemy.pool import NullPool

    rengine = rag_db.configure_engine(f"sqlite+aiosqlite:///{(tmp_path / 'rag.db').as_posix()}", poolclass=NullPool)
    async with rengine.begin() as c:
        await c.run_sync(rag_db.Base.metadata.create_all)

    # LLM service -> mock FreeLLMAPI over real HTTP.
    llm_settings.FREELLMAPI_BASE_URL = freellmapi_url
    provider = build_provider(llm_settings)
    llm_main.app.dependency_overrides[get_provider] = lambda: provider

    llm_transport = httpx.ASGITransport(app=llm_main.app)
    rag_transport = httpx.ASGITransport(app=rag_main.app)
    agent_main.app.dependency_overrides[agent_main.get_orchestrator] = lambda: Orchestrator(
        agent_settings, RAGClient(agent_settings, transport=rag_transport), LLMClient(agent_settings, transport=llm_transport)
    )

    # Backend's pooled clients -> agent-core / rag-core apps.
    backend_http._clients["agent_core"] = httpx.AsyncClient(transport=httpx.ASGITransport(app=agent_main.app), base_url="http://agent-core", timeout=30)
    backend_http._clients["rag_core"] = httpx.AsyncClient(transport=rag_transport, base_url="http://rag-core", timeout=30)

    yield {"agent_app": agent_main.app, "rag_app": rag_main.app, "llm_app": llm_main.app, "provider": provider}

    for name in ("agent_core", "rag_core"):
        await backend_http._clients.pop(name).aclose()
    agent_main.app.dependency_overrides.clear()
    llm_main.app.dependency_overrides.clear()
    await provider.aclose()
    await bengine.dispose()
    await rengine.dispose()


@pytest.fixture
async def client(stack, tmp_path):
    from app.integrations.storage import LocalStorage, set_storage
    from app.main import app

    set_storage(LocalStorage(str(tmp_path / "storage")))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://backend") as c:
        yield c
