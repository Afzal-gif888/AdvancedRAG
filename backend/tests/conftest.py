"""Backend tests. In-memory SQLite by default; set TEST_DATABASE_URL to a PostgreSQL database
(e.g. postgresql+asyncpg://advancerag:...@localhost:5433/advancerag_test) to run the same suite on
PostgreSQL, with the schema built by the real Alembic migrations."""
import os

TEST_DB = os.environ.get("TEST_DATABASE_URL", "sqlite+aiosqlite://")
IS_PG = TEST_DB.startswith("postgresql")

# Must be set before `app` is imported (settings are read at import time).
os.environ.update(
    {
        "ADVANCERAG_ENV_FILE": "none",  # never read a developer .env in tests
        "INGEST_POLL_INTERVAL_SECONDS": "0.01",
        "INGEST_STALE_CONFIRM_SECONDS": "0.05",
        "ENVIRONMENT": "test",
        "DATABASE_URL": TEST_DB,
        "JWT_SECRET": "test-access-secret-0123456789abcdef0123456789",
        "JWT_REFRESH_SECRET": "test-refresh-secret-0123456789abcdef012345",
        "WORKER_ENABLED": "false",
        "RATE_LIMIT_ENABLED": "false",
        "EMAIL_PROVIDER": "console",
        "LOG_JSON": "false",
        "FRONTEND_ORIGIN": "http://localhost:5173",
        # Never reach a locally running dev stack from tests (port 9 = discard, nothing listens).
        "AGENT_CORE_URL": "http://127.0.0.1:9",
        "RAG_CORE_URL": "http://127.0.0.1:9",
    }
)

import uuid  # noqa: E402
from typing import Dict, List, Optional  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.pool import NullPool, StaticPool  # noqa: E402

from app.api.routes import auth as auth_routes  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.db import session as db_session  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.integrations.agent_core import AgentResult, AgentServiceUnavailable, StreamEvent, get_agent_client  # noqa: E402
from app.integrations.rag_core import RAGCoreError  # noqa: E402
from app.integrations.storage import LocalStorage, set_storage  # noqa: E402
from app.main import app  # noqa: E402
from app.middleware import rate_limit  # noqa: E402

PASSWORD = "CorrectHorse9"


class FakeAgent:
    """Stands in for Agent Core. `mode`: ok | unavailable | error_event | empty."""

    def __init__(self):
        self.mode = "ok"
        self.answer = ["Hello", " from", " the agent."]
        self.replace: Optional[str] = None  # final verified answer replacing the streamed draft
        self.requests: List = []

    async def invoke(self, req):
        self.requests.append(req)
        if self.mode == "unavailable":
            raise AgentServiceUnavailable()
        content = "" if self.mode == "empty" else "".join(self.answer)
        return AgentResult(content=content, citations=[{"filename": "a.pdf", "page": 1, "secret": "x"}], metadata={"model": "m", "chain_of_thought": "hidden"})

    async def stream(self, req):
        self.requests.append(req)
        if self.mode == "unavailable":
            raise AgentServiceUnavailable()
        yield StreamEvent("agent.started", {})
        yield StreamEvent("retrieval.started", {"query": "internal rewrite"})
        yield StreamEvent("retrieval.completed", {"retrieved_chunks": 3, "internal": "x"})
        if self.mode == "error_event":
            yield StreamEvent("error", {"message": "boom at http://internal:8003"})
            return
        if self.mode != "empty":
            for part in self.answer:
                yield StreamEvent("content.delta", {"delta": part})
        if self.replace is not None:
            yield StreamEvent("content.replace", {"content": self.replace})
        yield StreamEvent(
            "generation.completed",
            {"citations": [{"filename": "a.pdf", "page": 2}], "metadata": {"model": "gemini-x", "reasoning": "secret"}},
        )


class FakeRAG:
    """Stands in for RAG Core's asynchronous ingestion API. By default an attempt finishes as soon as
    it starts; tests can script `next_statuses` (consumed by successive status polls)."""

    def __init__(self):
        self.ingested: List[Dict] = []
        self.deleted: List[uuid.UUID] = []
        self.fail: Optional[RAGCoreError] = None
        self.attempts: Dict[uuid.UUID, Dict] = {}
        self.next_statuses: List[Optional[Dict]] = []
        self.status_calls = 0
        self.start_status: Optional[Dict] = None  # what start returns (default: finished at once)

    async def start_ingestion(self, **kw):
        if self.fail:
            raise self.fail
        self.ingested.append(kw)
        status = dict(self.start_status) if self.start_status else {"status": "succeeded", "stage": "done", "chunks": 3, "attempt_id": str(uuid.uuid4())}
        self.attempts[kw["document_id"]] = status
        return status

    async def ingestion_status(self, *, document_id, user_id):
        self.status_calls += 1
        if self.fail:
            raise self.fail
        if self.next_statuses:
            status = self.next_statuses.pop(0)
            if status is not None:
                self.attempts[document_id] = status
            return status
        return self.attempts.get(document_id)

    async def delete_document(self, *, document_id, user_id):
        if self.fail:
            raise self.fail
        self.deleted.append(document_id)
        self.attempts.pop(document_id, None)


from app.integrations.email import MockEmailService  # noqa: E402

# Captures every email the app sends (password-reset links); never contacts a provider.
OUTBOX = MockEmailService()


@pytest.fixture(autouse=True)
async def database(pg_schema):
    if IS_PG:
        engine = db_session.configure_engine(TEST_DB, poolclass=NullPool)
        tables = ", ".join(t.name for t in reversed(Base.metadata.sorted_tables))
        async with engine.begin() as conn:
            await conn.execute(text(f"TRUNCATE {tables} CASCADE"))
    else:
        engine = db_session.configure_engine(
            "sqlite+aiosqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture(scope="session")
def pg_schema():
    """On PostgreSQL: build the schema once per run with the real migrations (down then up)."""
    if IS_PG:
        from alembic import command
        from alembic.config import Config

        cfg = Config("alembic.ini")
        cfg.attributes["database_url"] = TEST_DB
        cfg.attributes["skip_logging_config"] = True
        command.downgrade(cfg, "base")
        command.upgrade(cfg, "head")
    return IS_PG


@pytest.fixture(autouse=True)
def storage(tmp_path):
    store = LocalStorage(str(tmp_path / "storage"))
    set_storage(store)
    return store


@pytest.fixture(autouse=True)
async def reset_rate_limits():
    rate_limit._backend = None
    yield


@pytest.fixture(autouse=True)
async def close_outbound_clients():
    """Pooled httpx clients are bound to the event loop that created them; never share them
    between tests (each test has its own loop)."""
    yield
    from app.integrations.http import close_clients

    await close_clients()


@pytest.fixture
def agent():
    fake = FakeAgent()
    app.dependency_overrides[get_agent_client] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_agent_client, None)


@pytest.fixture
def rag():
    return FakeRAG()


@pytest.fixture(autouse=True)
def outbox(monkeypatch):
    """Every test captures outgoing email instead of sending it."""
    OUTBOX.clear()
    monkeypatch.setattr(auth_routes, "get_email_service", lambda: OUTBOX)
    return OUTBOX


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


@pytest.fixture
def make_client():
    """Independent clients (separate cookie jars) for multi-user / multi-device tests."""
    clients = []

    def _make():
        c = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")
        clients.append(c)
        return c

    yield _make


async def login(client: httpx.AsyncClient, email: str, password: str = PASSWORD) -> Dict[str, str]:
    """Email + password login; returns the bearer header."""
    r = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def register_and_login(client: httpx.AsyncClient, email: Optional[str] = None, password: str = PASSWORD) -> Dict[str, str]:
    """Sign-up (name, email, password) signs the user straight in."""
    email = email or f"user-{uuid.uuid4().hex[:8]}@example.com"
    r = await client.post("/api/v1/auth/register", json={"email": email, "password": password, "full_name": "Test User"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
async def auth(client):
    return await register_and_login(client)


async def create_conversation(client, headers, title=None) -> dict:
    r = await client.post("/api/v1/conversations", json={"title": title} if title else None, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


async def upload(client, headers, name="notes.txt", content=b"hello world", mime="text/plain"):
    return await client.post("/api/v1/documents", files={"file": (name, content, mime)}, headers=headers)


def parse_sse(text: str) -> List[Dict]:
    import json

    events = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        name = next(l[6:].strip() for l in lines if l.startswith("event:"))
        data = json.loads("".join(l[5:].strip() for l in lines if l.startswith("data:")))
        events.append({"event": name, "data": data})
    return events


__all__ = ["settings"]
