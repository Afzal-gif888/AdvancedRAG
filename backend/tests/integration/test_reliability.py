"""Message lifecycle, crash recovery, bounded job retries, reconciliation and log hygiene."""
import asyncio
import logging
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select

from app.core.config import settings
from app.core.exceptions import AppError
from app.core.time import utcnow
from app.db import session as db_session
from app.db.models import BackgroundJob, Document, Message
from app.integrations.agent_core import AgentServiceUnavailable, StreamEvent
from app.integrations.rag_core import RAGCoreError
from app.workers import cleanup_jobs
from app.workers.runner import process_one
from conftest import PASSWORD, create_conversation, parse_sse, register_and_login, upload

API = "/api/v1"


async def _messages(client, auth, cid):
    return (await client.get(f"{API}/conversations/{cid}/messages", headers=auth)).json()["items"]


# ---------------------------------------------------------------- assistant message lifecycle

class ScriptedAgent:
    def __init__(self, script):
        self.script = script

    async def stream(self, req):
        for item in self.script:
            if isinstance(item, Exception):
                raise item
            if item == "hang":
                await asyncio.sleep(3600)
            yield item


@pytest.fixture
def scripted(monkeypatch):
    from app.integrations.agent_core import get_agent_client
    from app.main import app

    def _use(script):
        agent = ScriptedAgent(script)
        app.dependency_overrides[get_agent_client] = lambda: agent
        return agent

    yield _use
    app.dependency_overrides.pop(get_agent_client, None)


async def test_partial_output_is_kept_when_generation_fails(client, auth, scripted):
    scripted([StreamEvent("content.delta", {"delta": "Half an"}), StreamEvent("content.delta", {"delta": " answer"}),
              StreamEvent("error", {"code": "LLM_TIMEOUT", "message": "x"})])
    conv = await create_conversation(client, auth)
    events = parse_sse((await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "q"}, headers=auth)).text)
    assert events[-1]["data"]["code"] == "LLM_TIMEOUT"
    assert events[-1]["data"]["message"] == "The language model took too long to respond."
    msg = (await _messages(client, auth, conv["id"]))[1]
    assert msg["status"] == "failed" and msg["content"] == "Half an answer" and msg["error_code"] == "LLM_TIMEOUT"


@pytest.mark.parametrize("code,expected", [("RAG_SERVICE_UNAVAILABLE", "RAG_SERVICE_UNAVAILABLE"), ("LLM_RATE_LIMITED", "LLM_RATE_LIMITED"),
                                            ("LLM_AUTH_FAILED", "GENERATION_FAILED"), ("weird<script>", "GENERATION_FAILED")])
async def test_agent_error_codes_are_whitelisted(client, auth, scripted, code, expected):
    scripted([StreamEvent("error", {"code": code, "message": "internal detail http://10.0.0.1"})])
    conv = await create_conversation(client, auth)
    r = await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "q"}, headers=auth)
    assert parse_sse(r.text)[-1]["data"]["code"] == expected and "10.0.0.1" not in r.text


async def test_agent_core_unavailable(client, auth, scripted):
    scripted([AgentServiceUnavailable()])
    conv = await create_conversation(client, auth)
    events = parse_sse((await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "q"}, headers=auth)).text)
    assert events[-1]["data"]["code"] == "AGENT_SERVICE_UNAVAILABLE"
    msgs = await _messages(client, auth, conv["id"])
    assert msgs[0]["content"] == "q" and msgs[1]["status"] == "failed"


async def test_streaming_state_is_flushed_while_generating(client, auth, scripted, monkeypatch):
    monkeypatch.setattr(settings, "STREAM_FLUSH_SECONDS", 0)
    seen = {}

    class Probe:
        async def stream(self, req):
            yield StreamEvent("content.delta", {"delta": "first "})
            async with db_session.session_scope() as db:
                seen["row"] = await db.get(Message, req.message_id)
            yield StreamEvent("content.delta", {"delta": "second"})

    from app.integrations.agent_core import get_agent_client
    from app.main import app

    app.dependency_overrides[get_agent_client] = lambda: Probe()
    conv = await create_conversation(client, auth)
    await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "q"}, headers=auth)
    assert seen["row"].status == "streaming" and seen["row"].content == "first "
    assert (await _messages(client, auth, conv["id"]))[1]["status"] == "completed"


async def test_client_disconnect_marks_cancelled_with_partial_content(auth, scripted, client):
    """Drive the SSE generator directly and close it mid-stream, as Starlette does on disconnect."""
    from app.services import agent_service, message_service
    from app.db.models import User
    from app.schemas.message import MessageCreate

    scripted([])  # unused
    conv = await create_conversation(client, auth)
    async with db_session.session_scope() as db:
        user = await db.scalar(select(User))
        turn = await message_service.prepare_turn(db, user, uuid.UUID(conv["id"]), MessageCreate(content="q"))
    agent = ScriptedAgent([StreamEvent("content.delta", {"delta": "partial"}), "hang"])
    gen = agent_service.stream_turn(agent, turn)
    assert "message.created" in await gen.__anext__()
    assert "content.delta" in await gen.__anext__()
    task = asyncio.ensure_future(gen.__anext__())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # The shielded persistence finishes in the background; wait for it deterministically.
    for _ in range(100):
        msg = (await _messages(client, auth, conv["id"]))[1]
        if msg["status"] == "cancelled":
            break
        await asyncio.sleep(0.05)
    assert msg["status"] == "cancelled" and msg["content"] == "partial" and msg["error_code"] == "CANCELLED"


async def test_crashed_generation_is_recovered_and_retryable(client, auth, agent):
    conv = await create_conversation(client, auth)
    async with db_session.session_scope() as db:
        from app.db.models import User

        user = await db.scalar(select(User))
        now = utcnow() - timedelta(hours=1)
        db.add_all([
            Message(conversation_id=uuid.UUID(conv["id"]), user_id=user.id, role="user", content="q", status="completed", created_at=now, updated_at=now),
            Message(conversation_id=uuid.UUID(conv["id"]), user_id=user.id, role="assistant", content="partial befo", status="streaming",
                    created_at=now + timedelta(milliseconds=1), updated_at=now),
        ])
        await db.commit()
        assert await cleanup_jobs.recover_generations(db) == 1
    msg = (await _messages(client, auth, conv["id"]))[1]
    assert msg["status"] == "failed" and msg["error_code"] == "INTERRUPTED" and msg["content"] == "partial befo"
    r = await client.post(f"{API}/conversations/{conv['id']}/messages/{msg['id']}/retry", headers=auth)
    assert parse_sse(r.text)[-1]["event"] == "message.completed"


async def test_recent_pending_generation_is_not_touched(client, auth):
    async with db_session.session_scope() as db:
        assert await cleanup_jobs.recover_generations(db) == 0


async def test_messages_can_be_paged_newest_first(client, auth, agent):
    conv = await create_conversation(client, auth)
    for i in range(3):
        await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": f"m{i}"}, headers=auth)
    page = (await client.get(f"{API}/conversations/{conv['id']}/messages?order=desc&limit=2", headers=auth)).json()
    assert page["total"] == 6 and [m["role"] for m in page["items"]] == ["assistant", "user"]
    assert page["items"][1]["content"] == "m2"


# ---------------------------------------------------------------- background jobs

async def _job(db, **filters):
    return await db.scalar(select(BackgroundJob).filter_by(**filters))


async def test_stale_running_job_is_retried_then_failed_after_max_attempts(client, auth, rag, storage):
    doc = (await upload(client, auth)).json()
    async with db_session.session_scope() as db:
        job = await _job(db, job_type="document_ingest")
        job.status, job.locked_at, job.attempts = "running", utcnow() - timedelta(hours=1), 1
        await db.commit()
        await cleanup_jobs.requeue_stale_jobs(db)
        await db.refresh(job)
        assert job.status == "pending"
        job.status, job.locked_at, job.attempts = "running", utcnow() - timedelta(hours=1), job.max_attempts
        await db.commit()
        await cleanup_jobs.requeue_stale_jobs(db)
        await db.refresh(job)
        assert job.status == "failed" and job.error == "WORKER_CRASHED"
    body = (await client.get(f"{API}/documents/{doc['id']}", headers=auth)).json()
    assert body["status"] == "failed"


async def test_document_stuck_processing_without_a_job_is_reconciled(client, auth, monkeypatch):
    monkeypatch.setattr(settings, "RECONCILE_INTERVAL_SECONDS", 0)
    doc = (await upload(client, auth)).json()
    async with db_session.session_scope() as db:
        await db.execute(BackgroundJob.__table__.delete())
        d = await db.get(Document, uuid.UUID(doc["id"]))
        d.status, d.updated_at = "processing", utcnow() - timedelta(hours=1)
        await db.commit()
        stats = await cleanup_jobs.reconcile_documents(db)
    assert stats["requeued_ingest"] == 1
    async with db_session.session_scope() as db:
        # Second time (a job existed and is gone again): fail rather than loop forever.
        await db.execute(BackgroundJob.__table__.update().values(status="failed"))
        d = await db.get(Document, uuid.UUID(doc["id"]))
        d.updated_at = utcnow() - timedelta(hours=1)
        await db.commit()
        stats = await cleanup_jobs.reconcile_documents(db)
    assert stats["failed"] == 1
    assert (await client.get(f"{API}/documents/{doc['id']}", headers=auth)).json()["status"] == "failed"


async def test_failed_delete_is_redriven_with_a_bounded_budget(client, auth, rag, storage, monkeypatch):
    from app.services import job_service

    monkeypatch.setattr(settings, "RECONCILE_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(job_service, "retry_delay", lambda attempt: timedelta(0))
    doc = (await upload(client, auth)).json()
    await process_one(rag=rag, storage=storage)
    rag.fail = RAGCoreError("RAG_SERVICE_UNAVAILABLE", retryable=True)
    await client.delete(f"{API}/documents/{doc['id']}", headers=auth)
    for _ in range(settings.DELETE_JOB_MAX_ATTEMPTS + 2):
        await process_one(rag=rag, storage=storage)
    async with db_session.session_scope() as db:
        job = await _job(db, job_type="document_delete")
        assert job.status == "failed" and job.attempts == settings.DELETE_JOB_MAX_ATTEMPTS
        assert (await db.get(Document, uuid.UUID(doc["id"]))).status == "deleted"  # still hidden, not orphaned
        stats = await cleanup_jobs.reconcile_documents(db)
    assert stats["redriven_delete"] == 1
    rag.fail = None
    await process_one(rag=rag, storage=storage)
    async with db_session.session_scope() as db:
        assert await db.get(Document, uuid.UUID(doc["id"])) is None
    assert [str(d) for d in rag.deleted] == [doc["id"]]


async def test_delete_redrive_gives_up_after_budget(client, auth, monkeypatch):
    monkeypatch.setattr(settings, "RECONCILE_INTERVAL_SECONDS", 0)
    doc = (await upload(client, auth)).json()
    await client.delete(f"{API}/documents/{doc['id']}", headers=auth)
    async with db_session.session_scope() as db:
        from app.db.models import User

        user = await db.scalar(select(User))
        await db.execute(BackgroundJob.__table__.update().values(status="failed"))
        for _ in range(cleanup_jobs.MAX_DELETE_REDRIVES):
            db.add(BackgroundJob(user_id=user.id, job_type="document_delete", resource_id=uuid.UUID(doc["id"]), status="failed"))
        await db.commit()
        stats = await cleanup_jobs.reconcile_documents(db)
    assert stats["abandoned_delete"] == 1 and stats["redriven_delete"] == 0


async def test_vectors_created_after_a_concurrent_delete_are_removed(client, auth, rag, storage):
    doc = (await upload(client, auth)).json()

    original = rag.start_ingestion

    async def ingest_then_user_deletes(**kw):
        status = await original(**kw)
        r = await client.delete(f"{API}/documents/{doc['id']}", headers=auth)  # user deletes mid-ingest
        assert r.status_code == 204
        return status

    rag.start_ingestion = ingest_then_user_deletes
    await process_one(rag=rag, storage=storage)
    assert [str(d) for d in rag.deleted] == [doc["id"]]


# ---------------------------------------------------------------- rate limiting & logs

async def test_login_is_limited_per_account_across_ips(client, monkeypatch):
    from app.middleware import rate_limit

    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    await client.post(f"{API}/auth/register", json={"email": "target@example.com", "password": PASSWORD, "full_name": "Test User"})
    ips = iter(f"10.0.0.{i}" for i in range(100))
    monkeypatch.setattr(rate_limit, "client_ip", lambda request: next(ips))
    codes = [(await client.post(f"{API}/auth/login", json={"email": "target@example.com", "password": "Wrong12345"})).status_code for _ in range(11)]
    assert codes[:10] == [401] * 10 and codes[10] == 429


async def test_limiter_failure_fails_closed_outside_development(client, monkeypatch):
    from app.middleware import rate_limit

    class Broken:
        async def hit(self, key, window):
            raise ConnectionError("redis down")

    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    rate_limit.set_backend(Broken())
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    r = await client.post(f"{API}/auth/login", json={"email": "x@example.com", "password": "Wrong12345"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "RATE_LIMITER_UNAVAILABLE"
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    r = await client.post(f"{API}/auth/login", json={"email": "x@example.com", "password": "Wrong12345"})
    assert r.status_code == 401  # development fails open (logged)


async def test_secrets_never_reach_logs(client, auth, agent, caplog):
    from app.core.logging import JsonFormatter

    caplog.set_level(logging.INFO)
    conv = await create_conversation(client, auth)
    await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "q"}, headers=auth)
    share = (await client.post(f"{API}/conversations/{conv['id']}/share", headers=auth)).json()
    await client.get(f"{API}/shares/{share['token']}")
    await client.post(f"{API}/auth/refresh")
    fmt = JsonFormatter()
    # "httpx" records here come from the *test client* logging the URLs it requests; the app's
    # own outbound httpx logging is silenced by configure_logging (asserted below).
    text = "\n".join(fmt.format(r) for r in caplog.records if r.name != "httpx")
    assert "/api/v1/shares/{share_token}" in text  # route template is logged...
    assert share["token"] not in text  # ...the token is not
    assert auth["Authorization"].split()[1] not in text and PASSWORD not in text
    assert "eyJ" not in text

    from app.core.logging import configure_logging

    configure_logging()
    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING


def test_redaction_filter():
    from advancerag_common.observability import redact

    leaky = 'Authorization: Bearer abc.def-123 api_key=sk-1234567890abcd token="xyz" ?key=AIzaSECRET eyJhbGciOi.eyJzdWIiOiIx.c2lnbmF0dXJl'
    out = redact(leaky)
    for secret in ("abc.def-123", "sk-1234567890abcd", "xyz", "AIzaSECRET", "eyJhbGciOi"):
        assert secret not in out


async def test_readiness_reports_storage_and_services(client, monkeypatch):
    r = await client.get("/ready")
    body = r.json()
    assert r.status_code == 200 and body["checks"]["database"] and body["checks"]["storage"]
    assert body["services_required"] is False
    monkeypatch.setattr(settings, "READINESS_REQUIRE_SERVICES", True)
    r = await client.get("/ready")
    assert r.status_code == 503 and r.json()["checks"]["agent_core"] is False


async def test_metrics_endpoint(client):
    await client.get("/health")
    r = await client.get("/metrics")
    assert r.status_code == 200 and "http_request_ms_count" in r.text
