"""Sleeping dependencies (Render free): chat turns and document jobs wake them first, within a bounded
budget, keep their leases alive while waiting, and fail truthfully (and retryably) when they do not wake."""
import asyncio
import uuid

import pytest
from sqlalchemy import select

from advancerag_common import wake
from app.core.config import settings
from app.db import session as db_session
from app.db.models import BackgroundJob
from app.integrations.agent_core import StreamEvent
from app.workers import document_jobs
from app.workers.runner import process_one
from conftest import create_conversation, parse_sse, upload

API = "/api/v1"
AGENT_URL, RAG_URL, LLM_URL = "https://agent.test", "https://rag.test", "https://llm.test"


@pytest.fixture
def urls(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_CORE_URL", AGENT_URL)
    monkeypatch.setattr(settings, "RAG_CORE_URL", RAG_URL)
    monkeypatch.setattr(settings, "WAKE_EXTRA_URLS", LLM_URL)
    monkeypatch.setattr(settings, "WAKE_RELAY_URL", "https://web.test/wake")


def fake_wake(monkeypatch, awake: dict, delay: float = 0.0):
    calls = []

    async def ensure_awake(url, *, budget, relay=None, **kw):
        calls.append((url, budget, relay))
        await asyncio.sleep(delay)
        return awake.get(url, True)

    monkeypatch.setattr(wake, "ensure_awake", ensure_awake)
    return calls


class Agent:
    def __init__(self):
        self.calls = 0

    async def stream(self, req):
        self.calls += 1
        yield StreamEvent("content.delta", {"delta": "Hello!"})
        yield StreamEvent("generation.completed", {"citations": [], "metadata": {}})


@pytest.fixture
def stream_agent():
    from app.integrations.agent_core import get_agent_client
    from app.main import app

    agent = Agent()
    app.dependency_overrides[get_agent_client] = lambda: agent
    yield agent
    app.dependency_overrides.pop(get_agent_client, None)


async def _send(client, auth, text="hi"):
    conv = await create_conversation(client, auth)
    r = await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": text}, headers=auth)
    return parse_sse(r.text)


async def test_chat_waits_for_agent_and_llm_then_answers(client, auth, stream_agent, urls, monkeypatch):
    monkeypatch.setattr(settings, "CHAT_WAKE_SECONDS", 90)
    calls = fake_wake(monkeypatch, {})
    events = await _send(client, auth)
    assert events[-1]["event"] == "message.completed" and stream_agent.calls == 1
    # No documents: RAG Core is not needed, so general chat never waits on it.
    assert sorted(u for u, _, _ in calls) == [AGENT_URL, LLM_URL]
    assert all(b == 90 and r == "https://web.test/wake" for _, b, r in calls)


async def test_chat_reports_a_clear_starting_error_when_the_agent_does_not_wake(client, auth, stream_agent, urls, monkeypatch):
    monkeypatch.setattr(settings, "CHAT_WAKE_SECONDS", 90)
    fake_wake(monkeypatch, {AGENT_URL: False})
    events = await _send(client, auth)
    assert events[-1]["event"] == "error"
    assert events[-1]["data"]["code"] == "AI_SERVICE_STARTING"
    assert "starting up" in events[-1]["data"]["message"]
    assert events[-1]["data"]["assistant_message"]["status"] == "failed"
    assert stream_agent.calls == 0  # never called a service that is not up


async def test_rag_not_waking_does_not_block_general_chat(client, auth, stream_agent, urls, monkeypatch, storage, rag):
    monkeypatch.setattr(settings, "CHAT_WAKE_SECONDS", 90)
    await upload(client, auth)
    await process_one(rag=rag, storage=storage)  # the user now has a ready document
    calls = fake_wake(monkeypatch, {RAG_URL: False})
    events = await _send(client, auth)
    assert RAG_URL in [u for u, _, _ in calls]  # woken because documents may be searched
    assert events[-1]["event"] == "message.completed"


async def test_waiting_turn_streams_status_events(client, auth, stream_agent, urls, monkeypatch):
    from app.services import agent_service

    monkeypatch.setattr(settings, "CHAT_WAKE_SECONDS", 90)
    monkeypatch.setattr(agent_service, "WAKING_STATUS_EVERY_SECONDS", 0.01)
    fake_wake(monkeypatch, {}, delay=0.1)
    events = await _send(client, auth)
    assert any(e["event"] == "agent.waking" for e in events)
    assert events[-1]["event"] == "message.completed"


async def test_wake_is_off_by_default(client, auth, stream_agent, monkeypatch):
    calls = fake_wake(monkeypatch, {AGENT_URL: False})
    events = await _send(client, auth)
    assert events[-1]["event"] == "message.completed" and calls == []


# ---------------------------------------------------------------- document jobs

async def _job(job_type="document_ingest"):
    async with db_session.session_scope() as db:
        return (await db.scalars(select(BackgroundJob).where(BackgroundJob.job_type == job_type))).all()[0]


async def test_ingestion_wakes_rag_core_first_and_heartbeats_while_waiting(client, auth, rag, storage, urls, monkeypatch):
    monkeypatch.setattr(settings, "RAG_WAKE_SECONDS", 300)
    monkeypatch.setattr(document_jobs, "WAKE_HEARTBEAT_SECONDS", 0.01)
    beats = []
    original = document_jobs._heartbeat

    async def beat(db, job, progress):
        beats.append(progress)
        await original(db, job, progress)

    monkeypatch.setattr(document_jobs, "_heartbeat", beat)
    calls = fake_wake(monkeypatch, {}, delay=0.1)
    doc = (await upload(client, auth)).json()
    await process_one(rag=rag, storage=storage)
    assert calls and calls[0][0] == RAG_URL and calls[0][1] == 300
    assert "waking_rag" in beats  # lease renewed during the wait
    assert len(rag.ingested) == 1  # exactly one ingestion
    assert (await client.get(f"{API}/documents/{doc['id']}", headers=auth)).json()["status"] == "ready"


async def test_rag_core_not_waking_is_a_bounded_retryable_failure_never_ready(client, auth, rag, storage, urls, monkeypatch):
    monkeypatch.setattr(settings, "RAG_WAKE_SECONDS", 300)
    fake_wake(monkeypatch, {RAG_URL: False})
    doc = (await upload(client, auth)).json()
    await process_one(rag=rag, storage=storage)
    job = await _job()
    assert job.status == "pending" and job.attempts == 1 and job.error == "RAG_SERVICE_UNAVAILABLE"  # retry scheduled
    assert rag.ingested == [] and rag.status_calls == 0  # RAG Core was not called while asleep
    assert (await client.get(f"{API}/documents/{doc['id']}", headers=auth)).json()["status"] == "processing"

    # Retries are bounded: after the last attempt the document fails with a truthful message.
    for _ in range(settings.JOB_MAX_ATTEMPTS - 1):
        async with db_session.session_scope() as db:
            j = await db.get(BackgroundJob, job.id)
            j.run_after = j.created_at
            await db.commit()
        await process_one(rag=rag, storage=storage)
    final = (await client.get(f"{API}/documents/{doc['id']}", headers=auth)).json()
    assert final["status"] == "failed" and "unavailable" in final["processing_error"]
    assert (await _job()).attempts == settings.JOB_MAX_ATTEMPTS and rag.ingested == []


async def test_delete_wakes_rag_core_before_removing_vectors(client, auth, rag, storage, urls, monkeypatch):
    await upload(client, auth)
    await process_one(rag=rag, storage=storage)
    monkeypatch.setattr(settings, "RAG_WAKE_SECONDS", 300)
    calls = fake_wake(monkeypatch, {})
    doc_id = (await client.get(f"{API}/documents", headers=auth)).json()["items"][0]["id"]
    assert (await client.delete(f"{API}/documents/{doc_id}", headers=auth)).status_code == 204
    await process_one(rag=rag, storage=storage)
    assert [u for u, _, _ in calls] == [RAG_URL] and rag.deleted == [uuid.UUID(doc_id)]


async def test_active_jobs_are_detected_for_keep_awake(client, auth):
    from app.services import job_service

    async with db_session.session_scope() as db:
        assert await job_service.has_active_jobs(db) is False
    await upload(client, auth)
    async with db_session.session_scope() as db:
        assert await job_service.has_active_jobs(db) is True
