"""Ingestion job lifecycle (backend side): the job starts RAG Core's asynchronous ingestion (or joins
an attempt that is already running), polls with a heartbeat, and never treats a slow ingestion as
a failure. Failures, crashes and deletes end in a clean, truthful state."""
import uuid
from datetime import timedelta

from sqlalchemy import select

from app.core.config import settings
from app.core.time import utcnow
from app.db import session as db_session
from app.db.models import BackgroundJob, Document
from app.integrations.rag_core import RAGCoreError
from app.workers import cleanup_jobs
from app.workers.runner import process_one
from conftest import upload

API = "/api/v1"
RUNNING = {"status": "running", "stage": "embedding", "attempt_id": "a1"}
SUCCEEDED = {"status": "succeeded", "stage": "done", "chunks": 4850, "attempt_id": "a1"}


async def _jobs(db, job_type="document_ingest"):
    return (await db.scalars(select(BackgroundJob).where(BackgroundJob.job_type == job_type).order_by(BackgroundJob.created_at))).all()


async def _doc(client, auth, doc):
    return (await client.get(f"{API}/documents/{doc['id']}", headers=auth)).json()


async def test_long_ingestion_stays_processing_and_heartbeats_until_ready(client, auth, rag, storage, monkeypatch):
    """The 812-page case: RAG Core is busy far longer than any HTTP timeout. The job keeps polling,
    the document stays `processing`, and the job's heartbeat keeps it from looking crashed."""
    doc = (await upload(client, auth)).json()
    seen = []
    original = rag.ingestion_status

    async def status(**kw):
        result = await original(**kw)
        async with db_session.session_scope() as db:
            job = (await _jobs(db))[0]
            seen.append((job.locked_at, (await db.get(Document, uuid.UUID(doc["id"]))).status, job.progress))
        return result

    rag.ingestion_status = status
    start = rag.start_ingestion

    async def start_running(**kw):
        await start(**kw)
        rag.attempts[kw["document_id"]] = dict(RUNNING)
        return dict(RUNNING)

    rag.start_ingestion = start_running
    rag.next_statuses = [None] + [dict(RUNNING)] * 25 + [dict(SUCCEEDED)]  # first call: no attempt yet
    await process_one(rag=rag, storage=storage)

    assert len(rag.ingested) == 1  # exactly one ingestion started
    assert all(s == "processing" for _, s, _ in seen[1:-1]) and seen[-1][1] == "processing"
    beats = [t for t, _, _ in seen[1:]]
    assert beats == sorted(beats) and beats[-1] > beats[0]  # heartbeat advanced on every poll
    assert any(p == "rag:embedding" for _, _, p in seen)
    assert (await _doc(client, auth, doc))["status"] == "ready"
    async with db_session.session_scope() as db:
        job = (await _jobs(db))[0]
    assert job.status == "succeeded" and job.attempts == 1 and job.progress == "indexed"


async def test_retried_job_joins_the_running_attempt_instead_of_starting_another(client, auth, rag, storage):
    doc = (await upload(client, auth)).json()
    rag.attempts[uuid.UUID(doc["id"])] = dict(RUNNING)  # e.g. the previous worker died while RAG Core kept going
    rag.next_statuses = [dict(RUNNING), dict(RUNNING), dict(SUCCEEDED)]
    await process_one(rag=rag, storage=storage)
    assert rag.ingested == []  # joined, not restarted
    assert (await _doc(client, auth, doc))["status"] == "ready"


async def test_already_succeeded_attempt_is_recorded_without_reingesting(client, auth, rag, storage):
    doc = (await upload(client, auth)).json()
    rag.attempts[uuid.UUID(doc["id"])] = dict(SUCCEEDED)
    await process_one(rag=rag, storage=storage)
    assert rag.ingested == [] and (await _doc(client, auth, doc))["status"] == "ready"


async def test_permanent_rag_failure_fails_once_with_a_truthful_message(client, auth, rag, storage):
    doc = (await upload(client, auth)).json()
    rag.start_status = dict(RUNNING)
    rag.next_statuses = [None, {"status": "failed", "error_code": "PARSE_FAILED", "retryable": False}]
    await process_one(rag=rag, storage=storage)
    body = await _doc(client, auth, doc)
    assert body["status"] == "failed" and body["processing_error"] == "The file appears to be damaged or password-protected."
    async with db_session.session_scope() as db:
        job = (await _jobs(db))[0]
    assert job.status == "failed" and job.attempts == 1  # not retried


async def test_crashed_rag_attempt_is_discarded_then_retried_cleanly(client, auth, rag, storage, monkeypatch):
    from app.services import job_service

    monkeypatch.setattr(job_service, "retry_delay", lambda attempt: timedelta(0))
    doc = (await upload(client, auth)).json()
    rag.start_status = dict(RUNNING)
    rag.next_statuses = [None, {"status": "stale", "error_code": "INGESTION_INTERRUPTED", "retryable": True}]
    await process_one(rag=rag, storage=storage)
    assert [str(d) for d in rag.deleted] == [doc["id"]]  # partial writes of the dead attempt removed
    assert (await _doc(client, auth, doc))["status"] == "processing"  # retry scheduled, not failed
    rag.start_status = None
    await process_one(rag=rag, storage=storage)  # fresh attempt
    assert len(rag.ingested) == 2 and (await _doc(client, auth, doc))["status"] == "ready"


async def test_status_blips_are_tolerated_within_the_budget(client, auth, rag, storage):
    doc = (await upload(client, auth)).json()
    calls = {"n": 0}
    original = rag.ingestion_status

    async def flaky(**kw):
        calls["n"] += 1
        if calls["n"] in (2, 3):
            raise RAGCoreError("RAG_SERVICE_UNAVAILABLE", retryable=True)
        return await original(**kw)

    rag.ingestion_status = flaky
    rag.next_statuses = [None, dict(RUNNING), dict(SUCCEEDED)]
    await process_one(rag=rag, storage=storage)
    assert len(rag.ingested) == 1 and (await _doc(client, auth, doc))["status"] == "ready"


async def test_delete_during_ingestion_cancels_job_and_discards_rag_data(client, auth, rag, storage):
    doc = (await upload(client, auth)).json()
    original = rag.ingestion_status
    polls = {"n": 0}

    async def status_then_user_deletes(**kw):
        polls["n"] += 1
        if polls["n"] == 3:
            assert (await client.delete(f"{API}/documents/{doc['id']}", headers=auth)).status_code == 204
        return await original(**kw)

    rag.ingestion_status = status_then_user_deletes
    rag.next_statuses = [None] + [dict(RUNNING)] * 10
    start = rag.start_ingestion

    async def start_running(**kw):
        await start(**kw)
        return dict(RUNNING)

    rag.start_ingestion = start_running
    await process_one(rag=rag, storage=storage)
    async with db_session.session_scope() as db:
        job = (await _jobs(db))[0]
        row = await db.get(Document, uuid.UUID(doc["id"]))
    assert job.status == "cancelled" and job.progress == "document_deleted"
    assert row.status == "deleted"  # never resurrected as ready
    assert [str(d) for d in rag.deleted] == [doc["id"]]  # running attempt cancelled + partial data removed


async def test_live_job_is_not_mistaken_for_a_crashed_one(client, auth):
    """requeue_stale_jobs only touches jobs whose heartbeat stopped."""
    await upload(client, auth)
    async with db_session.session_scope() as db:
        job = (await _jobs(db))[0]
        job.status, job.attempts, job.locked_at = "running", 1, utcnow() - timedelta(seconds=30)  # heartbeating
        await db.commit()
        assert await cleanup_jobs.requeue_stale_jobs(db) == 0
        job.locked_at = utcnow() - timedelta(seconds=settings.JOB_STALE_AFTER_SECONDS + 1)  # worker died
        await db.commit()
        assert await cleanup_jobs.requeue_stale_jobs(db) == 1
        await db.refresh(job)
        assert job.status == "pending"


async def test_final_failure_queues_rag_cleanup_that_respects_a_later_retry(client, auth, rag, storage, monkeypatch):
    from app.services import job_service

    monkeypatch.setattr(job_service, "retry_delay", lambda attempt: timedelta(0))
    doc = (await upload(client, auth)).json()
    rag.fail = RAGCoreError("RAG_SERVICE_UNAVAILABLE", retryable=True)
    for _ in range(settings.JOB_MAX_ATTEMPTS):
        await process_one(rag=rag, storage=storage)
    assert (await _doc(client, auth, doc))["status"] == "failed"
    async with db_session.session_scope() as db:
        cleanup = await _jobs(db, "document_delete")
    assert len(cleanup) == 1 and cleanup[0].payload == {"purpose": "cleanup_failed_ingest"}
    rag.fail = None
    await process_one(rag=rag, storage=storage)  # RAG-only cleanup: no file deletion, row kept
    assert [str(d) for d in rag.deleted] == [doc["id"]]
    assert (await _doc(client, auth, doc))["status"] == "failed" and await storage.exists(
        (await _stored_key(doc["id"])))

    # A cleanup that runs after the user already retried must not touch the new ingestion.
    rag.deleted.clear()
    async with db_session.session_scope() as db:
        user_id = (await db.get(Document, uuid.UUID(doc["id"]))).user_id
        job_service.enqueue(db, user_id, "document_delete", resource_id=uuid.UUID(doc["id"]), payload={"purpose": "cleanup_failed_ingest"})
        await db.commit()
    assert (await client.post(f"{API}/documents/{doc['id']}/reprocess", headers=auth)).status_code == 200
    while await process_one(rag=rag, storage=storage):
        pass
    assert rag.deleted == [] and (await _doc(client, auth, doc))["status"] == "ready"


async def _stored_key(doc_id):
    async with db_session.session_scope() as db:
        return (await db.get(Document, uuid.UUID(doc_id))).storage_key


# ---------------------------------------------------------------- ownership, sleep, staleness

async def test_stale_sweep_never_overwrites_a_job_that_finished_meanwhile(client, auth):
    """The lost update found live: the sweep read the job as running (stale lease), the runner then
    recorded `failed`, and the sweep wrote `pending` over it. The sweep is now a compare-and-set."""
    from types import SimpleNamespace

    from app.services import job_service

    await upload(client, auth)
    async with db_session.session_scope() as db:
        job = (await _jobs(db))[0]
        job.status, job.attempts, job.locked_at = "running", 1, utcnow() - timedelta(hours=2)
        await db.commit()
        job_id = job.id

    async with db_session.session_scope() as db:
        read = db.scalars

        async def read_then_runner_finishes(stmt, *a, **kw):
            rows = (await read(stmt, *a, **kw)).all()  # the sweep has seen the job as running...
            async with db_session.session_scope() as runner:  # ...then the runner records its outcome
                assert await job_service.finish(runner, job_id, 1, status="failed", error="PARSE_FAILED", locked_at=None)
                await runner.commit()
            return SimpleNamespace(all=lambda: rows)

        db.scalars = read_then_runner_finishes
        assert await cleanup_jobs.requeue_stale_jobs(db) == 0
    async with db_session.session_scope() as db:
        job = await db.get(BackgroundJob, job_id)
    assert job.status == "failed" and job.error == "PARSE_FAILED"


async def test_run_whose_job_was_recovered_stops_without_touching_state(client, auth, rag, storage):
    doc = (await upload(client, auth)).json()
    rag.start_status = dict(RUNNING)
    original = rag.ingestion_status
    polls = {"n": 0}

    async def sweep_recovers_job_mid_run(**kw):
        polls["n"] += 1
        if polls["n"] == 3:  # e.g. the host slept past the lease and the sweep re-queued the job
            async with db_session.session_scope() as db:
                job = (await _jobs(db))[0]
                job.status, job.locked_at = "pending", None
                await db.commit()
        return await original(**kw)

    rag.ingestion_status = sweep_recovers_job_mid_run
    rag.next_statuses = [None] + [dict(RUNNING)] * 10
    assert await process_one(rag=rag, storage=storage)
    async with db_session.session_scope() as db:
        job = (await _jobs(db))[0]
    assert job.status == "pending" and job.error is None  # the recovered state was not overwritten
    assert (await _doc(client, auth, doc))["status"] == "processing"  # and the document was not failed
    rag.next_statuses = [dict(SUCCEEDED)]
    assert await process_one(rag=rag, storage=storage)  # the recovered run joins and finishes
    assert (await _doc(client, auth, doc))["status"] == "ready" and len(rag.ingested) == 1


async def test_host_sleep_does_not_count_towards_the_ingestion_ceiling(client, auth, rag, storage, monkeypatch):
    from app.workers import document_jobs

    monkeypatch.setattr(settings, "INGEST_MAX_SECONDS", 600)
    clock = {"t": 1000.0}

    def fake_monotonic():
        clock["t"] += 0.5
        return clock["t"]

    monkeypatch.setattr(document_jobs.time, "monotonic", fake_monotonic)
    doc = (await upload(client, auth)).json()
    rag.start_status = dict(RUNNING)
    original = rag.ingestion_status
    polls = {"n": 0}

    async def sleeps_for_hours(**kw):
        polls["n"] += 1
        if polls["n"] == 3:
            clock["t"] += 6 * 3600  # the laptop slept for six hours mid-ingestion
        return await original(**kw)

    rag.ingestion_status = sleeps_for_hours
    rag.next_statuses = [None] + [dict(RUNNING)] * 6 + [dict(SUCCEEDED)]
    await process_one(rag=rag, storage=storage)
    assert (await _doc(client, auth, doc))["status"] == "ready" and rag.deleted == []


async def test_active_time_ceiling_still_stops_a_runaway_ingestion(client, auth, rag, storage, monkeypatch):
    monkeypatch.setattr(settings, "INGEST_MAX_SECONDS", 0.2)
    doc = (await upload(client, auth)).json()
    rag.start_status = dict(RUNNING)
    rag.next_statuses = [None] + [dict(RUNNING)] * 10_000
    await process_one(rag=rag, storage=storage)
    body = await _doc(client, auth, doc)
    assert body["status"] == "failed" and body["processing_error"].startswith("Indexing took too long")
    assert [str(d) for d in rag.deleted] == [doc["id"]]


async def test_a_single_stale_reading_does_not_discard_a_live_ingestion(client, auth, rag, storage, monkeypatch):
    monkeypatch.setattr(settings, "INGEST_STALE_CONFIRM_SECONDS", 3600)
    doc = (await upload(client, auth)).json()
    rag.start_status = dict(RUNNING)
    stale = {"status": "stale", "error_code": "INGESTION_INTERRUPTED", "retryable": True}
    rag.next_statuses = [None, dict(stale), dict(RUNNING), dict(SUCCEEDED)]  # e.g. right after a resume
    await process_one(rag=rag, storage=storage)
    assert rag.deleted == [] and (await _doc(client, auth, doc))["status"] == "ready"


async def test_worker_recovers_a_dead_jobs_lease_without_waiting_for_housekeeping(client, auth, monkeypatch):
    from app.workers.runner import Worker

    monkeypatch.setattr(settings, "STALE_SWEEP_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(settings, "RECONCILE_INTERVAL_SECONDS", 3600)
    monkeypatch.setattr(settings, "WORKER_POLL_INTERVAL_SECONDS", 3600)
    worker = Worker(concurrency=1)
    await worker.start()
    try:
        await upload(client, auth)
        async with db_session.session_scope() as db:
            job = (await _jobs(db))[0]  # a job whose worker died: lease (heartbeat) long expired
            job.status, job.attempts, job.locked_at = "running", 1, utcnow() - timedelta(seconds=settings.JOB_STALE_AFTER_SECONDS + 5)
            await db.commit()
        import asyncio
        for _ in range(100):
            await asyncio.sleep(0.05)
            async with db_session.session_scope() as db:
                status = (await _jobs(db))[0].status
            if status != "running":
                break
        assert status == "pending"
    finally:
        await worker.stop()
