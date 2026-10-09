"""Ingestion lifecycle: asynchronous start + status, one active attempt per document, cleanup of
every partial write on failure/cancel/crash, delete during ingestion. Uses a recording vector store
so leftover vectors are counted exactly (the frozen RAG pipeline itself is unchanged)."""
import asyncio
import json
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import func, select

from rag_core import db as rag_db
from rag_core import errors, ingestion
from rag_core.config import settings
from rag_core.db import Chunk, IndexedDocument, IngestionAttempt, utcnow
from rag_core.embeddings import FakeEmbeddings
from rag_core.pipeline import Services
from rag_core.rerank import NoopReranker
from rag_core.vectorstore import VectorStore

U1, U2 = uuid.uuid4(), uuid.uuid4()
TEXT = "\n\n".join(f"Section {i}. The unit {i} maintenance interval is {i * 7} days and its torque is {i * 3} Nm." for i in range(1200))


class RecordingStore(VectorStore):
    """In-memory vector store: {namespace user_id: {vector id}}. Can fail or pause at the Nth upsert."""

    name = "recording"

    def __init__(self):
        self.ids = {}
        self.upserts = 0
        self.fail_at = None
        self.pause_at = None
        self.paused = asyncio.Event()
        self.release = asyncio.Event()

    async def upsert(self, user_id, records, db=None):
        self.upserts += 1
        if self.pause_at == self.upserts:
            self.paused.set()
            await self.release.wait()
        if self.fail_at == self.upserts:
            raise errors.vector_store_unavailable()
        self.ids.setdefault(user_id, set()).update(r.id for r in records)

    async def query(self, user_id, vector, top_k, document_ids, db=None):
        return []

    async def delete(self, user_id, ids, db=None):
        self.ids.setdefault(user_id, set()).difference_update(ids)

    async def count_existing(self, user_id, ids, db=None):
        return len(self.ids.get(user_id, set()) & set(ids))

    def count(self, user_id=U1):
        return len(self.ids.get(user_id, set()))


@pytest.fixture(autouse=True)
async def own_connections(database, tmp_path):
    """Concurrent sessions need real transactions: in-memory SQLite shares ONE connection, where a
    second session's rollback would undo the running ingestion's writes. PostgreSQL is used as is."""
    from conftest import IS_PG
    from sqlalchemy.pool import NullPool

    if IS_PG:
        yield
        return
    engine = rag_db.configure_engine(f"sqlite+aiosqlite:///{(tmp_path / 'rag.db').as_posix()}", poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.run_sync(rag_db.Base.metadata.create_all)
    yield
    await engine.dispose()


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(settings, "INGEST_UPSERT_SLICE", 100)
    monkeypatch.setattr(settings, "INGEST_EMBED_SLICE", 64)
    return RecordingStore()


@pytest.fixture
async def rag(store):
    from rag_core import main

    svc = Services(settings, FakeEmbeddings(64), store, NoopReranker())
    main.app.dependency_overrides[main.get_services] = lambda: svc
    import httpx

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://rag") as c:
        yield c, svc
    main.app.dependency_overrides.clear()
    await ingestion.shutdown()


def _meta(doc, user=U1, name="manual.txt"):
    return {"metadata": json.dumps({"document_id": str(doc), "user_id": str(user), "filename": name, "mime_type": "text/plain"})}


async def start(client, doc, user=U1, data=TEXT.encode()):
    return await client.post("/api/v1/rag/ingestions", data=_meta(doc, user), files={"file": ("manual.txt", data, "text/plain")})


async def wait_terminal(client, doc, user=U1, timeout=20.0):
    for _ in range(int(timeout / 0.02)):
        r = await client.get(f"/api/v1/rag/ingestions/{doc}", params={"user_id": str(user)})
        if r.status_code == 404 or r.json()["status"] in ("succeeded", "failed", "stale"):
            return r
        await asyncio.sleep(0.02)
    raise AssertionError("ingestion did not finish")


async def db_counts(doc):
    async with rag_db.SessionLocal() as s:
        chunks = await s.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == doc))
        indexed = await s.get(IndexedDocument, doc)
        attempt = await s.get(IngestionAttempt, doc)
    return chunks, indexed, attempt


async def test_start_returns_immediately_and_status_reaches_succeeded(rag, store):
    client, _ = rag
    doc = uuid.uuid4()
    r = await start(client, doc)
    assert r.status_code == 202 and r.json()["status"] == "running" and r.json()["started"] is True
    final = (await wait_terminal(client, doc)).json()
    chunks, indexed, attempt = await db_counts(doc)
    assert final["status"] == "succeeded" and final["chunks"] == chunks == indexed.chunk_count > 100
    assert store.count() == chunks  # one vector per chunk, nothing else
    assert {"parse_s", "embed_s", "upsert_s", "pages"} <= set(final["timings"])


async def test_duplicate_start_joins_the_running_attempt(rag, store):
    client, _ = rag
    doc = uuid.uuid4()
    store.pause_at = 1
    first = (await start(client, doc)).json()
    await asyncio.wait_for(store.paused.wait(), 10)
    second = await start(client, doc)  # retry / duplicate job while the first is mid-write
    assert second.status_code == 202 and second.json()["started"] is False and second.json()["attempt_id"] == first["attempt_id"]
    sync = await client.post("/api/v1/rag/ingest", data=_meta(doc), files={"file": ("manual.txt", TEXT.encode(), "text/plain")})
    assert sync.status_code == 409 and sync.json()["error"]["code"] == "INGESTION_IN_PROGRESS"
    store.release.set()
    final = (await wait_terminal(client, doc)).json()
    chunks, _, _ = await db_counts(doc)
    assert final["status"] == "succeeded" and final["attempt_id"] == first["attempt_id"]
    assert store.upserts == -(-chunks // 100) and store.count() == chunks  # exactly one ingestion ran


async def test_failure_after_partial_writes_leaves_nothing_behind(rag, store):
    client, _ = rag
    doc = uuid.uuid4()
    store.fail_at = 3  # two slices (200 vectors) written, then the vector store fails
    await start(client, doc)
    final = (await wait_terminal(client, doc)).json()
    chunks, indexed, attempt = await db_counts(doc)
    assert final["status"] == "failed" and final["error_code"] == "VECTOR_STORE_UNAVAILABLE" and final["retryable"] is True
    assert store.count() == 0 and chunks == 0 and indexed is None and attempt.vector_extent == 0
    # A retry after the transient failure starts a fresh attempt and completes cleanly.
    store.fail_at = None
    assert (await start(client, doc)).json()["started"] is True
    final = (await wait_terminal(client, doc)).json()
    chunks, _, _ = await db_counts(doc)
    assert final["status"] == "succeeded" and store.count() == chunks


async def test_failed_reingest_does_not_leave_the_partially_overwritten_old_index(rag, store):
    client, _ = rag
    doc = uuid.uuid4()
    await start(client, doc)
    assert (await wait_terminal(client, doc)).json()["status"] == "succeeded"
    store.upserts, store.fail_at = 0, 2
    await start(client, doc, data=(TEXT + "\n\nAppendix.").encode())
    final = (await wait_terminal(client, doc)).json()
    chunks, indexed, _ = await db_counts(doc)
    assert final["status"] == "failed" and store.count() == 0 and chunks == 0 and indexed is None


async def test_permanent_failures_are_not_retryable_and_write_nothing(rag, store):
    client, _ = rag
    doc = uuid.uuid4()
    await start(client, doc, data=b"   \n  ")
    final = (await wait_terminal(client, doc)).json()
    assert final["status"] == "failed" and final["error_code"] == "NO_EXTRACTABLE_TEXT" and final["retryable"] is False
    assert store.upserts == 0 and store.count() == 0


@pytest.mark.skipif(not __import__('conftest').IS_PG, reason="SQLite allows one writer: the delete would wait on the paused ingestion's transaction. Runs on PostgreSQL (TEST_DATABASE_URL).")
async def test_delete_during_ingestion_cancels_and_removes_partial_writes(rag, store):
    client, _ = rag
    doc = uuid.uuid4()
    store.pause_at = 2  # first slice written, second about to be written
    await start(client, doc)
    await asyncio.wait_for(store.paused.wait(), 10)
    assert store.count() == 100
    r = await client.delete(f"/api/v1/rag/documents/{doc}", params={"user_id": str(U1)})
    assert r.status_code == 204
    store.release.set()  # the paused write lands after the delete...
    for _ in range(500):
        if doc not in ingestion._tasks:
            break
        await asyncio.sleep(0.02)
    chunks, indexed, attempt = await db_counts(doc)
    assert store.count() == 0 and chunks == 0 and indexed is None and attempt is None  # ...and is removed again
    assert (await client.get(f"/api/v1/rag/ingestions/{doc}", params={"user_id": str(U1)})).status_code == 404


async def test_stale_attempt_from_a_crashed_process_is_taken_over_and_its_leftovers_removed(rag, store):
    client, _ = rag
    doc = uuid.uuid4()
    # A process died after writing 900 vectors (more than this document will have).
    store.ids[U1] = {f"{doc}:{i}" for i in range(900)}
    old = utcnow() - timedelta(seconds=settings.INGEST_STALE_SECONDS + 5)
    async with rag_db.SessionLocal() as s:
        s.add(IngestionAttempt(document_id=doc, user_id=U1, attempt_id=uuid.uuid4(), status="running", stage="upserting",
                               chunk_count=0, vector_extent=900, retryable=False, heartbeat_at=old, started_at=old))
        await s.commit()
    status = (await client.get(f"/api/v1/rag/ingestions/{doc}", params={"user_id": str(U1)})).json()
    assert status["status"] == "stale" and status["error_code"] == "INGESTION_INTERRUPTED" and status["retryable"] is True
    assert (await start(client, doc)).json()["started"] is True
    final = (await wait_terminal(client, doc)).json()
    chunks, _, _ = await db_counts(doc)
    assert final["status"] == "succeeded" and store.count() == chunks < 900  # no ids beyond the new document


async def test_shutdown_mid_ingestion_records_interrupted_and_cleans_up(rag, store):
    client, _ = rag
    doc = uuid.uuid4()
    store.pause_at = 2
    await start(client, doc)
    await asyncio.wait_for(store.paused.wait(), 10)
    await ingestion.shutdown()
    chunks, indexed, attempt = await db_counts(doc)
    assert attempt.status == "failed" and attempt.error_code == "INGESTION_INTERRUPTED" and attempt.retryable
    assert store.count() == 0 and chunks == 0 and indexed is None


async def test_other_users_cannot_start_or_read_an_ingestion(rag, store):
    client, _ = rag
    doc = uuid.uuid4()
    await start(client, doc)
    await wait_terminal(client, doc)
    r = await start(client, doc, user=U2)
    assert r.status_code == 409 and r.json()["error"]["code"] == "DOCUMENT_OWNERSHIP_MISMATCH"
    assert (await client.get(f"/api/v1/rag/ingestions/{doc}", params={"user_id": str(U2)})).status_code == 404
    assert (await client.delete(f"/api/v1/rag/documents/{doc}", params={"user_id": str(U2)})).status_code == 404
    assert store.count(U2) == 0 and store.count() > 0


@pytest.mark.skipif(not __import__('conftest').IS_PG, reason="Needs concurrent writers (PostgreSQL); SQLite allows one writer.")
async def test_attempt_running_in_this_process_is_alive_even_with_an_old_heartbeat(rag, store):
    """Right after the host resumes from sleep the heartbeat is hours old, but the attempt is alive
    in this process: it must not be reported stale nor taken over by a duplicate start."""
    client, _ = rag
    doc = uuid.uuid4()
    store.pause_at = 1
    first = (await start(client, doc)).json()
    await asyncio.wait_for(store.paused.wait(), 10)
    async with rag_db.SessionLocal() as s:
        row = await s.get(IngestionAttempt, doc)
        row.heartbeat_at = utcnow() - timedelta(hours=6)
        await s.commit()
    status = (await client.get(f"/api/v1/rag/ingestions/{doc}", params={"user_id": str(U1)})).json()
    assert status["status"] == "running"
    again = (await start(client, doc)).json()
    assert again["started"] is False and again["attempt_id"] == first["attempt_id"]
    store.release.set()
    final = (await wait_terminal(client, doc)).json()
    chunks, _, _ = await db_counts(doc)
    assert final["status"] == "succeeded" and final["attempt_id"] == first["attempt_id"] and store.count() == chunks


@pytest.mark.skipif(not __import__('conftest').IS_PG, reason="Needs concurrent writers (PostgreSQL); SQLite allows one writer.")
async def test_superseded_attempt_never_discards_the_new_owners_vectors(rag, store):
    client, _ = rag
    doc = uuid.uuid4()
    store.pause_at = 2
    await start(client, doc)
    await asyncio.wait_for(store.paused.wait(), 10)
    newer = uuid.uuid4()
    async with rag_db.SessionLocal() as s:  # another process took the document over (stale lease elsewhere)
        row = await s.get(IngestionAttempt, doc)
        row.attempt_id = newer
        await s.commit()
    store.ids.setdefault(U1, set()).update(f"{doc}:{i}" for i in range(100, 300))  # the newer attempt's writes
    store.release.set()
    for _ in range(500):
        if doc not in ingestion._tasks:
            break
        await asyncio.sleep(0.02)
    async with rag_db.SessionLocal() as s:
        row = await s.get(IngestionAttempt, doc)
    assert row.attempt_id == newer and row.status == "running"  # untouched
    assert {f"{doc}:{i}" for i in range(100, 300)} <= store.ids[U1]  # the new owner's vectors survive
