"""Backend worker <-> real RAG Core app: duplicate ingestion jobs for one document run concurrently,
yet RAG Core executes exactly one ingestion (per-document lock) and the result is consistent."""
import asyncio
import uuid

from sqlalchemy import func, select

API = "/api/v1"
TEXT = "\n\n".join(f"Clause {i}: the inspection interval for valve {i} is {i * 5} days." for i in range(300))


async def test_duplicate_ingest_jobs_run_one_rag_ingestion(client, stack):
    from app.db import session as backend_db
    from app.db.models import Document
    from app.integrations.rag_core import RAGCoreClient
    from app.integrations.storage import get_storage
    from app.services import job_service
    from app.workers.runner import process_one
    from rag_core import db as rag_db
    from rag_core import ingestion
    from rag_core.db import Chunk, IngestionAttempt

    r = await client.post(f"{API}/auth/register", json={"email": f"d-{uuid.uuid4().hex[:6]}@example.com", "password": "Contract-pass1", "full_name": "Dup"})
    h = {"Authorization": f"Bearer {r.json()['access_token']}"}
    doc = (await client.post(f"{API}/documents", files={"file": ("valves.txt", TEXT.encode(), "text/plain")}, headers=h)).json()
    async with backend_db.session_scope() as db:  # a duplicate job for the same document (e.g. redelivery)
        row = await db.get(Document, uuid.UUID(doc["id"]))
        job_service.enqueue(db, row.user_id, "document_ingest", resource_id=row.id)
        await db.commit()

    claims = []
    original = ingestion.IngestionManager.claim

    async def spy(self, **kw):
        row, started = await original(self, **kw)
        claims.append(started)
        return row, started

    ingestion.IngestionManager.claim = spy
    try:
        rag = RAGCoreClient("http://rag-core")
        results = await asyncio.gather(process_one(rag=rag, storage=get_storage()), process_one(rag=rag, storage=get_storage()))
    finally:
        ingestion.IngestionManager.claim = original

    assert results == [True, True]
    assert claims.count(True) == 1  # one ingestion started; the other job joined it (or found it done)
    assert (await client.get(f"{API}/documents/{doc['id']}", headers=h)).json()["status"] == "ready"
    async with rag_db.SessionLocal() as s:
        chunks = await s.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == uuid.UUID(doc["id"])))
        with_vectors = await s.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == uuid.UUID(doc["id"]), Chunk.embedding.is_not(None)))
        attempt = await s.get(IngestionAttempt, uuid.UUID(doc["id"]))
    assert chunks > 5 and with_vectors == chunks and attempt.status == "succeeded" and attempt.chunk_count == chunks
