import pytest
from sqlalchemy import select
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.db import session as db_session
from app.db.models import BackgroundJob, Document
from app.integrations.rag_core import RAGCoreError
from app.workers.runner import process_one
# pyrefly: ignore [missing-import]
from conftest import create_conversation, upload

API = "/api/v1"
PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF"


async def drain(rag, storage, max_jobs=20):
    for _ in range(max_jobs):
        if not await process_one(rag=rag, storage=storage):
            return


async def test_upload_creates_record_job_and_file(client, auth, storage):
    r = await upload(client, auth, "report.pdf", PDF, "application/pdf")
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["status"] == "uploaded" and doc["size_bytes"] == len(PDF) and doc["mime_type"] == "application/pdf"
    assert "storage_key" not in doc and "user_id" not in doc

    async with db_session.session_scope() as db:
        row = await db.get(Document, __import__("uuid").UUID(doc["id"]))
        jobs = (await db.scalars(select(BackgroundJob))).all()
    assert row.storage_key.startswith("documents/") and "report" not in row.storage_key
    assert await storage.exists(row.storage_key)
    assert [j.job_type for j in jobs] == ["document_ingest"]


async def test_ingestion_lifecycle_reaches_ready(client, auth, rag, storage):
    doc = (await upload(client, auth, "report.pdf", PDF, "application/pdf")).json()
    await drain(rag, storage)
    r = await client.get(f"{API}/documents/{doc['id']}", headers=auth)
    assert r.json()["status"] == "ready"
    assert rag.ingested[0]["content"] == PDF and rag.ingested[0]["mime_type"] == "application/pdf"


async def test_ready_documents_are_passed_to_agent(client, auth, rag, storage, agent):
    """Knowledge Base scope: a ready document is searched in any conversation without attaching it."""
    doc = (await upload(client, auth)).json()
    await drain(rag, storage)
    for _ in range(2):  # two brand-new conversations
        conv = await create_conversation(client, auth)
        await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "q"}, headers=auth)
        req = agent.requests[-1]
        assert req.document_ids == [] and [d["document_id"] for d in req.library_documents] == [doc["id"]]


@pytest.fixture
async def file_database(tmp_path):
    """Real per-session connections (the default in-memory StaticPool shares one connection,
    which cannot model concurrent workers)."""
    from app.db.base import Base
    from conftest import IS_PG

    if IS_PG:  # the main PostgreSQL fixture already uses real pooled connections
        yield
        return
    engine = db_session.configure_engine(f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}", poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


@pytest.mark.usefixtures("file_database")
async def test_concurrent_workers_never_run_the_same_job_twice(client, auth, rag, storage):
    import asyncio

    for i in range(3):
        await upload(client, auth, f"c{i}.txt", f"content {i}".encode())
    results = await asyncio.gather(*[process_one(rag=rag, storage=storage) for _ in range(6)])
    assert sum(results) == 3
    assert sorted(str(d["document_id"]) for d in rag.ingested) == sorted(set(str(d["document_id"]) for d in rag.ingested))
    assert len(rag.ingested) == 3


async def test_rag_unavailable_retries_then_fails_safely(client, auth, rag, storage, monkeypatch):
    from app.services import job_service
    from datetime import timedelta

    monkeypatch.setattr(job_service, "retry_delay", lambda attempt: timedelta(0))
    rag.fail = RAGCoreError("RAG_SERVICE_UNAVAILABLE", retryable=True)
    doc = (await upload(client, auth)).json()

    await process_one(rag=rag, storage=storage)
    assert (await client.get(f"{API}/documents/{doc['id']}", headers=auth)).json()["status"] == "processing"
    await drain(rag, storage)
    body = (await client.get(f"{API}/documents/{doc['id']}", headers=auth)).json()
    assert body["status"] == "failed"
    assert body["processing_error"] == "The document processing service was unavailable. No indexed data was retained. Please try again."

    async with db_session.session_scope() as db:
        job = await db.scalar(select(BackgroundJob))
    assert job.status == "failed" and job.attempts == settings.JOB_MAX_ATTEMPTS

    # Reprocess once RAG Core is back.
    rag.fail = None
    r = await client.post(f"{API}/documents/{doc['id']}/reprocess", headers=auth)
    assert r.status_code == 200 and r.json()["status"] == "uploaded"
    await drain(rag, storage)
    assert (await client.get(f"{API}/documents/{doc['id']}", headers=auth)).json()["status"] == "ready"


async def test_non_retryable_rejection_fails_immediately(client, auth, rag, storage):
    rag.fail = RAGCoreError("RAG_REJECTED_DOCUMENT", retryable=False)
    doc = (await upload(client, auth)).json()
    await process_one(rag=rag, storage=storage)
    body = (await client.get(f"{API}/documents/{doc['id']}", headers=auth)).json()
    assert body["status"] == "failed" and body["processing_error"] == "Document ingestion failed. No indexed data was retained."


async def test_rejects_unsupported_extension(client, auth):
    r = await upload(client, auth, "malware.exe", b"MZ\x90\x00", "application/octet-stream")
    assert r.status_code == 415 and r.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"
    r = await upload(client, auth, "noext", b"hello", "text/plain")
    assert r.status_code == 415


async def test_rejects_content_that_does_not_match_extension(client, auth):
    # Browser MIME says PDF, extension says PDF, bytes are an executable.
    r = await upload(client, auth, "invoice.pdf", b"MZ\x90\x00\x03\x00 this is an exe", "application/pdf")
    assert r.status_code == 415 and r.json()["error"]["code"] == "FILE_CONTENT_MISMATCH"


async def test_browser_mime_type_is_ignored(client, auth):
    r = await upload(client, auth, "notes.txt", b"plain", "application/x-msdownload")
    assert r.status_code == 201 and r.json()["mime_type"] == "text/plain"


async def test_rejects_oversized_and_empty_files(client, auth, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 1024)
    r = await upload(client, auth, "big.txt", b"a" * 2048)
    assert r.status_code == 413 and r.json()["error"]["code"] == "FILE_TOO_LARGE"
    # Way over the limit: rejected by the body-size middleware before parsing, with the same
    # user-facing message as the per-file check.
    r = await upload(client, auth, "huge.txt", b"a" * 200_000)
    assert r.status_code == 413 and r.json()["error"]["code"] == "FILE_TOO_LARGE"
    assert r.json()["error"]["message"].startswith("Files must be at most")
    r = await upload(client, auth, "empty.txt", b"")
    assert r.status_code == 422 and r.json()["error"]["code"] == "EMPTY_FILE"


@pytest.mark.usefixtures("storage")
async def test_path_traversal_filename_is_neutralised(client, auth, tmp_path):
    r = await upload(client, auth, "../../../../evil.txt", b"x")
    assert r.status_code == 201 and r.json()["filename"] == "evil.txt"
    assert not (tmp_path.parent / "evil.txt").exists()


async def test_document_quota(client, auth, monkeypatch):
    monkeypatch.setattr(settings, "MAX_DOCUMENTS_PER_USER", 2)
    for i in range(2):
        assert (await upload(client, auth, f"d{i}.txt", f"quota {i}".encode())).status_code == 201
    r = await upload(client, auth, "d3.txt", b"quota 3")
    assert r.status_code == 409 and r.json()["error"]["code"] == "DOCUMENT_QUOTA_EXCEEDED"


async def test_many_documents_and_listing(client, auth):
    for i in range(7):
        assert (await upload(client, auth, f"doc{i}.md", f"# title {i}".encode())).status_code == 201
    page = (await client.get(f"{API}/documents?limit=5", headers=auth)).json()
    assert page["total"] == 7 and len(page["items"]) == 5
    assert (await client.get(f"{API}/documents?status=uploaded", headers=auth)).json()["total"] == 7


async def test_attach_detach_and_delete(client, auth, rag, storage):
    conv = await create_conversation(client, auth)
    doc = (await upload(client, auth)).json()
    r = await client.post(f"{API}/conversations/{conv['id']}/documents", json={"document_ids": [doc["id"], doc["id"]]}, headers=auth)
    assert r.status_code == 200 and len(r.json()) == 1
    # Idempotent re-attach.
    assert (await client.post(f"{API}/conversations/{conv['id']}/documents", json={"document_ids": [doc["id"]]}, headers=auth)).status_code == 200

    assert (await client.delete(f"{API}/conversations/{conv['id']}/documents/{doc['id']}", headers=auth)).status_code == 204
    assert (await client.get(f"{API}/conversations/{conv['id']}/documents", headers=auth)).json() == []
    assert (await client.delete(f"{API}/conversations/{conv['id']}/documents/{doc['id']}", headers=auth)).status_code == 404

    await client.post(f"{API}/conversations/{conv['id']}/documents", json={"document_ids": [doc["id"]]}, headers=auth)
    async with db_session.session_scope() as db:
        key = (await db.get(Document, __import__("uuid").UUID(doc["id"]))).storage_key

    assert (await client.delete(f"{API}/documents/{doc['id']}", headers=auth)).status_code == 204
    # Immediately hidden and detached...
    assert (await client.get(f"{API}/documents/{doc['id']}", headers=auth)).status_code == 404
    assert (await client.get(f"{API}/conversations/{conv['id']}/documents", headers=auth)).json() == []
    # ...then vectors, file and row are removed by the worker.
    await drain(rag, storage)
    assert [str(d) for d in rag.deleted] == [doc["id"]]
    assert not await storage.exists(key)
    async with db_session.session_scope() as db:
        assert await db.get(Document, __import__("uuid").UUID(doc["id"])) is None


async def test_download_is_attachment_with_nosniff(client, auth):
    doc = (await upload(client, auth, "notes.txt", b"hello world")).json()
    r = await client.get(f"{API}/documents/{doc['id']}/content", headers=auth)
    assert r.status_code == 200 and r.content == b"hello world"
    assert r.headers["content-disposition"].startswith("attachment;")
    assert r.headers["x-content-type-options"] == "nosniff"
