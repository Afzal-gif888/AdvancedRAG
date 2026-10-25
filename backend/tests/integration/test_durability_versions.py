"""Durable ingestion, deduplication, versioning, deletion and storage backends.

Root cause being fixed (reproduced here): with the local-disk backend an upload is lost when the instance
restarts before (or while) its job runs, and jobs enqueued later (reprocess, reconciler) had no copy of the
file, so the worker reported FILE_MISSING ("The uploaded file could not be found"). Durable backends keep the
bytes with the document, and the job references the object instead of carrying a base64 copy.
"""
import asyncio
import io
import struct
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import func, select, update

from app.core.config import settings
from app.core.time import utcnow
from app.db import session as db_session
from app.db.models import BackgroundJob, Document, StoredObject, StoredObjectPart
from app.integrations import storage as storage_mod
from app.integrations.storage import DatabaseStorage, LocalStorage, S3Storage, get_storage_for, local_copy, set_storage
from app.workers import cleanup_jobs
from app.workers.runner import process_one
from conftest import IS_PG, create_conversation, register_and_login, upload

API = "/api/v1"
PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF"


async def drain(rag, storage, max_jobs=30):
    for _ in range(max_jobs):
        if not await process_one(rag=rag, storage=storage):
            return


async def doc_row(doc_id):
    async with db_session.session_scope() as db:
        return await db.get(Document, uuid.UUID(doc_id))


async def get(client, auth, doc_id):
    return (await client.get(f"{API}/documents/{doc_id}", headers=auth)).json()


@pytest.fixture
def db_storage(monkeypatch):
    """Switch uploads to the database backend (a real DatabaseStorage on the test database)."""
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "db")
    store = DatabaseStorage()
    set_storage(store)
    return store


# ---------------------------------------------------------------- the reported failure, reproduced and fixed

async def test_local_disk_loss_reproduces_file_missing(client, auth, rag, storage):
    """Documented limitation of the `local` backend: if the disk is wiped (restart on an ephemeral filesystem)
    before the job runs, the file cannot be found, and the failure is explicit and non-retryable."""
    doc = (await upload(client, auth, "a.txt", b"important text")).json()
    row = await doc_row(doc["id"])
    await storage.delete(row.storage_key)  # the instance restarted: ephemeral disk is empty
    await drain(rag, storage)
    after = await get(client, auth, doc["id"])
    assert after["status"] == "failed" and "could not be found" in after["processing_error"]
    assert rag.ingested == []
    async with db_session.session_scope() as db:
        job = (await db.scalars(select(BackgroundJob).where(BackgroundJob.job_type == "document_ingest"))).one()
    assert job.error == "FILE_MISSING" and job.attempts == 1, "a missing file is not retried"


async def test_database_storage_survives_losing_the_disk(client, auth, rag, db_storage):
    """Same scenario with durable storage: nothing lives on the disk, so there is nothing to lose."""
    doc = (await upload(client, auth, "a.txt", b"important text that must survive")).json()
    row = await doc_row(doc["id"])
    assert row.storage_backend == "db"
    await drain(rag, db_storage)
    assert (await get(client, auth, doc["id"]))["status"] == "ready"
    assert rag.ingested[0]["content"] == b"important text that must survive"


async def test_reprocess_after_failure_can_read_the_original_again(client, auth, rag, db_storage):
    doc = (await upload(client, auth, "a.txt", b"retry me")).json()
    rag.fail = __import__("app.integrations.rag_core", fromlist=["RAGCoreError"]).RAGCoreError("PARSE_FAILED", retryable=False)
    await drain(rag, db_storage)
    assert (await get(client, auth, doc["id"]))["status"] == "failed"
    rag.fail = None
    assert (await client.post(f"{API}/documents/{doc['id']}/reprocess", headers=auth)).status_code == 200
    await drain(rag, db_storage)
    assert (await get(client, auth, doc["id"]))["status"] == "ready"
    assert rag.ingested[-1]["content"] == b"retry me", "the reprocess job has no payload; it must read the stored object"


async def test_jobs_no_longer_carry_a_copy_of_the_file(client, auth, storage):
    await upload(client, auth, "a.txt", b"x" * 50_000)
    async with db_session.session_scope() as db:
        job = (await db.scalars(select(BackgroundJob))).one()
    assert job.payload == {}


async def test_legacy_job_with_base64_payload_still_works_and_is_scrubbed(client, auth, rag, storage):
    """Jobs queued by the previous version carry the file as base64; they are honoured once and then cleaned."""
    import base64

    doc = (await upload(client, auth, "a.txt", b"legacy bytes")).json()
    row = await doc_row(doc["id"])
    await storage.delete(row.storage_key)  # only the payload still has the file
    async with db_session.session_scope() as db:
        await db.execute(update(BackgroundJob).values(payload={"file_data": base64.b64encode(b"legacy bytes").decode()}))
        await db.commit()
    await drain(rag, storage)
    assert (await get(client, auth, doc["id"]))["status"] == "ready" and rag.ingested[0]["content"] == b"legacy bytes"
    async with db_session.session_scope() as db:
        assert (await db.scalars(select(BackgroundJob))).one().payload == {}


async def test_corrupted_object_is_detected_by_checksum(client, auth, rag, storage):
    doc = (await upload(client, auth, "a.txt", b"original content")).json()
    row = await doc_row(doc["id"])
    storage._path(row.storage_key).write_bytes(b"tampered content!")  # same size, different bytes
    await drain(rag, storage)
    after = await get(client, auth, doc["id"])
    assert after["status"] == "failed" and "did not match" in after["processing_error"] and rag.ingested == []


async def test_storage_outage_is_retried_not_reported_as_missing_file(client, auth, rag, storage, monkeypatch):
    doc = (await upload(client, auth, "a.txt", b"hello")).json()

    async def boom(key):
        raise ConnectionError("object store down")

    monkeypatch.setattr(storage, "exists", boom)
    await process_one(rag=rag, storage=storage)
    async with db_session.session_scope() as db:
        job = (await db.scalars(select(BackgroundJob))).one()
    assert job.status == "pending" and job.error == "STORAGE_UNAVAILABLE", "transient storage errors are retried with backoff"


async def test_extraction_report_is_stored_and_exposed(client, auth, rag, storage):
    doc = (await upload(client, auth, "a.pdf", PDF, "application/pdf")).json()
    report = {"method": "mixed", "partial": True, "total_pages": 3, "ocr_pages": 1, "unprocessed_pages": [3],
              "warnings": [{"page": 3, "code": "OCR_FAILED", "message": "OCR failed for this page."}]}
    rag.start_status = {"status": "succeeded", "stage": "done", "chunks": 3, "attempt_id": "a", "extraction": report}
    await drain(rag, storage)
    out = await get(client, auth, doc["id"])
    assert out["status"] == "ready" and out["extraction"]["partial"] is True and out["extraction"]["unprocessed_pages"] == [3]
    listing = (await client.get(f"{API}/documents", headers=auth)).json()["items"][0]
    assert listing["extraction"]["warnings"][0]["code"] == "OCR_FAILED"


# ---------------------------------------------------------------- database storage backend

async def test_database_storage_roundtrip_streams_in_parts_and_cleans_up(db_storage, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_DB_PART_BYTES", 64 * 1024)
    user = uuid.uuid4()
    async with db_session.session_scope() as db:
        from app.db.models import User

        db.add(User(id=user, email=f"{user.hex[:8]}@example.com", full_name="T", password_hash="x"))
        await db.commit()
    key = f"documents/{user}/{uuid.uuid4()}"
    data = bytes(range(256)) * 1000  # 256,000 bytes -> 4 parts
    await db_storage.put(key, io.BytesIO(data))
    assert await db_storage.exists(key) and await db_storage.read_bytes(key) == data
    assert [len(c) async for c in db_storage.stream(key)] == [65536, 65536, 65536, 59392]
    async with local_copy(db_storage, key) as path:
        assert path.read_bytes() == data
        copy = path
    assert not copy.exists(), "temporary copies are removed"
    await db_storage.delete(key)
    assert not await db_storage.exists(key)
    async with db_session.session_scope() as db:
        assert await db.scalar(select(func.count()).select_from(StoredObjectPart)) == 0


async def test_database_storage_is_atomic_with_the_document_row(client, auth, db_storage, monkeypatch):
    """If the transaction fails, no blob is left behind (it rolled back with the document)."""
    from app.services import job_service

    def broken(*a, **k):
        raise RuntimeError("enqueue failed")

    monkeypatch.setattr(job_service, "enqueue", broken)
    with pytest.raises(RuntimeError):
        await upload(client, auth, "a.txt", b"will not persist")
    async with db_session.session_scope() as db:
        assert await db.scalar(select(func.count()).select_from(StoredObject)) == 0
        assert await db.scalar(select(func.count()).select_from(Document)) == 0


async def test_download_works_for_each_backend(client, auth, db_storage):
    doc = (await upload(client, auth, "a.txt", b"download me")).json()
    r = await client.get(f"{API}/documents/{doc['id']}/content", headers=auth)
    assert r.status_code == 200 and r.content == b"download me" and r.headers["x-content-type-options"] == "nosniff"


def test_get_storage_for_returns_the_documents_own_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(storage_mod, "_storage", None)
    storage_mod._by_backend.clear()
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "db")
    monkeypatch.setattr(settings, "LOCAL_STORAGE_PATH", str(tmp_path))
    assert isinstance(get_storage_for("db"), DatabaseStorage)
    assert isinstance(get_storage_for("local"), LocalStorage)       # an older document stays readable after the switch
    assert isinstance(get_storage_for(None), LocalStorage)          # rows created before the column existed
    storage_mod._by_backend.clear()


def test_s3_backend_roundtrip_with_a_mocked_s3(monkeypatch, tmp_path):
    """S3Storage against moto's in-process S3: put, exists, stream, local_copy, delete."""
    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")
    with moto.mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="adv-test")
        s3 = S3Storage(bucket="adv-test", access_key="x", secret_key="y", region="us-east-1")
        key = f"documents/{uuid.uuid4()}/{uuid.uuid4()}"
        data = b"s3 object body " * 5000

        async def go():
            await s3.put(key, io.BytesIO(data))
            assert await s3.exists(key) and await s3.read_bytes(key) == data
            async with local_copy(s3, key) as path:
                assert path.read_bytes() == data
            await s3.delete(key)
            assert not await s3.exists(key)

        asyncio.run(go())


# ---------------------------------------------------------------- deduplication

async def test_duplicate_upload_returns_the_existing_document_and_queues_nothing(client, auth):
    first = await upload(client, auth, "a.txt", b"same content")
    again = await upload(client, auth, "renamed.txt", b"same content")
    assert first.status_code == 201 and again.status_code == 200
    assert again.json()["id"] == first.json()["id"] and again.json()["duplicate"] is True and first.json()["duplicate"] is False
    async with db_session.session_scope() as db:
        assert await db.scalar(select(func.count()).select_from(Document)) == 1
        assert await db.scalar(select(func.count()).select_from(BackgroundJob)) == 1


async def test_dedup_is_per_user_and_never_leaks_another_users_document(make_client):
    a, b = make_client(), make_client()
    ha, hb = await register_and_login(a), await register_and_login(b)
    ra = await upload(a, ha, "a.txt", b"shared secret text")
    rb = await upload(b, hb, "a.txt", b"shared secret text")
    assert rb.status_code == 201 and rb.json()["id"] != ra.json()["id"] and rb.json()["duplicate"] is False


async def test_failed_or_deleted_documents_do_not_block_a_new_upload(client, auth, rag, storage):
    first = (await upload(client, auth, "a.txt", b"text one")).json()
    assert (await client.delete(f"{API}/documents/{first['id']}", headers=auth)).status_code == 204
    again = await upload(client, auth, "a.txt", b"text one")
    assert again.status_code == 201 and again.json()["id"] != first["id"]


@pytest.mark.skipif(not IS_PG, reason="the advisory lock that serialises identical uploads is PostgreSQL-specific")
async def test_concurrent_identical_uploads_create_one_document(make_client):
    clients = [make_client() for _ in range(1)]
    c = clients[0]
    h = await register_and_login(c)
    results = await asyncio.gather(*[upload(c, h, f"copy{i}.txt", b"race content") for i in range(6)])
    assert sorted(r.status_code for r in results).count(201) == 1
    assert {r.json()["id"] for r in results} == {next(r.json()["id"] for r in results if r.status_code == 201)}
    async with db_session.session_scope() as db:
        assert await db.scalar(select(func.count()).select_from(Document)) == 1


# ---------------------------------------------------------------- versioning / replacement

async def ready_doc(client, auth, rag, storage, name="policy.txt", content=b"version one text"):
    doc = (await upload(client, auth, name, content)).json()
    await drain(rag, storage)
    assert (await get(client, auth, doc["id"]))["status"] == "ready"
    return doc


async def test_replacement_keeps_the_old_version_searchable_until_the_new_one_is_ready(client, auth, rag, storage):
    v1 = await ready_doc(client, auth, rag, storage)
    r = await client.post(f"{API}/documents/{v1['id']}/replace", files={"file": ("policy.txt", b"version two text", "text/plain")}, headers=auth)
    assert r.status_code == 201
    v2 = r.json()
    assert v2["version"] == 2 and v2["replaces_id"] == v1["id"] and v2["logical_id"] == v1["id"] and v2["status"] == "uploaded"
    # while v2 is still indexing, the library offers only v1
    from app.services import document_service

    owner = (await doc_row(v1["id"])).user_id
    async with db_session.session_scope() as db:
        ready = [d.id for d in await document_service.list_library_documents(db, owner, (), 50)]
    assert ready == [uuid.UUID(v1["id"])]
    await drain(rag, storage)
    assert (await get(client, auth, v2["id"]))["status"] == "ready"
    assert (await get(client, auth, v1["id"]))["status"] == "superseded"
    async with db_session.session_scope() as db:
        ready = [d.id for d in await document_service.list_library_documents(db, owner, (), 50)]
    assert ready == [uuid.UUID(v2["id"])], "exactly one version is searchable"
    listing = (await client.get(f"{API}/documents", headers=auth)).json()
    assert [d["id"] for d in listing["items"]] == [v2["id"]] and listing["total"] == 1


async def test_a_failed_replacement_leaves_the_current_version_untouched(client, auth, rag, storage):
    v1 = await ready_doc(client, auth, rag, storage)
    v2 = (await client.post(f"{API}/documents/{v1['id']}/replace", files={"file": ("p.txt", b"broken new version", "text/plain")}, headers=auth)).json()
    from app.integrations.rag_core import RAGCoreError

    rag.fail = RAGCoreError("PARSE_FAILED", retryable=False)
    await drain(rag, storage)
    assert (await get(client, auth, v2["id"]))["status"] == "failed"
    assert (await get(client, auth, v1["id"]))["status"] == "ready", "the old version keeps working"


async def test_identical_replacement_is_a_noop_and_concurrent_replacements_are_refused(client, auth, rag, storage):
    v1 = await ready_doc(client, auth, rag, storage)
    same = await client.post(f"{API}/documents/{v1['id']}/replace", files={"file": ("p.txt", b"version one text", "text/plain")}, headers=auth)
    assert same.status_code == 200 and same.json()["id"] == v1["id"] and same.json()["duplicate"] is True
    first = await client.post(f"{API}/documents/{v1['id']}/replace", files={"file": ("p.txt", b"version two", "text/plain")}, headers=auth)
    second = await client.post(f"{API}/documents/{v1['id']}/replace", files={"file": ("p.txt", b"version three", "text/plain")}, headers=auth)
    assert first.status_code == 201 and second.status_code == 409 and second.json()["error"]["code"] == "REPLACEMENT_IN_PROGRESS"


async def test_only_ready_documents_can_be_replaced(client, auth):
    d = (await upload(client, auth, "a.txt", b"still uploading")).json()
    r = await client.post(f"{API}/documents/{d['id']}/replace", files={"file": ("a.txt", b"new", "text/plain")}, headers=auth)
    assert r.status_code == 409 and r.json()["error"]["code"] == "DOCUMENT_NOT_REPLACEABLE"


async def test_versions_listing_and_rollback_roll_forward(client, auth, rag, storage):
    v1 = await ready_doc(client, auth, rag, storage)
    v2 = (await client.post(f"{API}/documents/{v1['id']}/replace", files={"file": ("p.txt", b"version two", "text/plain")}, headers=auth)).json()
    await drain(rag, storage)
    versions = (await client.get(f"{API}/documents/{v2['id']}/versions", headers=auth)).json()
    assert [(v["version"], v["status"]) for v in versions] == [(2, "ready"), (1, "superseded")]
    back = await client.post(f"{API}/documents/{v1['id']}/restore", headers=auth)
    assert back.status_code == 200 and back.json()["status"] == "ready"
    assert (await get(client, auth, v2["id"]))["status"] == "superseded"
    fwd = await client.post(f"{API}/documents/{v2['id']}/restore", headers=auth)
    assert fwd.json()["status"] == "ready" and (await get(client, auth, v1["id"]))["status"] == "superseded"
    # a document that is current cannot be "restored"
    assert (await client.post(f"{API}/documents/{v2['id']}/restore", headers=auth)).status_code == 404


async def test_conversation_attachments_follow_the_current_version(client, auth, rag, storage):
    v1 = await ready_doc(client, auth, rag, storage)
    conv = await create_conversation(client, auth)
    assert (await client.post(f"{API}/conversations/{conv['id']}/documents", json={"document_ids": [v1["id"]]}, headers=auth)).status_code == 200
    v2 = (await client.post(f"{API}/documents/{v1['id']}/replace", files={"file": ("p.txt", b"version two", "text/plain")}, headers=auth)).json()
    await drain(rag, storage)
    attached = (await client.get(f"{API}/conversations/{conv['id']}/documents", headers=auth)).json()
    assert [d["id"] for d in attached] == [v2["id"]]


async def test_deleting_a_document_deletes_every_version_everywhere(client, auth, rag, db_storage):
    v1 = await ready_doc(client, auth, rag, db_storage)
    v2 = (await client.post(f"{API}/documents/{v1['id']}/replace", files={"file": ("p.txt", b"version two", "text/plain")}, headers=auth)).json()
    await drain(rag, db_storage)
    assert (await client.delete(f"{API}/documents/{v2['id']}", headers=auth)).status_code == 204
    await drain(rag, db_storage)
    assert {str(d) for d in rag.deleted} >= {v1["id"], v2["id"]}
    async with db_session.session_scope() as db:
        assert await db.scalar(select(func.count()).select_from(Document)) == 0, "rows are hard-deleted once cleanup succeeded"
        assert await db.scalar(select(func.count()).select_from(StoredObject)) == 0
        assert await db.scalar(select(func.count()).select_from(StoredObjectPart)) == 0
    assert (await client.get(f"{API}/documents/{v1['id']}", headers=auth)).status_code == 404
    assert (await client.get(f"{API}/documents", headers=auth)).json()["total"] == 0


async def test_superseded_versions_are_purged_after_the_retention_window(client, auth, rag, storage, monkeypatch):
    monkeypatch.setattr(settings, "SUPERSEDED_RETENTION_DAYS", 7)
    v1 = await ready_doc(client, auth, rag, storage)
    await client.post(f"{API}/documents/{v1['id']}/replace", files={"file": ("p.txt", b"version two", "text/plain")}, headers=auth)
    await drain(rag, storage)
    async with db_session.session_scope() as db:
        assert await cleanup_jobs.purge_superseded_versions(db) == 0          # still inside the window: rollback possible
        await db.execute(update(Document).where(Document.id == uuid.UUID(v1["id"])).values(superseded_at=utcnow() - timedelta(days=8)))
        await db.commit()
        assert await cleanup_jobs.purge_superseded_versions(db) == 1
    await drain(rag, storage)
    assert uuid.UUID(v1["id"]) in rag.deleted
    assert (await client.get(f"{API}/documents/{v1['id']}", headers=auth)).status_code == 404


async def test_other_users_cannot_replace_restore_list_or_delete_my_versions(make_client, rag, storage):
    a, b = make_client(), make_client()
    ha, hb = await register_and_login(a), await register_and_login(b)
    d = (await upload(a, ha, "p.txt", b"owner text")).json()
    await drain(rag, storage)
    for method, path, kw in [
        ("POST", f"/documents/{d['id']}/replace", {"files": {"file": ("p.txt", b"hijack", "text/plain")}}),
        ("POST", f"/documents/{d['id']}/restore", {}),
        ("GET", f"/documents/{d['id']}/versions", {}),
        ("DELETE", f"/documents/{d['id']}", {}),
    ]:
        r = await b.request(method, f"{API}{path}", headers=hb, **kw)
        assert r.status_code == 404, (method, path, r.status_code)
    assert (await get(a, ha, d["id"]))["status"] == "ready"


# ---------------------------------------------------------------- images

def png(width, height):
    body = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0) + b"\x00" * 4
    return body + b"\x00" * 64


def jpeg(width, height):
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, height, width, 3) + b"\x01\x11\x00\x02\x11\x01\x03\x11\x01"
    return b"\xff\xd8" + b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + sof + b"\xff\xd9"


async def test_images_are_refused_unless_enabled_by_configuration(client, auth):
    r = await upload(client, auth, "scan.png", png(100, 100), "image/png")
    assert r.status_code == 415 and r.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"


async def test_image_uploads_when_enabled_check_signature_and_pixel_count(client, auth, monkeypatch):
    monkeypatch.setattr(settings, "ALLOWED_UPLOAD_EXTENSIONS", "pdf,txt,md,csv,docx,png,jpg,jpeg")
    settings.__dict__.pop("allowed_extensions", None)
    monkeypatch.setattr(settings, "MAX_IMAGE_PIXELS", 1_000_000)
    try:
        ok = await upload(client, auth, "scan.png", png(800, 600), "image/png")
        assert ok.status_code == 201 and ok.json()["mime_type"] == "image/png"
        assert (await upload(client, auth, "photo.jpg", jpeg(1000, 700), "image/jpeg")).status_code == 201
        big = await upload(client, auth, "bomb.png", png(30000, 30000), "image/png")
        assert big.status_code == 413 and big.json()["error"]["code"] == "IMAGE_TOO_LARGE"
        bigj = await upload(client, auth, "bomb.jpg", jpeg(30000, 30000), "image/jpeg")
        assert bigj.status_code == 413
        fake = await upload(client, auth, "fake.png", b"GIF89a not a png at all", "image/png")
        assert fake.status_code == 415 and fake.json()["error"]["code"] == "FILE_CONTENT_MISMATCH"
        truncated = await upload(client, auth, "trunc.png", b"\x89PNG\r\n\x1a\n\x00\x00", "image/png")
        assert truncated.status_code == 415
    finally:
        settings.__dict__.pop("allowed_extensions", None)


# ---------------------------------------------------------------- job queue housekeeping

async def test_scrub_removes_legacy_file_copies_from_finished_jobs(client, auth, rag, storage):
    import base64

    await upload(client, auth, "a.txt", b"payload scrub")
    await drain(rag, storage)
    async with db_session.session_scope() as db:
        await db.execute(update(BackgroundJob).values(payload={"file_data": base64.b64encode(b"x" * 1000).decode(), "keep": "me"}))
        await db.commit()
        await cleanup_jobs.scrub_job_payloads(db)
        job = (await db.scalars(select(BackgroundJob))).one()
        await db.refresh(job)
    assert job.payload == {"keep": "me"}
