"""Document ingestion/deletion jobs. All parsing/chunking/embedding/indexing is RAG Core's job;
these handlers only move bytes to RAG Core and track state.

Ingestion is asynchronous end to end: the job hands the file to RAG Core (which returns at once and
indexes in the background under a per-document lock), then polls the attempt's status with short
HTTP calls, heartbeating the job on every poll. No HTTP timeout bounds the ingestion itself; a long
document simply stays `processing` while RAG Core is alive and working on it.
"""
import asyncio
import base64
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from advancerag_common import wake
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import BackgroundJob, Document
from app.db.models import document as doc_status
from app.core.time import utcnow
from app.db.models import job as job_status
from app.integrations.rag_core import RAGCoreClient, RAGCoreError
from app.integrations.storage import Storage, file_sha256, get_storage_for, local_copy
from app.services import document_service, job_service

logger = logging.getLogger("advancerag.jobs")

NO_DATA_KEPT = "No indexed data was retained."
SAFE_PROCESSING_ERRORS = {
    "RAG_SERVICE_UNAVAILABLE": f"The document processing service was unavailable. {NO_DATA_KEPT} Please try again.",
    "FILE_MISSING": "The uploaded file could not be found. Please upload it again.",
    "UNSUPPORTED_CONTENT": "This file type cannot be searched yet.",
    "NO_EXTRACTABLE_TEXT": "No text could be extracted (scanned or image-only documents are not supported yet).",
    "PARSE_FAILED": "The file appears to be damaged or password-protected.",
    "DOCUMENT_TOO_LARGE": "This document has too much text to index.",
    "EMBEDDING_REJECTED": "This document could not be indexed.",
    "INGESTION_TIMEOUT": f"Indexing took too long and was stopped. {NO_DATA_KEPT}",
    "INGESTION_INTERRUPTED": f"Indexing was interrupted. {NO_DATA_KEPT} Please try again.",
    "WORKER_CRASHED": f"Indexing was interrupted. {NO_DATA_KEPT} Please try again.",
    "FILE_CORRUPT": "The stored file did not match what was uploaded. Please upload it again.",
    "EMBEDDING_MODEL_MISMATCH": "This document was indexed with a different search model and must be re-indexed.",
}
DEFAULT_PROCESSING_ERROR = f"Document ingestion failed. {NO_DATA_KEPT}"

# Payload marker for the RAG-only cleanup queued after an ingestion finally fails.
CLEANUP_FAILED_INGEST = "cleanup_failed_ingest"
# While waiting for a sleeping RAG Core to wake, renew the job's lease this often (well inside JOB_STALE_AFTER_SECONDS).
WAKE_HEARTBEAT_SECONDS = 30.0


class JobCancelled(Exception):
    """The job's work became moot (e.g. the document was deleted). Recorded as `cancelled`."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def processing_error_message(code: str) -> str:
    return SAFE_PROCESSING_ERRORS.get(code, DEFAULT_PROCESSING_ERROR)


async def _current_status(db: AsyncSession, document_id) -> Optional[str]:
    return await db.scalar(select(Document.status).where(Document.id == document_id).execution_options(populate_existing=True))


async def _heartbeat(db: AsyncSession, job: BackgroundJob, progress: str) -> None:
    # Renews the lease; requeue_stale_jobs only recovers jobs whose heartbeat stopped. Raises
    # JobSuperseded if the job was recovered meanwhile (this run must then stop quietly).
    await job_service.heartbeat(db, job.id, job.attempts, progress)


async def _ensure_rag_awake(db: AsyncSession, job: BackgroundJob) -> None:
    """Wake RAG Core if it is asleep (RAG_WAKE_SECONDS > 0, e.g. Render free) before talking to it,
    heartbeating the job meanwhile so the wait never looks like a crashed worker. Gives up after the
    budget with a retryable error; the job's bounded attempts and backoff take it from there."""
    if settings.RAG_WAKE_SECONDS <= 0:
        return
    waking = asyncio.create_task(wake.ensure_awake(settings.RAG_CORE_URL, budget=settings.RAG_WAKE_SECONDS,
                                                   relay=settings.WAKE_RELAY_URL))
    try:
        while not (await asyncio.wait({waking}, timeout=WAKE_HEARTBEAT_SECONDS))[0]:
            await _heartbeat(db, job, "waking_rag")
        if not waking.result():
            raise RAGCoreError("RAG_SERVICE_UNAVAILABLE", retryable=True)
    finally:
        if not waking.done():
            waking.cancel()


async def _discard_quietly(rag: RAGCoreClient, doc: Document) -> None:
    """Stop RAG Core's attempt (if any) and remove everything it wrote. Best effort."""
    try:
        await rag.delete_document(document_id=doc.id, user_id=doc.user_id)
    except RAGCoreError:
        logger.warning("ingestion_discard_failed", extra={"document_id": str(doc.id)})


async def _wait_for_ingestion(db: AsyncSession, job: BackgroundJob, rag: RAGCoreClient, doc: Document, status: dict) -> dict:
    # The ceiling counts active polling time only: if the host sleeps, RAG Core is frozen too, so
    # wall-clock (or Windows' monotonic) time spent asleep must not turn into a "timeout".
    active, last = 0.0, time.monotonic()
    max_step = settings.INGEST_POLL_INTERVAL_SECONDS * 3 + settings.RAG_CORE_TIMEOUT_SECONDS
    errors_since: Optional[float] = None
    stale_since: Optional[float] = None
    while True:
        now = time.monotonic()
        active += min(now - last, max_step)
        last = now
        state = status.get("status")
        if state == "succeeded":
            return status
        if state == "failed":
            # RAG Core already removed everything this attempt wrote.
            raise RAGCoreError(status.get("error_code") or "INGESTION_FAILED", retryable=bool(status.get("retryable")))
        if state == "stale":
            # Confirmed over two polls: right after a host resumes, a live attempt can briefly look
            # stale until its heartbeat catches up.
            stale_since = stale_since or now
            if now - stale_since >= settings.INGEST_STALE_CONFIRM_SECONDS:
                # RAG Core died mid-ingestion: clear its partial writes, then retry from scratch.
                await _discard_quietly(rag, doc)
                raise RAGCoreError("INGESTION_INTERRUPTED", retryable=True)
        else:
            stale_since = None
        if active > settings.INGEST_MAX_SECONDS:
            await _discard_quietly(rag, doc)
            raise RAGCoreError("INGESTION_TIMEOUT", retryable=False)

        await asyncio.sleep(settings.INGEST_POLL_INTERVAL_SECONDS)
        if (await _current_status(db, doc.id)) in (None, doc_status.DELETED):
            await _discard_quietly(rag, doc)  # cancels the running attempt and removes its writes
            raise JobCancelled("document_deleted")
        await _heartbeat(db, job, f"rag:{status.get('stage') or state}")
        try:
            fresh = await rag.ingestion_status(document_id=doc.id, user_id=doc.user_id)
        except RAGCoreError as exc:
            if not exc.retryable:
                raise
            errors_since = errors_since or time.monotonic()
            if time.monotonic() - errors_since > settings.INGEST_STATUS_ERROR_BUDGET_SECONDS:
                raise
            continue
        errors_since = None
        if fresh is None:  # the attempt record vanished without the document being deleted
            raise RAGCoreError("INGESTION_INTERRUPTED", retryable=True)
        status = fresh


def _legacy_payload_to_file(job: BackgroundJob) -> Optional[Path]:
    """Jobs queued by the previous version carried the file as base64 in their payload. They are still
    honoured (once), written to a temporary file; new jobs never carry file content."""
    data = (job.payload or {}).get("file_data")
    if not data:
        return None
    fd, name = tempfile.mkstemp(prefix="legacy-", dir=settings.TEMP_DIR or None)
    with os.fdopen(fd, "wb") as out:
        out.write(base64.b64decode(data))
    return Path(name)


async def _send_to_rag(db: AsyncSession, job: BackgroundJob, rag: RAGCoreClient, doc: Document) -> dict:
    """Hand the original file to RAG Core without loading it into memory: from storage (by the backend the
    document was stored with) to a local path, integrity-checked, then streamed."""
    legacy = _legacy_payload_to_file(job)
    if legacy is not None:
        try:
            return await rag.start_ingestion(document_id=doc.id, user_id=doc.user_id, filename=doc.filename, mime_type=doc.mime_type, content=legacy)
        finally:
            legacy.unlink(missing_ok=True)
    storage = get_storage_for(doc.storage_backend)
    try:
        exists = await storage.exists(doc.storage_key)
    except Exception:
        logger.exception("ingest_storage_error", extra={"document_id": str(doc.id), "storage_backend": doc.storage_backend})
        raise RAGCoreError("STORAGE_UNAVAILABLE", retryable=True)
    if not exists:
        # The one failure that retrying cannot fix. The log says why it can happen: the object is not in the
        # backend this document was stored with (typically `local` on an ephemeral disk after a restart).
        logger.error("ingest_file_missing", extra={"document_id": str(doc.id), "storage_backend": doc.storage_backend,
                                                   "attempt": job.attempts, "age_s": round((utcnow() - doc.created_at).total_seconds())})
        raise RAGCoreError("FILE_MISSING", retryable=False)
    try:
        async with local_copy(storage, doc.storage_key) as path:
            digest = await asyncio.to_thread(file_sha256, path)
            if digest != doc.checksum_sha256 or path.stat().st_size != doc.size_bytes:
                logger.error("ingest_file_corrupt", extra={"document_id": str(doc.id), "storage_backend": doc.storage_backend})
                raise RAGCoreError("FILE_CORRUPT", retryable=False)
            return await rag.start_ingestion(document_id=doc.id, user_id=doc.user_id, filename=doc.filename, mime_type=doc.mime_type, content=path)
    except FileNotFoundError:
        raise RAGCoreError("FILE_MISSING", retryable=False)
    except RAGCoreError:
        raise
    except Exception as exc:
        if isinstance(exc, (OSError, ConnectionError)):
            logger.exception("ingest_storage_error", extra={"document_id": str(doc.id), "storage_backend": doc.storage_backend})
            raise RAGCoreError("STORAGE_UNAVAILABLE", retryable=True) from exc
        raise


async def ingest_document(db: AsyncSession, job: BackgroundJob, rag: RAGCoreClient, storage: Storage) -> None:
    doc = await db.scalar(select(Document).where(Document.id == job.resource_id))
    if doc is None or doc.status == doc_status.DELETED:
        raise JobCancelled("document_deleted")  # deleted while queued: nothing to do
    await document_service.mark_status(db, doc.id, doc_status.PROCESSING)
    await _heartbeat(db, job, "starting")
    await _ensure_rag_awake(db, job)

    # Idempotent: a retried/duplicate job joins the attempt that is already running, or just
    # records the result of one that already finished, instead of starting a second ingestion.
    status = await rag.ingestion_status(document_id=doc.id, user_id=doc.user_id)
    if status is None or status.get("status") not in ("running", "succeeded"):
        await _heartbeat(db, job, "sending")
        status = await _send_to_rag(db, job, rag, doc)
    final = await _wait_for_ingestion(db, job, rag, doc, status)

    # The user may have deleted the document just as indexing finished.
    if (await _current_status(db, doc.id)) in (None, doc_status.DELETED):
        await _discard_quietly(rag, doc)
        raise JobCancelled("document_deleted")
    # Ready + (if this is a new version) the old version superseded: one transaction, committed with the job outcome.
    await document_service.finalize_ready(db, doc.id, (final or {}).get("extraction"))
    job.progress = "indexed"
    if job.payload and "file_data" in job.payload:  # legacy jobs: do not keep a copy of the file in the job table
        job.payload = {k: v for k, v in job.payload.items() if k != "file_data"}


async def delete_document(db: AsyncSession, job: BackgroundJob, rag: RAGCoreClient, storage: Storage) -> None:
    payload = job.payload or {}
    if payload.get("purpose") == CLEANUP_FAILED_INGEST:
        # Only while the document is still failed: a retry (reprocess) owns the index from then on.
        if (await _current_status(db, job.resource_id)) != doc_status.FAILED:
            return
        await _ensure_rag_awake(db, job)
        await rag.delete_document(document_id=job.resource_id, user_id=job.user_id)
        return
    await _ensure_rag_awake(db, job)
    await rag.delete_document(document_id=job.resource_id, user_id=job.user_id)
    key = payload.get("storage_key")
    if key:
        await get_storage_for(payload.get("storage_backend")).delete(key)
    # No copy of the document may outlive it: queued/finished ingest jobs of older versions of this code carried
    # the file as base64 in their payload.
    await db.execute(
        update(BackgroundJob)
        .where(BackgroundJob.resource_id == job.resource_id, BackgroundJob.job_type == job_status.DOCUMENT_INGEST, BackgroundJob.id != job.id)
        .values(payload={})
    )
    await db.execute(delete(Document).where(Document.id == job.resource_id, Document.status == doc_status.DELETED))


async def on_final_failure(db: AsyncSession, job: BackgroundJob, code: str) -> None:
    if job.job_type == job_status.DOCUMENT_INGEST:
        await document_service.mark_status(db, job.resource_id, doc_status.FAILED, processing_error_message(code))
        # Guarantee no index data survives a failed document even when RAG Core could not be
        # reached to clean up (bounded retries; skipped if the user reprocesses meanwhile).
        job_service.enqueue(db, job.user_id, job_status.DOCUMENT_DELETE, resource_id=job.resource_id,
                            payload={"purpose": CLEANUP_FAILED_INGEST}, max_attempts=settings.DELETE_JOB_MAX_ATTEMPTS)
    elif job.job_type == job_status.DOCUMENT_DELETE:
        # The document stays hidden (status=deleted); the reconciler re-drives deletion with a
        # bounded budget so the file and vectors are not orphaned.
        logger.error("document_delete_failed", extra={"document_id": str(job.resource_id), "code": code})
