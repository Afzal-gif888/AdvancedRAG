"""Document ingestion/deletion jobs. All parsing/chunking/embedding/indexing is RAG Core's job;
these handlers only move bytes to RAG Core and track state.

Ingestion is asynchronous end to end: the job hands the file to RAG Core (which returns at once and
indexes in the background under a per-document lock), then polls the attempt's status with short
HTTP calls, heartbeating the job on every poll. No HTTP timeout bounds the ingestion itself; a long
document simply stays `processing` while RAG Core is alive and working on it.
"""
import asyncio
import logging
import time
from typing import Optional

from advancerag_common import wake
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import BackgroundJob, Document
from app.db.models import document as doc_status
from app.db.models import job as job_status
from app.integrations.rag_core import RAGCoreClient, RAGCoreError
from app.integrations.storage import Storage
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


async def _wait_for_ingestion(db: AsyncSession, job: BackgroundJob, rag: RAGCoreClient, doc: Document, status: dict) -> None:
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
            return
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
        if not await storage.exists(doc.storage_key):
            raise RAGCoreError("FILE_MISSING", retryable=False)
        content = await storage.read_bytes(doc.storage_key)
        status = await rag.start_ingestion(
            document_id=doc.id, user_id=doc.user_id, filename=doc.filename, mime_type=doc.mime_type, content=content
        )
    await _wait_for_ingestion(db, job, rag, doc, status)

    # The user may have deleted the document just as indexing finished.
    if (await _current_status(db, doc.id)) in (None, doc_status.DELETED):
        await _discard_quietly(rag, doc)
        raise JobCancelled("document_deleted")
    await document_service.mark_status(db, doc.id, doc_status.READY)
    job.progress = "indexed"


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
        await storage.delete(key)
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
