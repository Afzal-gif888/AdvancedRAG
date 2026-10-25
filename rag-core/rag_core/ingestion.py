"""Ingestion lifecycle: one active ingestion per document, run in the background, observable,
crash-safe. Parsing/chunking/embedding/indexing itself is pipeline.ingest (unchanged frozen RAG).

  claim      per-document lock: compare-and-set on rag_ingestions.attempt_id (works across
             processes/replicas). A request for a document whose attempt is running and alive joins
             that attempt instead of starting another one.
  run        background task with a heartbeat. Before committing, the attempt re-checks under a row
             lock that it still owns the document; otherwise it discards what it wrote.
  failure    every failure ends with this document's vectors/chunks removed and the attempt marked
             failed with a sanitized code + message (retryable or not).
  crash      a dead process stops heartbeating: after INGEST_STALE_SECONDS the attempt reports
             `stale`, and the next claim takes over (its writes overwrite the same deterministic ids,
             and any extra ids from the dead attempt are within the recorded vector_extent).
  delete     pipeline.delete_document removes the attempt row: the running attempt notices at its
             next checkpoint / heartbeat / commit check and discards its writes.
"""
import asyncio
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from sqlalchemy import select, update
from sqlalchemy.exc import DataError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from advancerag_common.observability import METRICS
from advancerag_common.service import ServiceError
from rag_core import db as rag_db
from rag_core import pipeline
from rag_core.db import IndexedDocument, IngestionAttempt, utcnow

log = logging.getLogger("advancerag.rag.ingestion")

RUNNING, SUCCEEDED, FAILED, CANCELLED = "running", "succeeded", "failed", "cancelled"
INTERRUPTED = "INGESTION_INTERRUPTED"
FAILED_CODE = "INGESTION_FAILED"

_tasks: Dict[uuid.UUID, asyncio.Task] = {}
# document_id -> attempt_id of attempts executing in THIS process. Such an attempt is alive no matter
# how old its heartbeat looks (e.g. right after the host resumed from sleep, before the heartbeat
# task has run again); other processes still rely on the heartbeat.
_local: Dict[uuid.UUID, uuid.UUID] = {}
_semaphore: Optional[asyncio.Semaphore] = None


def _slots(limit: int) -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(limit)
    return _semaphore


def _discard_spool(path: Optional[Path]) -> None:
    """Remove a spooled upload. Never raises: a leftover temp file must not mask the real outcome."""
    if path is not None:
        try:
            os.unlink(path)
        except OSError:
            pass


def _age(ts: Optional[datetime]) -> float:
    return (utcnow() - ts).total_seconds() if ts else float("inf")


def is_alive(row: IngestionAttempt, stale_after: float) -> bool:
    if row.status != RUNNING:
        return False
    return _local.get(row.document_id) == row.attempt_id or _age(row.heartbeat_at) < stale_after


def snapshot(row: IngestionAttempt, stale_after: float) -> dict:
    stale = row.status == RUNNING and not is_alive(row, stale_after)
    return {
        "document_id": str(row.document_id),
        "attempt_id": str(row.attempt_id),
        "status": "stale" if stale else row.status,
        "stage": row.stage,
        "chunks": row.chunk_count,
        "error_code": INTERRUPTED if stale else row.error_code,
        "error_message": "Indexing was interrupted." if stale else row.error_message,
        "retryable": True if stale else bool(row.retryable),
        "timings": row.timings or {},
        "extraction": row.extraction,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        "heartbeat_age_seconds": round(_age(row.heartbeat_at), 1),
    }


class _Control:
    """Hooks pipeline.ingest calls at each checkpoint, bound to one attempt."""

    def __init__(self, document_id: uuid.UUID, attempt_id: uuid.UUID):
        self.document_id, self.attempt_id = document_id, attempt_id
        self.stage_name, self.progress = "queued", {}
        self.cancelled = False

    def _owned(self):
        return (IngestionAttempt.document_id == self.document_id) & (IngestionAttempt.attempt_id == self.attempt_id)

    async def stage(self, name: str, **info) -> None:
        self.stage_name, self.progress = name, info
        if self.cancelled:
            raise pipeline.IngestionCancelled()

    async def still_owned(self) -> bool:
        """This attempt still owns the document (not deleted, superseded or finished elsewhere)."""
        if self.cancelled:
            return False
        async with rag_db.SessionLocal() as db:
            return await db.scalar(select(IngestionAttempt.attempt_id).where(self._owned())) is not None

    async def may_discard(self) -> bool:
        """After a failure: remove what this attempt wrote, unless a newer attempt now owns the
        document (it writes the same deterministic ids and its own cleanup covers the extent)."""
        async with rag_db.SessionLocal() as db:
            row = await db.get(IngestionAttempt, self.document_id)
            return row is None or row.attempt_id == self.attempt_id

    async def record_extent(self, extent: int) -> int:
        async with rag_db.SessionLocal() as db:
            row = await db.scalar(select(IngestionAttempt).where(self._owned()).with_for_update())
            if row is None:
                raise pipeline.IngestionCancelled()
            row.vector_extent = max(row.vector_extent, extent)
            row.stage, row.heartbeat_at = "writing", utcnow()
            effective = row.vector_extent
            await db.commit()
        return effective

    async def before_commit(self, db: AsyncSession, *, chunks: int, timings: dict, extraction: Optional[dict] = None) -> None:
        # Same transaction as the chunk rows: a concurrent delete (which locks/deletes this row)
        # either happens entirely before (we see no row -> cancel) or after our commit.
        row = await db.scalar(select(IngestionAttempt).where(self._owned()).with_for_update())
        if row is None or self.cancelled:
            raise pipeline.IngestionCancelled()
        row.status, row.stage, row.chunk_count = SUCCEEDED, "done", chunks
        row.timings, row.finished_at, row.heartbeat_at = dict(timings), utcnow(), utcnow()
        row.extraction = extraction
        row.error_code = row.error_message = None

    async def heartbeat(self, every: float) -> None:
        while True:
            await asyncio.sleep(every)
            try:
                async with rag_db.SessionLocal() as db:
                    result = await db.execute(
                        update(IngestionAttempt).where(self._owned(), IngestionAttempt.status == RUNNING)
                        .values(heartbeat_at=utcnow(), stage=self.stage_name[:40])
                    )
                    await db.commit()
                if result.rowcount == 0:
                    self.cancelled = True  # deleted, superseded or finished
                    return
            except Exception:
                log.warning("ingestion_heartbeat_failed", extra={"document_id": str(self.document_id)})


class IngestionManager:
    def __init__(self, svc: pipeline.Services):
        self.svc = svc
        self.cfg = svc.settings

    # ------------------------------------------------------------------ claim (lock)

    async def claim(self, *, document_id: uuid.UUID, user_id: uuid.UUID) -> Tuple[IngestionAttempt, bool]:
        """Returns (attempt row, started). started=False: an alive attempt already owns the document."""
        for _ in range(3):
            async with rag_db.SessionLocal() as db:
                row = await db.scalar(select(IngestionAttempt).where(IngestionAttempt.document_id == document_id).with_for_update())
                indexed = await db.get(IndexedDocument, document_id)
                if (row is not None and row.user_id != user_id) or (indexed is not None and indexed.user_id != user_id):
                    raise ServiceError("DOCUMENT_OWNERSHIP_MISMATCH", "Document belongs to another user.", 409)
                if row is not None and is_alive(row, self.cfg.INGEST_STALE_SECONDS):
                    return row, False
                now, new_id = utcnow(), uuid.uuid4()
                extent = max(row.vector_extent if row else 0, indexed.max_chunk_count if indexed else 0)
                fields = dict(attempt_id=new_id, status=RUNNING, stage="queued", chunk_count=0, vector_extent=extent,
                              error_code=None, error_message=None, retryable=False, timings=None,
                              heartbeat_at=now, started_at=now, finished_at=None)
                try:
                    if row is None:
                        db.add(IngestionAttempt(document_id=document_id, user_id=user_id, **fields))
                        await db.commit()
                    else:
                        # Compare-and-set: only one claimant can replace this exact attempt.
                        result = await db.execute(
                            update(IngestionAttempt)
                            .where(IngestionAttempt.document_id == document_id, IngestionAttempt.attempt_id == row.attempt_id)
                            .values(**fields)
                        )
                        if result.rowcount != 1:
                            await db.rollback()
                            continue
                        await db.commit()
                except IntegrityError:
                    await db.rollback()  # another claimant inserted first: re-read and join it
                    continue
                if row is not None and row.status == RUNNING:
                    log.warning("ingestion_stale_attempt_taken_over", extra={"document_id": str(document_id), "previous_attempt": str(row.attempt_id)})
                return await db.get(IngestionAttempt, document_id, populate_existing=True), True
        raise ServiceError("INGESTION_BUSY", "The document is being claimed by another request.", 409, retryable=True)

    # ------------------------------------------------------------------ run

    async def _finish_failed(self, document_id: uuid.UUID, user_id: uuid.UUID, attempt_id: uuid.UUID,
                             code: str, message: str, retryable: bool) -> None:
        async with rag_db.SessionLocal() as db:
            row = await db.scalar(select(IngestionAttempt).where(
                IngestionAttempt.document_id == document_id, IngestionAttempt.attempt_id == attempt_id).with_for_update())
            if row is None:
                return  # deleted meanwhile: delete_document already removed everything
            # No partial index may survive a failed attempt (also covers a previous version that
            # this attempt had started to overwrite).
            await pipeline.discard_index(db, self.svc, document_id=document_id, user_id=user_id, extent=row.vector_extent)
            row = await db.scalar(select(IngestionAttempt).where(
                IngestionAttempt.document_id == document_id, IngestionAttempt.attempt_id == attempt_id))
            if row is None:
                return
            row.status, row.error_code, row.error_message = FAILED, code[:64], message[:300]
            row.retryable, row.finished_at, row.vector_extent = retryable, utcnow(), 0
            await db.commit()

    async def _run(self, *, row: IngestionAttempt, filename: str, mime_type: str, data: Optional[bytes] = None,
                   path: Optional[Path] = None) -> dict:
        document_id, user_id, attempt_id = row.document_id, row.user_id, row.attempt_id
        control = _Control(document_id, attempt_id)
        started_at = time.perf_counter()
        _local[document_id] = attempt_id
        beat = asyncio.create_task(control.heartbeat(self.cfg.INGEST_HEARTBEAT_SECONDS))
        try:
            async with _slots(self.cfg.INGEST_CONCURRENCY):
                await control.stage("parsing")
                async with rag_db.SessionLocal() as db:
                    result = await pipeline.ingest(db, self.svc, document_id=document_id, user_id=user_id, filename=filename,
                                                   mime_type=mime_type, data=data, path=path, control=control)
            METRICS.inc("ingestions", outcome="succeeded", partial=str(bool((result.get("extraction") or {}).get("partial"))).lower())
            METRICS.observe_ms("ingestion_total", round((time.perf_counter() - started_at) * 1000, 2), outcome="succeeded")
            log.info("ingestion_succeeded", extra={"document_id": str(document_id), "attempt_id": str(attempt_id),
                                                   "chunks": result["chunks"], **result.get("timings", {})})
            return result
        except pipeline.IngestionCancelled:
            log.info("ingestion_cancelled", extra={"document_id": str(document_id), "attempt_id": str(attempt_id)})
            raise ServiceError("INGESTION_CANCELLED", "Indexing was cancelled.", 409)
        except asyncio.CancelledError:
            # Process shutting down: leave nothing half-written; the backend will retry.
            await asyncio.shield(self._finish_failed(document_id, user_id, attempt_id, INTERRUPTED, "Indexing was interrupted.", True))
            raise
        except ServiceError as exc:
            METRICS.inc("ingestions", outcome="failed", code=exc.code)
            await self._finish_failed(document_id, user_id, attempt_id, exc.code, exc.message, exc.retryable)
            raise
        except (IntegrityError, DataError):
            log.exception("ingestion_failed_permanently", extra={"document_id": str(document_id)})
            await self._finish_failed(document_id, user_id, attempt_id, FAILED_CODE, "Document ingestion failed.", False)
            raise ServiceError(FAILED_CODE, "Document ingestion failed.", 500)
        except Exception:
            log.exception("ingestion_failed", extra={"document_id": str(document_id)})
            await self._finish_failed(document_id, user_id, attempt_id, FAILED_CODE, "Document ingestion failed.", True)
            raise ServiceError(FAILED_CODE, "Document ingestion failed.", 503, retryable=True)
        finally:
            beat.cancel()
            if _local.get(document_id) == attempt_id:
                _local.pop(document_id, None)
            _discard_spool(path)

    # ------------------------------------------------------------------ public API

    async def start(self, *, document_id: uuid.UUID, user_id: uuid.UUID, filename: str, mime_type: str, data: Optional[bytes] = None,
                    path: Optional[Path] = None) -> dict:
        """Asynchronous: claim and return immediately; the work continues in the background.
        ``path`` is a spooled copy of the upload that this call takes ownership of (it is deleted when the
        attempt ends, or at once if the document is already being indexed)."""
        try:
            row, started = await self.claim(document_id=document_id, user_id=user_id)
        except BaseException:
            _discard_spool(path)
            raise
        if not started:
            _discard_spool(path)
        if started:
            task = asyncio.create_task(self._run(row=row, filename=filename, mime_type=mime_type, data=data, path=path))
            _tasks[document_id] = task
            task.add_done_callback(lambda t, d=document_id: (_tasks.pop(d, None) if _tasks.get(d) is t else None,
                                                             t.cancelled() or t.exception()))
        return {**snapshot(row, self.cfg.INGEST_STALE_SECONDS), "started": started}

    async def run_sync(self, *, document_id: uuid.UUID, user_id: uuid.UUID, filename: str, mime_type: str, data: Optional[bytes] = None,
                       path: Optional[Path] = None) -> dict:
        """Synchronous variant (POST /ingest): same lock/cleanup guarantees, waits for the result.
        A client disconnect does not cancel the ingestion (shielded)."""
        try:
            row, started = await self.claim(document_id=document_id, user_id=user_id)
        except BaseException:
            _discard_spool(path)
            raise
        if not started:
            _discard_spool(path)
            raise ServiceError("INGESTION_IN_PROGRESS", "This document is already being indexed.", 409, retryable=True)
        task = asyncio.create_task(self._run(row=row, filename=filename, mime_type=mime_type, data=data, path=path))
        _tasks[document_id] = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done() and _tasks.get(document_id) is task:
                _tasks.pop(document_id, None)

    async def status(self, *, document_id: uuid.UUID, user_id: uuid.UUID) -> Optional[dict]:
        async with rag_db.SessionLocal() as db:
            row = await db.scalar(select(IngestionAttempt).where(IngestionAttempt.document_id == document_id, IngestionAttempt.user_id == user_id))
            return snapshot(row, self.cfg.INGEST_STALE_SECONDS) if row else None


async def shutdown(timeout: float = 20.0) -> None:
    """Cancel in-flight ingestions on shutdown; each one cleans up and records INGESTION_INTERRUPTED."""
    tasks = list(_tasks.values())
    for t in tasks:
        t.cancel()
    if tasks:
        await asyncio.wait(tasks, timeout=timeout)
