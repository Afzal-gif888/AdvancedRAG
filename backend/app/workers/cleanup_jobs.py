"""Periodic recovery and housekeeping (runs at worker start and every RECONCILE_INTERVAL_SECONDS).

- requeue_stale_jobs:     jobs left `running` by a crashed worker -> retry, or fail once attempts are used up
- reconcile_documents:    documents whose state no longer has a live job behind it -> re-drive or fail
- recover_generations:    assistant messages stuck pending/streaming -> failed (INTERRUPTED), retryable
- purge_expired_auth_records: old sessions/tokens/finished jobs

Every path is bounded: no job or document can be retried forever.
"""
import logging
from datetime import timedelta

from sqlalchemy import and_, delete, exists, func, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.time import utcnow
from app.db.models import AuthToken, BackgroundJob, Document, UserSession
from app.db.models import document as doc_status
from app.db.models import job as job_status
from app.services import document_service, job_service, message_service

logger = logging.getLogger("advancerag.jobs")

RETENTION = timedelta(days=30)
# How many separate delete jobs a deleted document may get before it needs manual attention.
MAX_DELETE_REDRIVES = 5


async def requeue_stale_jobs(db: AsyncSession) -> int:
    from app.workers import document_jobs

    cutoff = utcnow() - timedelta(seconds=settings.JOB_STALE_AFTER_SECONDS)
    stale = (
        await db.scalars(select(BackgroundJob).where(BackgroundJob.status == job_status.RUNNING, BackgroundJob.locked_at < cutoff))
    ).all()
    recovered = 0
    for job in stale:
        final = job.attempts >= job.max_attempts
        values = ({"status": job_status.FAILED, "error": "WORKER_CRASHED", "completed_at": utcnow()} if final
                  else {"status": job_status.PENDING, "run_after": utcnow() + job_service.retry_delay(job.attempts)})
        # Compare-and-set: a run that finished (or heartbeated) since the SELECT is left alone.
        result = await db.execute(
            update(BackgroundJob)
            .where(BackgroundJob.id == job.id, BackgroundJob.status == job_status.RUNNING, BackgroundJob.locked_at < cutoff)
            .values(locked_at=None, **values)
            .execution_options(synchronize_session=False)
        )
        if result.rowcount == 1:
            recovered += 1
            if final:
                await document_jobs.on_final_failure(db, job, "WORKER_CRASHED")
    await db.commit()
    if recovered:
        logger.warning("stale_jobs_recovered", extra={"count": recovered})
    return recovered


def _live_job(job_type: str):
    return exists().where(
        BackgroundJob.resource_id == Document.id,
        BackgroundJob.job_type == job_type,
        BackgroundJob.status.in_([job_status.PENDING, job_status.RUNNING]),
    )


async def reconcile_documents(db: AsyncSession) -> dict:
    grace = utcnow() - timedelta(seconds=settings.RECONCILE_INTERVAL_SECONDS)
    stats = {"failed": 0, "requeued_ingest": 0, "redriven_delete": 0, "abandoned_delete": 0}

    # 1) uploaded/processing with no live ingest job: the job was lost -> requeue once, else fail.
    orphans = (
        await db.scalars(
            select(Document).where(
                Document.status.in_([doc_status.UPLOADED, doc_status.PROCESSING]),
                Document.updated_at < grace,
                ~_live_job(job_status.DOCUMENT_INGEST),
            )
        )
    ).all()
    for doc in orphans:
        attempts = await db.scalar(
            select(func.count()).select_from(BackgroundJob).where(BackgroundJob.resource_id == doc.id, BackgroundJob.job_type == job_status.DOCUMENT_INGEST)
        )
        if attempts:
            doc.status = doc_status.FAILED
            doc.processing_error = "Processing did not complete. Please try again."
            stats["failed"] += 1
        else:
            job_service.enqueue(db, doc.user_id, job_status.DOCUMENT_INGEST, resource_id=doc.id)
            stats["requeued_ingest"] += 1

    # 2) deleted documents whose delete job gave up: re-drive with a bounded budget.
    deleted = (
        await db.scalars(
            select(Document).where(Document.status == doc_status.DELETED, Document.updated_at < grace, ~_live_job(job_status.DOCUMENT_DELETE))
        )
    ).all()
    for doc in deleted:
        tries = await db.scalar(
            select(func.count()).select_from(BackgroundJob).where(BackgroundJob.resource_id == doc.id, BackgroundJob.job_type == job_status.DOCUMENT_DELETE)
        )
        if tries >= MAX_DELETE_REDRIVES:
            stats["abandoned_delete"] += 1
            logger.error("document_delete_abandoned", extra={"document_id": str(doc.id)})
            continue
        job_service.enqueue(db, doc.user_id, job_status.DOCUMENT_DELETE, resource_id=doc.id,
                            payload={"storage_key": doc.storage_key, "storage_backend": doc.storage_backend},
                            max_attempts=settings.DELETE_JOB_MAX_ATTEMPTS)
        stats["redriven_delete"] += 1

    await db.commit()
    if any(stats.values()):
        logger.warning("documents_reconciled", extra=stats)
        job_service.notify_worker()
    return stats


async def recover_generations(db: AsyncSession) -> int:
    count = await message_service.mark_stale_generations(db, settings.STALE_GENERATION_SECONDS)
    if count:
        logger.warning("interrupted_generations_recovered", extra={"count": count})
    return count


async def purge_expired_auth_records(db: AsyncSession) -> None:
    cutoff = utcnow() - RETENTION
    await db.execute(delete(UserSession).where(or_(UserSession.expires_at < cutoff, UserSession.revoked_at < cutoff)))
    await db.execute(delete(AuthToken).where(AuthToken.expires_at < cutoff))
    # Keep failed delete jobs (they count towards MAX_DELETE_REDRIVES) while their document exists.
    await db.execute(
        delete(BackgroundJob).where(
            BackgroundJob.status.in_([job_status.SUCCEEDED, job_status.FAILED, job_status.CANCELLED]),
            BackgroundJob.updated_at < cutoff,
            ~and_(
                BackgroundJob.job_type == job_status.DOCUMENT_DELETE,
                exists().where(Document.id == BackgroundJob.resource_id),
            ),
        )
    )
    await db.commit()


async def purge_superseded_versions(db: AsyncSession) -> int:
    """Superseded versions are kept for SUPERSEDED_RETENTION_DAYS (so a replacement can be rolled back), then
    deleted like any other document: vectors, file and row, through the normal delete job."""
    cutoff = utcnow() - timedelta(days=settings.SUPERSEDED_RETENTION_DAYS)
    old = (await db.scalars(select(Document).where(Document.status == doc_status.SUPERSEDED, Document.superseded_at < cutoff))).all()
    if old:
        await document_service._soft_delete(db, old)
        await db.commit()
        job_service.notify_worker()
        logger.info("superseded_versions_purged", extra={"count": len(old)})
    return len(old)


async def scrub_job_payloads(db: AsyncSession) -> None:
    """Jobs written by earlier versions carried the whole file as base64 in `payload`. Once a job is finished
    nothing needs it: drop that copy now instead of keeping it for the 30-day job retention."""
    finished = "('succeeded','failed','cancelled')"
    if db.bind is not None and db.bind.dialect.name == "postgresql":
        await db.execute(text(f"UPDATE background_jobs SET payload = payload - 'file_data' WHERE status IN {finished} AND payload ? 'file_data'"))
    else:
        await db.execute(text(f"UPDATE background_jobs SET payload = json_remove(payload, '$.file_data') WHERE status IN {finished} AND json_extract(payload, '$.file_data') IS NOT NULL"))
    await db.commit()


async def run_all(db: AsyncSession) -> None:
    await requeue_stale_jobs(db)
    await reconcile_documents(db)
    await recover_generations(db)
    await purge_superseded_versions(db)
    await scrub_job_payloads(db)
    await purge_expired_auth_records(db)
