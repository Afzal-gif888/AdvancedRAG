"""Durable job queue on the background_jobs table.

Claiming uses SELECT ... FOR UPDATE SKIP LOCKED on PostgreSQL, so any number of API
processes can run workers safely. Swap for a dedicated queue (arq/Celery) without
touching callers: they only use enqueue()/notify_worker().
"""
import asyncio
import uuid
from datetime import timedelta
from typing import Optional

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.time import utcnow
from advancerag_common.observability import METRICS
from app.db.models import BackgroundJob
from app.db.models import job as job_status

_wakeup: Optional[asyncio.Event] = None


def wakeup_event() -> asyncio.Event:
    global _wakeup
    if _wakeup is None:
        _wakeup = asyncio.Event()
    return _wakeup


def notify_worker() -> None:
    if _wakeup is not None:
        _wakeup.set()


def enqueue(
    db: AsyncSession,
    user_id: uuid.UUID,
    job_type: str,
    *,
    resource_id: Optional[uuid.UUID] = None,
    payload: Optional[dict] = None,
    max_attempts: Optional[int] = None,
) -> BackgroundJob:
    """Adds the job to the caller's transaction — it becomes visible only when they commit."""
    job = BackgroundJob(
        user_id=user_id,
        job_type=job_type,
        resource_id=resource_id,
        payload=payload or {},
        status=job_status.PENDING,
        max_attempts=max_attempts or settings.JOB_MAX_ATTEMPTS,
        run_after=utcnow(),
    )
    db.add(job)
    return job


async def claim_next(db: AsyncSession) -> Optional[BackgroundJob]:
    now = utcnow()
    for _ in range(5):
        job_id = await db.scalar(
            select(BackgroundJob.id)
            .where(BackgroundJob.status == job_status.PENDING, BackgroundJob.run_after <= now)
            .order_by(BackgroundJob.run_after, BackgroundJob.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        if job_id is None:
            return None
        # Compare-and-set: exactly one worker wins, even where SKIP LOCKED is unavailable (SQLite).
        result = await db.execute(
            update(BackgroundJob)
            .where(BackgroundJob.id == job_id, BackgroundJob.status == job_status.PENDING)
            .values(status=job_status.RUNNING, locked_at=now, attempts=BackgroundJob.attempts + 1)
        )
        await db.commit()
        if result.rowcount == 1:
            return await db.get(BackgroundJob, job_id, populate_existing=True)
    return None


class JobSuperseded(Exception):
    """This worker no longer owns the job (it was recovered as stale and handed to another run).
    The current holder must stop without touching the job or its resource."""


def _owned(job_id: uuid.UUID, attempt: int):
    return (BackgroundJob.id == job_id) & (BackgroundJob.status == job_status.RUNNING) & (BackgroundJob.attempts == attempt)


async def heartbeat(db: AsyncSession, job_id: uuid.UUID, attempt: int, progress: Optional[str] = None) -> None:
    """Refresh the job's lease (locked_at). Raises JobSuperseded if the job was taken away."""
    values = {"locked_at": utcnow()}
    if progress is not None:
        values["progress"] = progress[:100]
    result = await db.execute(update(BackgroundJob).where(_owned(job_id, attempt)).values(**values))
    await db.commit()
    if result.rowcount != 1:
        raise JobSuperseded()


async def finish(db: AsyncSession, job_id: uuid.UUID, attempt: int, **values) -> bool:
    """Compare-and-set the job's outcome; only the run that still owns it may record one.
    Returns False (nothing written) when the job was recovered by someone else meanwhile."""
    result = await db.execute(update(BackgroundJob).where(_owned(job_id, attempt)).values(**values))
    return result.rowcount == 1


async def queue_stats(db: AsyncSession) -> dict:
    """Depth and age of the queue, per job type: pending/running counts, the age of the oldest pending job
    (a stalled worker shows up here first) and running jobs whose heartbeat stopped (stuck)."""
    now = utcnow()
    stats: dict = {}
    rows = (await db.execute(select(BackgroundJob.job_type, BackgroundJob.status, func.count(), func.min(BackgroundJob.run_after))
                             .where(BackgroundJob.status.in_((job_status.PENDING, job_status.RUNNING)))
                             .group_by(BackgroundJob.job_type, BackgroundJob.status))).all()
    for job_type, status, count, oldest in rows:
        entry = stats.setdefault(job_type, {"pending": 0, "running": 0, "oldest_pending_age_s": 0.0, "stuck": 0})
        entry[status] = count
        if status == job_status.PENDING and oldest is not None:
            entry["oldest_pending_age_s"] = max(0.0, (now - oldest).total_seconds())
    cutoff = now - timedelta(seconds=settings.JOB_STALE_AFTER_SECONDS)
    stuck = (await db.execute(select(BackgroundJob.job_type, func.count())
                              .where(BackgroundJob.status == job_status.RUNNING, BackgroundJob.locked_at < cutoff)
                              .group_by(BackgroundJob.job_type))).all()
    for job_type, count in stuck:
        stats.setdefault(job_type, {"pending": 0, "running": 0, "oldest_pending_age_s": 0.0, "stuck": 0})["stuck"] = count
    return stats


async def publish_queue_metrics(db: AsyncSession) -> dict:
    stats = await queue_stats(db)
    for job_type in {"document_ingest", "document_delete", *stats}:
        s = stats.get(job_type, {"pending": 0, "running": 0, "oldest_pending_age_s": 0.0, "stuck": 0})
        METRICS.set_gauge("queue_jobs", s["pending"], job_type=job_type, state="pending")
        METRICS.set_gauge("queue_jobs", s["running"], job_type=job_type, state="running")
        METRICS.set_gauge("queue_oldest_pending_age_seconds", round(s["oldest_pending_age_s"], 1), job_type=job_type)
        METRICS.set_gauge("queue_stuck_jobs", s["stuck"], job_type=job_type)
    return stats


async def has_active_jobs(db: AsyncSession) -> bool:
    """True while any job is queued (including waiting for a retry) or running."""
    found = await db.scalar(
        select(BackgroundJob.id).where(BackgroundJob.status.in_((job_status.PENDING, job_status.RUNNING))).limit(1)
    )
    return found is not None


def retry_delay(attempt: int) -> timedelta:
    return timedelta(seconds=min(30 * (2 ** (attempt - 1)), 15 * 60))
