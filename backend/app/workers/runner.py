"""In-process async worker. Started from the app lifespan when WORKER_ENABLED=true;
can also run standalone: `python -m app.workers.runner`."""
import asyncio
import logging
import time
from typing import Optional

import httpx
from advancerag_common import wake
from advancerag_common.observability import METRICS, new_trace_id, trace_id_var

from app.core.config import settings
from app.core.context import request_id_var
from app.core.time import utcnow
from app.db.models import job as job_status
from app.db.session import session_scope
from app.integrations.rag_core import RAGCoreClient, RAGCoreError
from app.integrations.storage import Storage, get_storage
from app.services import job_service
from app.workers import cleanup_jobs, document_jobs

logger = logging.getLogger("advancerag.worker")

HANDLERS = {
    job_status.DOCUMENT_INGEST: document_jobs.ingest_document,
    job_status.DOCUMENT_DELETE: document_jobs.delete_document,
}


async def process_one(rag: Optional[RAGCoreClient] = None, storage: Optional[Storage] = None) -> bool:
    """Claim and run a single job. Returns False when the queue is empty."""
    rag = rag or RAGCoreClient()
    storage = storage or get_storage()
    async with session_scope() as db:
        job = await job_service.claim_next(db)
        if job is None:
            return False
        token = request_id_var.set(f"job-{job.id.hex[:16]}")
        trace_token = trace_id_var.set(new_trace_id())  # each job is its own trace; its calls to RAG Core carry it
        started = time.perf_counter()
        outcome = "error"
        # Every outcome is a compare-and-set on (id, running, attempt): if the stale-job sweep
        # recovered this job meanwhile (e.g. the machine slept past the lease), this run records
        # nothing and the recovered run takes over. No lost updates in either direction.
        job_id, attempt, max_attempts, job_type = job.id, job.attempts, job.max_attempts, job.job_type
        resource_id, error_code = job.resource_id, None
        try:
            handler = HANDLERS.get(job.job_type)
            if handler is None:
                raise RAGCoreError("UNKNOWN_JOB_TYPE", retryable=False)
            await handler(db, job, rag, storage)
            owned = await job_service.finish(db, job_id, attempt, status=job_status.SUCCEEDED, error=None,
                                             progress=job.progress, completed_at=utcnow(), locked_at=None)
            await db.commit()
            outcome = "succeeded" if owned else "superseded"
        except job_service.JobSuperseded:
            await db.rollback()
            outcome = "superseded"
        except document_jobs.JobCancelled as exc:
            await db.rollback()
            owned = await job_service.finish(db, job_id, attempt, status=job_status.CANCELLED, error=None,
                                             progress=exc.reason, completed_at=utcnow(), locked_at=None)
            await db.commit()
            outcome = "cancelled" if owned else "superseded"
        except Exception as exc:
            await db.rollback()
            # Rollback expires loaded objects; reload explicitly (lazy loads are not allowed under asyncio).
            await db.refresh(job)
            code = exc.code if isinstance(exc, RAGCoreError) else "INTERNAL_ERROR"
            error_code = code
            retryable = exc.retryable if isinstance(exc, RAGCoreError) else True
            if not isinstance(exc, RAGCoreError):
                logger.exception("job_handler_crashed", extra={"job_id": str(job_id)})
            elif retryable:
                wake.mark_unhealthy(settings.RAG_CORE_URL)  # re-check (and wake) RAG Core before the retry
            if retryable and attempt < max_attempts:
                owned = await job_service.finish(db, job_id, attempt, status=job_status.PENDING, error=code, locked_at=None,
                                                 run_after=utcnow() + job_service.retry_delay(attempt))
                outcome = "retry_scheduled" if owned else "superseded"
            else:
                owned = await job_service.finish(db, job_id, attempt, status=job_status.FAILED, error=code, locked_at=None,
                                                 completed_at=utcnow())
                if owned:
                    await document_jobs.on_final_failure(db, job, code)
                outcome = "failed" if owned else "superseded"
            await db.commit()
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            METRICS.inc("jobs_finished", job_type=job_type, outcome=outcome)
            METRICS.observe_ms("job_duration", duration_ms, job_type=job_type, outcome=outcome)
            if error_code:
                METRICS.inc("job_errors", job_type=job_type, code=error_code)
            logger.info(
                "job_finished",
                extra={
                    "job_id": str(job_id),
                    "job_type": job_type,
                    "document_id": str(resource_id) if resource_id else None,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "outcome": outcome,
                    "error_code": error_code,
                    "duration_ms": duration_ms,
                },
            )
            request_id_var.reset(token)
            trace_id_var.reset(trace_token)
        return True


class Worker:
    def __init__(self, concurrency: int = settings.WORKER_CONCURRENCY):
        self.concurrency = concurrency
        self._tasks = []
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        # Crash recovery before taking new work.
        async with session_scope() as db:
            await cleanup_jobs.run_all(db)
        self._tasks = [asyncio.create_task(self._loop(i), name=f"worker-{i}") for i in range(self.concurrency)]
        self._tasks.append(asyncio.create_task(self._cleanup_loop(), name="worker-cleanup"))
        self._tasks.append(asyncio.create_task(self._stale_loop(), name="worker-stale-sweep"))
        self._tasks.append(asyncio.create_task(self._metrics_loop(), name="worker-queue-metrics"))
        if settings.KEEP_AWAKE_URL:
            self._tasks.append(asyncio.create_task(self._keep_awake_loop(), name="worker-keep-awake"))
        logger.info("worker_started", extra={"concurrency": self.concurrency})

    async def stop(self) -> None:
        self._stopping.set()
        job_service.wakeup_event().set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _loop(self, index: int) -> None:
        wake = job_service.wakeup_event()
        while not self._stopping.is_set():
            try:
                if await process_one():
                    continue
            except Exception:
                logger.exception("worker_loop_error", extra={"worker": index})
            wake.clear()
            try:
                await asyncio.wait_for(wake.wait(), timeout=settings.WORKER_POLL_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                pass

    async def _stale_loop(self) -> None:
        while not self._stopping.is_set():
            await asyncio.sleep(settings.STALE_SWEEP_INTERVAL_SECONDS)
            try:
                async with session_scope() as db:
                    if await cleanup_jobs.requeue_stale_jobs(db):
                        job_service.notify_worker()
            except Exception:
                logger.exception("stale_sweep_failed")

    async def _metrics_loop(self) -> None:
        """Queue depth and age as gauges, so a stalled queue is visible (and alertable) without reading logs."""
        while not self._stopping.is_set():
            try:
                async with session_scope() as db:
                    await job_service.publish_queue_metrics(db)
            except Exception:
                logger.warning("queue_metrics_failed")
            await asyncio.sleep(settings.QUEUE_METRICS_INTERVAL_SECONDS)

    async def _keep_awake_loop(self) -> None:
        """Hosts like Render free put a service to sleep after 15 minutes without inbound requests,
        even mid-job. While jobs are queued or running, request our own public /health so this
        service (and, through its status polls, RAG Core) stays up until the work is done."""
        url = settings.KEEP_AWAKE_URL.rstrip("/") + "/health"
        while not self._stopping.is_set():
            await asyncio.sleep(settings.KEEP_AWAKE_INTERVAL_SECONDS)
            try:
                async with session_scope() as db:
                    active = await job_service.has_active_jobs(db)
                if active:
                    async with httpx.AsyncClient(timeout=20.0) as client:
                        await client.get(url)
            except Exception:
                logger.warning("keep_awake_failed")

    async def _cleanup_loop(self) -> None:
        while not self._stopping.is_set():
            await asyncio.sleep(settings.RECONCILE_INTERVAL_SECONDS)
            try:
                async with session_scope() as db:
                    await cleanup_jobs.run_all(db)
            except Exception:
                logger.exception("cleanup_failed")


async def _main() -> None:  # pragma: no cover - manual entry point
    from app.core.logging import configure_logging

    configure_logging()
    worker = Worker()
    await worker.start()
    try:
        await asyncio.Event().wait()
    finally:
        await worker.stop()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_main())
