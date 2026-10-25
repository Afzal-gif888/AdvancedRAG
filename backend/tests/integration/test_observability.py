"""Trace propagation, queue gauges and job metrics: failures must be visible without reading logs."""
import re
from datetime import timedelta

from advancerag_common.observability import METRICS, new_trace_id, parse_traceparent, trace_id_var, traceparent_header
from sqlalchemy import update

from app.core.config import settings
from app.core.time import utcnow
from app.db import session as db_session
from app.db.models import BackgroundJob
from app.integrations.http import outbound_headers
from app.integrations.rag_core import RAGCoreError
from app.main import app
from app.services import job_service
from app.workers.runner import process_one
from conftest import upload

API = "/api/v1"
TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"


# ---------------------------------------------------------------- trace context

def test_traceparent_parsing_accepts_only_well_formed_ids():
    assert parse_traceparent(f"00-{TRACE}-00f067aa0ba902b7-01") == TRACE
    for bad in [None, "", "garbage", f"01-{TRACE}-00f067aa0ba902b7-01", f"00-{'0' * 32}-00f067aa0ba902b7-01", f"00-{TRACE}-{'0' * 16}-01",
                f"00-{TRACE[:-1]}-00f067aa0ba902b7-01", f"00-{TRACE}-00f067aa0ba902b7-01\r\nX-Evil: 1"]:
        assert parse_traceparent(bad) is None, bad


def test_outbound_traceparent_keeps_the_trace_and_uses_a_fresh_span():
    token = trace_id_var.set(TRACE)
    try:
        a, b = traceparent_header(), traceparent_header()
    finally:
        trace_id_var.reset(token)
    assert a.startswith(f"00-{TRACE}-") and a.endswith("-01") and a != b
    assert traceparent_header() is None, "no trace in scope: no header is invented"


async def test_incoming_trace_is_adopted_forwarded_and_logged(client):
    """Drive the real middleware: a probe route reports what an internal call would send and logs a line."""
    import json
    import logging

    from app.core.logging import JsonFormatter

    captured = []

    class Capture(logging.Handler):
        def emit(self, record):
            captured.append(json.loads(JsonFormatter().format(record)))

    async def probe():
        logging.getLogger("advancerag.probe").warning("probe_line")
        return outbound_headers()

    app.add_api_route("/__trace_probe", probe)
    handler = Capture()
    logging.getLogger("advancerag.probe").addHandler(handler)
    try:
        r = await client.get("/__trace_probe", headers={"traceparent": f"00-{TRACE}-00f067aa0ba902b7-01"})
        forwarded = r.json()["traceparent"]
        assert forwarded.startswith(f"00-{TRACE}-") and forwarded != f"00-{TRACE}-00f067aa0ba902b7-01", "same trace, new span"
        assert [c["trace_id"] for c in captured if c["msg"] == "probe_line"] == [TRACE]

        generated = (await client.get("/__trace_probe")).json()["traceparent"].split("-")[1]
        assert re.fullmatch(r"[0-9a-f]{32}", generated) and generated != TRACE
        hostile = (await client.get("/__trace_probe", headers={"traceparent": "00-zzzz-00f067aa0ba902b7-01"})).json()["traceparent"].split("-")[1]
        assert re.fullmatch(r"[0-9a-f]{32}", hostile), "a malformed header is replaced, never forwarded"
    finally:
        logging.getLogger("advancerag.probe").removeHandler(handler)
        app.router.routes[:] = [rt for rt in app.router.routes if getattr(rt, "path", "") != "/__trace_probe"]


# ---------------------------------------------------------------- queue gauges

async def test_queue_stats_show_depth_age_and_stuck_jobs(client, auth, rag, storage):
    await upload(client, auth, "a.txt", b"first")
    await upload(client, auth, "b.txt", b"second")
    async with db_session.session_scope() as db:
        await db.execute(update(BackgroundJob).values(run_after=utcnow() - timedelta(seconds=120)))
        await db.commit()
        stats = await job_service.queue_stats(db)
    s = stats["document_ingest"]
    assert s["pending"] == 2 and s["running"] == 0 and 110 <= s["oldest_pending_age_s"] <= 400 and s["stuck"] == 0

    # claim one and let its heartbeat go stale: it must show up as stuck
    async with db_session.session_scope() as db:
        job = await job_service.claim_next(db)
        await db.execute(update(BackgroundJob).where(BackgroundJob.id == job.id).values(locked_at=utcnow() - timedelta(seconds=settings.JOB_STALE_AFTER_SECONDS + 60)))
        await db.commit()
        stats = await job_service.queue_stats(db)
    assert stats["document_ingest"]["running"] == 1 and stats["document_ingest"]["stuck"] == 1 and stats["document_ingest"]["pending"] == 1


async def test_gauges_are_exposed_on_the_metrics_endpoint(client, auth):
    await upload(client, auth, "a.txt", b"metrics please")
    async with db_session.session_scope() as db:
        await job_service.publish_queue_metrics(db)
    text = (await client.get("/metrics")).text
    assert re.search(r'queue_jobs\{job_type="document_ingest",state="pending"\} 1(\.0)?', text)
    assert 'queue_oldest_pending_age_seconds{job_type="document_ingest"}' in text and 'queue_stuck_jobs{job_type="document_ingest"} 0' in text


async def test_job_outcomes_and_error_codes_are_counted(client, auth, rag, storage):
    await upload(client, auth, "a.txt", b"ok content")
    await process_one(rag=rag, storage=storage)
    await upload(client, auth, "b.txt", b"bad content")
    rag.fail = RAGCoreError("PARSE_FAILED", retryable=False)
    await process_one(rag=rag, storage=storage)
    text = METRICS.render()
    assert 'jobs_finished_total{job_type="document_ingest",outcome="succeeded"}' in text
    assert 'jobs_finished_total{job_type="document_ingest",outcome="failed"}' in text
    assert 'job_errors_total{code="PARSE_FAILED",job_type="document_ingest"}' in text
    assert "job_duration_ms_count" in text
