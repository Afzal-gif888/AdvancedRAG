"""Ingestion speed work: length-sorted embedding batches, bounded Pinecone retries and concurrent upserts,
and the stronger "is it really indexed" verification. Speed must never change what is stored: same
ids, same vectors, same metadata, and a failed or unverified ingestion leaves nothing behind.

The Pinecone side runs against an in-memory fake behind httpx.MockTransport, so failures, retries and
concurrency are exact and no network is involved."""
import asyncio
import math
import uuid

import httpx
import pytest
from sqlalchemy import func, select

from rag_core import db as rag_db
from rag_core import errors, pipeline, vectorstore
from rag_core.config import Settings, settings
from rag_core.db import Chunk, IndexedDocument, chunk_id
from rag_core.embeddings import FakeEmbeddings
from rag_core.rerank import NoopReranker
from rag_core.vectorstore import PineconeStore, VectorRecord

USER = uuid.UUID(int=11)


# ---------------------------------------------------------------- fake Pinecone

class FakePinecone:
    """Data plane in memory. `fail(path, first_id, seen)` returns an HTTP status to fail with (or None);
    `seen` counts how often this exact request (path + first vector id) has arrived."""

    def __init__(self):
        self.vectors = {}  # (namespace, id) -> {"values": ..., "metadata": ...}
        self.requests = []  # (method, path, number of vectors/ids)
        self.seen = {}
        self.inflight = self.max_inflight = 0
        self.fail = lambda path, first_id, seen: None
        self.drop = set()  # ids acknowledged by upsert but never stored (a silent loss)
        self.retry_after = None

    async def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET":
            namespace = request.url.params.get("namespace")
            ids = request.url.params.get_list("ids")
            self.requests.append(("GET", path, len(ids)))
            status = self.fail(path, ids[0] if ids else None, self._count(path, ids[0] if ids else None))
            if status:
                return httpx.Response(status)
            return httpx.Response(200, json={"vectors": {i: {} for i in ids if (namespace, i) in self.vectors}})
        import json

        body = json.loads(request.content)
        items = body.get("vectors") or body.get("ids") or []
        first = (items[0]["id"] if items and isinstance(items[0], dict) else items[0]) if items else None
        self.requests.append(("POST", path, len(items)))
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await asyncio.sleep(0.01)  # lets concurrent requests overlap
            status = self.fail(path, first, self._count(path, first))
            if status:
                headers = {"Retry-After": self.retry_after} if self.retry_after else {}
                return httpx.Response(status, headers=headers, json={"error": "injected"})
            if path.endswith("/upsert"):
                for v in body["vectors"]:
                    if v["id"] not in self.drop:
                        self.vectors[(body["namespace"], v["id"])] = {"values": v["values"], "metadata": v["metadata"]}
                return httpx.Response(200, json={"upsertedCount": len(body["vectors"])})
            if path.endswith("/delete"):
                for i in body["ids"]:
                    self.vectors.pop((body["namespace"], i), None)
                return httpx.Response(200, json={})
            return httpx.Response(404)
        finally:
            self.inflight -= 1

    def _count(self, path, first):
        key = (path, first)
        self.seen[key] = self.seen.get(key, 0) + 1
        return self.seen[key]

    def store(self, **kw) -> PineconeStore:
        defaults = dict(api_key="k", index="idx", dimensions=64, api_url="https://api.test", host="https://idx.test",
                        transport=httpx.MockTransport(self.handler), upsert_concurrency=3, max_attempts=3)
        return PineconeStore(**{**defaults, **kw})

    def ids(self, document_id=None):
        return {i for (_, i) in self.vectors if document_id is None or i.startswith(f"{document_id}:")}

    def upserts(self):
        return [n for m, p, n in self.requests if m == "POST" and p.endswith("/upsert")]


@pytest.fixture
def pc():
    return FakePinecone()


@pytest.fixture
def delays(monkeypatch):
    """Backoff sleeps are recorded, not waited for."""
    recorded = []
    real_sleep = asyncio.sleep

    async def fast(seconds, *a, **k):
        if seconds >= 0.5:
            recorded.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(vectorstore.asyncio, "sleep", fast)
    return recorded


def records(document_id, n):
    return [VectorRecord(id=chunk_id(document_id, i), values=[float(i)] * 4, metadata={"document_id": str(document_id), "chunk_index": i})
            for i in range(n)]


# ---------------------------------------------------------------- embedding order

async def test_length_sorted_batches_return_the_same_vectors_in_the_callers_order(local_embeddings):
    """Sorting by length is purely an internal speed-up: every text gets exactly the vector it would
    get on its own, and the result order is the input order (shuffled lengths, duplicates included)."""
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split()
    texts = [" ".join(words[: 1 + (i * 7) % 12]) * (1 + i % 5) + f" item {i}" for i in range(14)] + ["alpha bravo", "alpha bravo"]
    batch = await local_embeddings.embed(texts, "document")
    assert len(batch) == len(texts) and all(len(v) == 384 for v in batch)
    for text, vec in zip(texts, batch):
        alone = (await local_embeddings.embed([text], "document"))[0]
        assert max(abs(a - b) for a, b in zip(vec, alone)) < 1e-6  # identical up to float noise
        assert abs(math.sqrt(sum(x * x for x in vec)) - 1.0) < 1e-6
    assert batch[-1] == batch[-2]  # identical texts, identical vectors


def test_embedding_batch_default_is_the_measured_fastest():
    fields = Settings.model_fields
    assert fields["EMBEDDING_BATCH_SIZE"].default == 4
    assert 1 <= fields["PINECONE_UPSERT_CONCURRENCY"].default <= 8 and 1 <= fields["PINECONE_MAX_ATTEMPTS"].default <= 5


# ---------------------------------------------------------------- verification sample

def test_verification_covers_every_upsert_request():
    doc = uuid.UUID(int=5)
    assert pipeline.verification_ids(doc, 1) == [chunk_id(doc, 0)]
    assert pipeline.verification_ids(doc, 250) == [chunk_id(doc, i) for i in (0, 100, 124, 200, 249)] or \
        pipeline.verification_ids(doc, 250) == [chunk_id(doc, i) for i in (0, 100, 125, 200, 249)]
    big = pipeline.verification_ids(doc, 5000)
    assert len(big) <= 54 and chunk_id(doc, 0) in big and chunk_id(doc, 4999) in big
    assert all(chunk_id(doc, i) in big for i in range(0, 5000, 100))  # one id per 100-vector request


# ---------------------------------------------------------------- bounded concurrency

async def test_upsert_sends_every_batch_once_with_bounded_concurrency(pc):
    doc = uuid.uuid4()
    store = pc.store(upsert_concurrency=3)
    await store.upsert(USER, records(doc, 1000))
    assert sorted(pc.upserts()) == [100] * 10  # ten 100-vector requests, none repeated
    assert pc.ids(doc) == {chunk_id(doc, i) for i in range(1000)}
    assert 2 <= pc.max_inflight <= 3  # concurrent, but never above the configured bound


async def test_concurrency_one_is_strictly_sequential(pc):
    store = pc.store(upsert_concurrency=1)
    await store.upsert(USER, records(uuid.uuid4(), 500))
    assert pc.max_inflight == 1


async def test_repeating_an_upsert_overwrites_the_same_ids(pc):
    doc = uuid.uuid4()
    store = pc.store()
    await store.upsert(USER, records(doc, 250))
    await store.upsert(USER, records(doc, 250))
    assert len(pc.vectors) == 250  # deterministic ids: a retry updates, never duplicates


# ---------------------------------------------------------------- bounded retries

async def test_transient_503_is_retried_with_backoff_then_succeeds(pc, delays):
    doc = uuid.uuid4()
    pc.fail = lambda path, first, seen: 503 if path.endswith("/upsert") and first.endswith(":100") and seen <= 2 else None
    await pc.store(max_attempts=3).upsert(USER, records(doc, 300))
    assert pc.ids(doc) == {chunk_id(doc, i) for i in range(300)}
    assert sum(1 for m, p, n in pc.requests if p.endswith("/upsert") and n == 100) == 5  # 3 batches + 2 retries of one
    assert 0.5 <= delays[0] < 0.8 and 1.0 <= delays[1] < 1.3  # exponential: 0.5 s, then 1 s (plus jitter)


async def test_429_honours_retry_after(pc, delays):
    doc = uuid.uuid4()
    pc.retry_after = "3"
    pc.fail = lambda path, first, seen: 429 if seen == 1 and path.endswith("/upsert") else None
    await pc.store().upsert(USER, records(doc, 100))
    assert len(pc.ids(doc)) == 100 and delays and delays[0] >= 3


async def test_retries_are_bounded_and_the_error_is_the_retryable_store_error(pc, delays):
    pc.fail = lambda path, first, seen: 503
    with pytest.raises(errors.ServiceError) as exc:
        await pc.store(max_attempts=3, upsert_concurrency=1).upsert(USER, records(uuid.uuid4(), 100))
    assert exc.value.code == errors.VECTOR_STORE_UNAVAILABLE and exc.value.retryable is True
    assert len(pc.requests) == 3  # exactly max_attempts, never an endless loop


async def test_a_permanent_failure_cancels_the_remaining_batches(pc, delays):
    pc.fail = lambda path, first, seen: 503 if first.endswith(":0") else None
    with pytest.raises(errors.ServiceError):
        await pc.store(max_attempts=2, upsert_concurrency=1).upsert(USER, records(uuid.uuid4(), 1000))
    assert len(pc.requests) == 2  # the first batch failed twice; nothing else was started


@pytest.mark.parametrize("status,code", [(400, errors.VECTOR_STORE_UNAVAILABLE), (401, errors.VECTOR_STORE_MISCONFIGURED),
                                         (403, errors.VECTOR_STORE_MISCONFIGURED), (404, errors.VECTOR_STORE_UNAVAILABLE)])
async def test_client_errors_are_never_retried(pc, delays, status, code):
    pc.fail = lambda path, first, seen: status
    with pytest.raises(errors.ServiceError) as exc:
        await pc.store(max_attempts=3).upsert(USER, records(uuid.uuid4(), 100))
    assert exc.value.code == code and len(pc.requests) == 1


async def test_network_errors_are_retried_too(delays):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("reset")
        return httpx.Response(200, json={})

    store = PineconeStore(api_key="k", index="i", dimensions=4, api_url="https://api.test", host="https://idx.test",
                          transport=httpx.MockTransport(handler), max_attempts=2)
    await store.upsert(USER, records(uuid.uuid4(), 10))
    assert calls["n"] == 2


async def test_verification_fetch_and_delete_are_retried(pc, delays):
    doc = uuid.uuid4()
    store = pc.store()
    await store.upsert(USER, records(doc, 100))
    pc.fail = lambda path, first, seen: 502 if seen == 1 else None
    assert await store.count_existing(USER, [chunk_id(doc, 0), chunk_id(doc, 99)]) == 2
    await store.delete(USER, [chunk_id(doc, i) for i in range(100)])
    assert pc.ids(doc) == set()


async def test_search_queries_are_not_retried(pc, delays):
    """Retries are for ingestion writes only: a failing chat query must fail fast, as before."""
    pc.fail = lambda path, first, seen: 503
    store = pc.store(max_attempts=3)
    with pytest.raises(errors.ServiceError):
        await store.query(USER, [0.1] * 4, 3, [uuid.uuid4()])
    assert len(pc.requests) == 1


# ---------------------------------------------------------------- whole pipeline against the fake

TEXT = "\n\n".join(f"Section {i}. The unit {i} maintenance interval is {i * 7} days and its torque is {i * 3} Nm." for i in range(2500))


@pytest.fixture
def svc_for(pc):
    def make(**store_kw):
        return pipeline.Services(settings, FakeEmbeddings(64), pc.store(**store_kw), NoopReranker())

    return make


async def _ingest(svc, doc):
    async with rag_db.SessionLocal() as db:
        return await pipeline.ingest(db, svc, document_id=doc, user_id=USER, filename="m.txt", mime_type="text/plain", data=TEXT.encode())


async def _db_chunks(doc):
    async with rag_db.SessionLocal() as db:
        return await db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == doc))


async def test_pipeline_writes_one_deterministic_vector_per_chunk(pc, svc_for):
    doc = uuid.uuid4()
    result = await _ingest(svc_for(), doc)
    n = result["chunks"]
    assert n > 450
    assert pc.ids(doc) == {chunk_id(doc, i) for i in range(n)} and await _db_chunks(doc) == n
    meta = pc.vectors[(f"user-{USER}", chunk_id(doc, 7))]["metadata"]
    assert meta == {"document_id": str(doc), "chunk_index": 7}  # same metadata as before (no text in the vector store)


async def test_a_transient_pinecone_error_mid_document_does_not_fail_the_ingestion(pc, svc_for, delays):
    doc = uuid.uuid4()
    pc.fail = lambda path, first, seen: 503 if path.endswith("/upsert") and first.endswith(":300") and seen == 1 else None
    result = await _ingest(svc_for(), doc)
    assert pc.ids(doc) == {chunk_id(doc, i) for i in range(result["chunks"])} and await _db_chunks(doc) == result["chunks"]


async def test_a_permanent_pinecone_error_fails_cleanly_and_leaves_nothing(pc, svc_for, delays):
    doc = uuid.uuid4()
    pc.fail = lambda path, first, seen: 503 if path.endswith("/upsert") and first.endswith(":200") else None
    with pytest.raises(errors.ServiceError) as exc:
        await _ingest(svc_for(upsert_concurrency=1), doc)
    assert exc.value.code == errors.VECTOR_STORE_UNAVAILABLE and exc.value.retryable
    assert pc.ids(doc) == set() and await _db_chunks(doc) == 0  # the earlier batches were removed again
    async with rag_db.SessionLocal() as db:
        assert await db.get(IndexedDocument, doc) is None
    # A retry afterwards (the Pinecone problem is over) completes and does not duplicate anything.
    pc.fail = lambda *a: None
    result = await _ingest(svc_for(), doc)
    assert pc.ids(doc) == {chunk_id(doc, i) for i in range(result["chunks"])}


async def test_a_batch_that_was_acknowledged_but_never_stored_is_never_reported_indexed(pc, svc_for, delays, monkeypatch):
    """Ready means verified: a lost 100-vector request in the middle is caught by the per-request sample."""
    monkeypatch.setattr(settings, "INGEST_VERIFY_SECONDS", 0.05)
    doc = uuid.uuid4()
    pc.drop = {chunk_id(doc, i) for i in range(200, 300)}  # the whole third request silently vanished
    with pytest.raises(errors.ServiceError) as exc:
        await _ingest(svc_for(), doc)
    assert exc.value.code == "VECTOR_INDEX_UNVERIFIED"
    assert pc.ids(doc) == set() and await _db_chunks(doc) == 0
    async with rag_db.SessionLocal() as db:
        assert await db.get(IndexedDocument, doc) is None


async def test_old_style_sampling_would_have_missed_that_batch(pc):
    """Documents why the sample grew: first, middle and last alone do not notice a lost middle batch."""
    doc = uuid.uuid4()
    n = 1000
    old = [chunk_id(doc, i) for i in sorted({0, n // 2, n - 1})]
    assert not any(chunk_id(doc, i) in old for i in range(200, 300))
    assert any(chunk_id(doc, i) in pipeline.verification_ids(doc, n) for i in range(200, 300))
