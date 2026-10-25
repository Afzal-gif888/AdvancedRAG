"""Index integrity (embedding model / dimension / metric) and bounded-memory upload handling."""
import json
import uuid
from pathlib import Path

import pytest
from sqlalchemy import update

from conftest import IS_PG
from rag_core import db as rag_db
from rag_core import errors, pipeline
from rag_core.config import settings
from rag_core.db import IndexedDocument

U1 = uuid.uuid4()
GUIDE = "Refunds are processed within five business days.\n\nShipping is free above fifty dollars."


async def _ingest_api(client, doc, data=GUIDE.encode(), name="g.txt", mime="text/plain", user=U1):
    meta = {"document_id": str(doc), "user_id": str(user), "filename": name, "mime_type": mime}
    return await client.post("/api/v1/rag/ingest", data={"metadata": json.dumps(meta)}, files={"file": (name, data, mime)})


async def _retrieve(client, docs, query="refunds", user=U1):
    return await client.post("/api/v1/rag/retrieve", json={"user_id": str(user), "query": query, "document_ids": [str(d) for d in docs]})


async def _set_model(doc, model=None, dims=None):
    values = {}
    if model is not None:
        values["embedding_model"] = model
    if dims is not None:
        values["embedding_dimensions"] = dims
    async with rag_db.SessionLocal() as s:
        await s.execute(update(IndexedDocument).where(IndexedDocument.document_id == doc).values(**values))
        await s.commit()


# ---------------------------------------------------------------- model / dimension integrity

async def test_documents_of_another_embedding_model_are_refused_loudly(client):
    doc = uuid.uuid4()
    assert (await _ingest_api(client, doc)).status_code == 200
    assert (await _retrieve(client, [doc])).json()["chunks"], "baseline: retrievable"
    await _set_model(doc, model="some-other-384d-model")   # same dimension, different embedding space
    r = await _retrieve(client, [doc])
    assert r.status_code == 409 and r.json()["error"]["code"] == "EMBEDDING_MODEL_MISMATCH" and r.json()["error"]["retryable"] is False


async def test_a_dimension_mismatch_is_refused_too(client):
    doc = uuid.uuid4()
    await _ingest_api(client, doc)
    await _set_model(doc, dims=1024)
    assert (await _retrieve(client, [doc])).status_code == 409


async def test_mismatched_documents_are_excluded_and_counted_while_good_ones_still_answer(client):
    good, stale = uuid.uuid4(), uuid.uuid4()
    await _ingest_api(client, good)
    await _ingest_api(client, stale)
    await _set_model(stale, model="old-model")
    body = (await _retrieve(client, [good, stale])).json()
    assert body["metadata"]["model_mismatch_documents"] == 1
    assert body["chunks"] and {c["document_id"] for c in body["chunks"]} == {str(good)}


async def test_readiness_reports_documents_that_need_a_reindex(client):
    doc = uuid.uuid4()
    await _ingest_api(client, doc)
    assert (await client.get("/ready")).json()["index_integrity"]["model_mismatch_documents"] == 0
    await _set_model(doc, model="old-model")
    ready = await client.get("/ready")
    assert ready.json()["index_integrity"]["model_mismatch_documents"] == 1
    assert ready.status_code == 200, "an index that needs re-indexing does not make the service itself unready"


async def test_pinecone_metric_must_match(real_provider_services):
    from mock_providers import state

    svc = real_provider_services
    ok, detail = await svc.vectors.health()
    assert ok is True, detail
    state.metric = "dotproduct"
    ok, detail = await svc.vectors.health()
    assert ok is False and detail.startswith("METRIC_MISMATCH")
    from rag_core.vectorstore import VectorRecord

    svc.vectors._data = None  # force the next call to describe the index again
    with pytest.raises(Exception) as exc:
        await svc.vectors.upsert(U1, [VectorRecord("d:0", [0.0] * svc.embeddings.dimensions, {"document_id": "d"})])
    assert getattr(exc.value, "code", "") == errors.VECTOR_STORE_MISCONFIGURED
    state.metric = "cosine"


# ---------------------------------------------------------------- uploads are spooled to disk and cleaned up

def spool_files(path: Path):
    return list(path.glob("ingest-*"))


async def test_spool_file_is_removed_after_a_successful_ingest(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SPOOL_DIR", str(tmp_path))
    assert (await _ingest_api(client, uuid.uuid4())).status_code == 200
    assert spool_files(tmp_path) == []


async def test_spool_file_is_removed_after_a_failed_ingest(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SPOOL_DIR", str(tmp_path))
    r = await _ingest_api(client, uuid.uuid4(), data=b"   ")           # no extractable text
    assert r.status_code == 422 and spool_files(tmp_path) == []


async def test_oversized_upload_is_rejected_while_spooling_and_leaves_nothing(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SPOOL_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "MAX_FILE_BYTES", 2 * 1024 * 1024)
    r = await _ingest_api(client, uuid.uuid4(), data=b"a" * (3 * 1024 * 1024))
    assert r.status_code == 413 and r.json()["error"]["code"] == "DOCUMENT_TOO_LARGE"
    assert spool_files(tmp_path) == []


async def test_path_and_bytes_give_identical_results(services, database, tmp_path):
    big = ("Sentence about topic %d with some filler words to make it longer. " * 60) % tuple(range(60))
    f = tmp_path / "in.txt"
    f.write_bytes(big.encode())
    out = []
    for kw in ({"data": big.encode()}, {"path": f}):
        async with rag_db.SessionLocal() as db:
            out.append(await pipeline.ingest(db, services, document_id=uuid.uuid4(), user_id=U1, filename="a.txt", mime_type="text/plain", **kw))
    assert out[0]["chunks"] == out[1]["chunks"] > 1
    assert f.exists(), "ingest(path=...) must not delete the caller's file; the ingestion manager owns spool cleanup"


async def test_chunk_limit_error_names_the_actual_constraint(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_CHUNKS_PER_DOCUMENT", 3)
    text = " ".join(f"word{i}" for i in range(2000))
    r = await _ingest_api(client, uuid.uuid4(), data=text.encode())
    err = r.json()["error"]
    assert r.status_code == 413 and err["code"] == "DOCUMENT_TOO_LARGE"
    assert "chunks" in err["message"] and "limit is 3" in err["message"]


# ---------------------------------------------------------------- reindex from stored chunks

async def test_reindex_repairs_a_mismatched_document_from_its_stored_chunks(client):
    doc = uuid.uuid4()
    assert (await _ingest_api(client, doc)).status_code == 200
    await _set_model(doc, model="some-other-384d-model")
    assert (await _retrieve(client, [doc])).status_code == 409

    r = await client.post(f"/api/v1/rag/documents/{doc}/reindex", params={"user_id": str(U1)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["previous_embedding_model"] == "some-other-384d-model" and body["chunks"] >= 1
    after = await _retrieve(client, [doc])
    assert after.status_code == 200 and after.json()["chunks"], "searchable again, without the original file"


async def test_reindex_is_scoped_to_the_owner_and_to_existing_documents(client):
    doc = uuid.uuid4()
    await _ingest_api(client, doc)
    other = uuid.uuid4()
    assert (await client.post(f"/api/v1/rag/documents/{doc}/reindex", params={"user_id": str(other)})).status_code == 404
    assert (await client.post(f"/api/v1/rag/documents/{uuid.uuid4()}/reindex", params={"user_id": str(U1)})).status_code == 404


async def test_reindex_does_not_change_the_stored_chunks(client):
    from sqlalchemy import select

    from rag_core.db import Chunk
    doc = uuid.uuid4()
    await _ingest_api(client, doc)

    async def snapshot():
        async with rag_db.SessionLocal() as s:
            return [(c.id, c.text, c.section) for c in (await s.scalars(select(Chunk).where(Chunk.document_id == doc).order_by(Chunk.chunk_index)))]
    before = await snapshot()
    assert (await client.post(f"/api/v1/rag/documents/{doc}/reindex", params={"user_id": str(U1)})).status_code == 200
    assert await snapshot() == before
