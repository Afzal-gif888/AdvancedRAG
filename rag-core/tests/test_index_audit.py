"""Index audit and recovery: detect that chunk rows and vectors disagree, repair what can be repaired, and say what cannot."""
import json
import uuid

from sqlalchemy import delete, select, update

from rag_core import db as rag_db
from rag_core.db import Chunk, IndexedDocument

U = uuid.uuid4()
TEXT = "\n\n".join(f"Paragraph {i}: " + ("the warehouse team counts inventory every quarter and records the result. " * 4) for i in range(8))


async def _ingest(client, user=U):
    doc = uuid.uuid4()
    meta = {"document_id": str(doc), "user_id": str(user), "filename": "inv.txt", "mime_type": "text/plain"}
    r = await client.post("/api/v1/rag/ingest", data={"metadata": json.dumps(meta)}, files={"file": ("inv.txt", TEXT.encode(), "text/plain")})
    assert r.status_code == 200, r.text
    return doc


async def _audit(client, doc, user=U):
    return await client.get(f"/api/v1/rag/documents/{doc}/audit", params={"user_id": str(user)})


async def _reindex(client, doc, user=U):
    return await client.post(f"/api/v1/rag/documents/{doc}/reindex", params={"user_id": str(user)})


async def _chunks(doc):
    async with rag_db.SessionLocal() as s:
        return (await s.scalars(select(Chunk).where(Chunk.document_id == doc).order_by(Chunk.chunk_index))).all()


async def test_a_healthy_document_audits_clean(client):
    doc = await _ingest(client)
    body = (await _audit(client, doc)).json()
    assert body["ok"] is True and body["problems"] == [] and body["repair"] is None
    assert body["chunks_expected"] == body["chunks_in_database"] == body["vectors_present"] >= 3 and body["vectors_missing"] == 0


async def test_lost_vectors_are_detected_and_repaired_by_reindex(client):
    doc = await _ingest(client)
    rows = await _chunks(doc)
    lost = [rows[0].id, rows[-1].id]
    async with rag_db.SessionLocal() as s:
        await s.execute(update(Chunk).where(Chunk.id.in_(lost)).values(embedding=None))   # the vector store lost them
        await s.commit()
    body = (await _audit(client, doc)).json()
    assert body["ok"] is False and body["problems"] == ["VECTORS_MISSING"] and body["vectors_missing"] == 2
    assert set(body["missing_sample"]) == set(lost) and "reindex" in body["repair"]

    assert (await _reindex(client, doc)).status_code == 200
    again = (await _audit(client, doc)).json()
    assert again["ok"] is True and again["vectors_missing"] == 0


async def test_missing_chunk_rows_cannot_be_repaired_from_vectors_and_the_audit_says_so(client):
    doc = await _ingest(client)
    rows = await _chunks(doc)
    async with rag_db.SessionLocal() as s:
        await s.execute(delete(Chunk).where(Chunk.id == rows[1].id))                      # a lost row in the middle
        await s.commit()
    body = (await _audit(client, doc)).json()
    assert body["ok"] is False and {"CHUNK_ROWS_MISSING", "CHUNK_INDEX_GAPS"} <= set(body["problems"])
    assert "reprocess" in body["repair"]
    assert (await _reindex(client, doc)).status_code == 200            # the surviving rows are re-embedded...
    assert (await _audit(client, doc)).json()["ok"] is False           # ...but the gap remains: reindex does not pretend otherwise


async def test_a_foreign_embedding_model_is_reported_and_cleared_by_reindex(client):
    doc = await _ingest(client)
    async with rag_db.SessionLocal() as s:
        await s.execute(update(IndexedDocument).where(IndexedDocument.document_id == doc).values(embedding_model="other-384d-model"))
        await s.commit()
    body = (await _audit(client, doc)).json()
    assert body["problems"] == ["EMBEDDING_MODEL_MISMATCH"] and body["embedding_model"] == "other-384d-model"
    await _reindex(client, doc)
    assert (await _audit(client, doc)).json()["ok"] is True


async def test_audit_is_read_only_and_owner_scoped(client):
    doc = await _ingest(client)
    before = [(c.id, c.text, c.embedding) for c in await _chunks(doc)]
    for _ in range(2):
        assert (await _audit(client, doc)).status_code == 200
    assert [(c.id, c.text, c.embedding) for c in await _chunks(doc)] == before
    assert (await _audit(client, doc, user=uuid.uuid4())).status_code == 404
    assert (await _audit(client, uuid.uuid4())).status_code == 404
