"""Keyword-leg ranking modes (PostgreSQL only: they use the real full-text machinery) and weighted fusion."""
import json
import uuid

import pytest
from sqlalchemy import select

from conftest import IS_PG
from rag_core import db as rag_db
from rag_core import pipeline
from rag_core.db import Chunk

U = uuid.uuid4()
pytestmark = pytest.mark.skipif(not IS_PG, reason="uses PostgreSQL full-text search")


async def _ingest(client, name, text):
    doc = uuid.uuid4()
    meta = {"document_id": str(doc), "user_id": str(U), "filename": name, "mime_type": "text/plain"}
    r = await client.post("/api/v1/rag/ingest", data={"metadata": json.dumps(meta)}, files={"file": (name, text.encode(), "text/plain")})
    assert r.status_code == 200, r.text
    return doc


async def _corpus(client):
    """Ten documents repeat the common words many times; exactly one mentions the rare word."""
    docs = []
    for i in range(10):
        docs.append(await _ingest(client, f"common{i}.txt", " ".join(["The weekly report covers the weekly schedule of the weekly meeting."] * 6)))
    rare = await _ingest(client, "rare.txt", "The quarterly xylophone inventory is counted by the warehouse team.")
    return docs + [rare], rare


async def _rank(mode, query, docs):
    async with rag_db.SessionLocal() as db:
        hits = await pipeline._keyword(db, U, query, docs, 20, mode=mode)
        ids = {c.id: c.document_id for c in (await db.scalars(select(Chunk).where(Chunk.document_id.in_(docs)))).all()}
    return [ids[cid] for cid, _ in hits]


async def test_idf_mode_puts_the_rare_term_first_where_cover_density_may_not(client):
    docs, rare = await _corpus(client)
    query = "weekly schedule xylophone"
    by_idf = await _rank("idf", query, docs)
    assert by_idf and by_idf[0] == rare, "BM25 weighting must prefer the chunk with the only rare term"


@pytest.mark.parametrize("mode", ["cover_density", "tf", "and_then_or", "idf"])
async def test_every_mode_returns_scoped_ranked_candidates(client, mode):
    docs, rare = await _corpus(client)
    ids = await _rank(mode, "xylophone inventory", docs)
    assert ids[0] == rare and set(ids) <= set(docs)
    other = await _rank(mode, "xylophone inventory", [uuid.uuid4()])
    assert other == [], "documents outside the allowed set are never returned"


async def test_stopword_only_or_empty_queries_return_nothing(client):
    docs, _ = await _corpus(client)
    for mode in ("idf", "cover_density", "and_then_or", "tf"):
        assert await _rank(mode, "the of and", docs) == []


async def test_quote_and_operator_characters_in_the_query_cannot_break_the_idf_query(client):
    docs, rare = await _corpus(client)
    for q in ["xylophone' | 'x", "xylophone & !inventory", "'; DROP TABLE rag_chunks; --", "xylophone <-> inventory", "(((xylophone"]:
        got = await _rank("idf", q, docs)
        assert isinstance(got, list)
    assert await _rank("idf", "xylophone", docs) == [rare]


def test_weighted_rrf_defaults_to_plain_rrf_and_a_weight_changes_the_order():
    a, b = ["x", "y"], ["y", "z"]
    plain = pipeline.rrf([a, b], 60)
    assert [i for i, _ in plain] == ["y", "x", "z"]
    assert pipeline.rrf([a, b], 60, None) == plain
    assert [i for i, _ in pipeline.rrf([a, b], 60, [1.0, 5.0])] == ["y", "z", "x"]


async def _retrieve(client, docs, query, top_k=None):
    body = {"user_id": str(U), "query": query, "document_ids": [str(d) for d in docs]}
    if top_k:
        body["top_k"] = top_k
    r = await client.post("/api/v1/rag/retrieve", json=body)
    assert r.status_code == 200, r.text
    return r.json()["chunks"]


async def test_retrieve_finds_an_exact_identifier_among_many_generic_chunks_by_default(client):
    from rag_core.config import settings
    assert settings.KEYWORD_RANKING == "idf"
    docs = []
    for i in range(12):   # many chunks that share the generic words of the question
        docs.append(await _ingest(client, f"spec{i}.txt", " ".join([f"Product {i} article number list: the product has an article number and a product code."] * 8)))
    target = await _ingest(client, "target.txt", "The special mounting kit has article number QZX7-4412-K and ships in a grey box.")
    # top_k = the fused candidate list the reranker receives: fusion must keep the identifier chunk in it. (The final top 6 is the
    # reranker's decision; RRF alone favours chunks present in both lists, which is why the evaluation shows the rescue after rerank.)
    chunks = await _retrieve(client, docs + [target], "Which product has the article number QZX7-4412-K?", top_k=20)
    assert any("QZX7-4412-K" in c["text"] for c in chunks), "an exact identifier must survive fusion with many generic matches"


async def test_the_original_ranking_remains_selectable_and_loses_the_identifier(client, monkeypatch):
    """Documents WHY the default changed: ts_rank_cd ignores term rarity, so generic words fill the candidate list."""
    from rag_core.config import settings
    docs = []
    for i in range(30):
        docs.append(await _ingest(client, f"spec{i}.txt", " ".join([f"Product {i} article number list: the product has an article number and a product code."] * 8)))
    target = await _ingest(client, "target.txt", "The special mounting kit has article number QZX7-4412-K and ships in a grey box.")
    monkeypatch.setattr(settings, "KEYWORD_RANKING", "cover_density")
    async with rag_db.SessionLocal() as db:
        old = await pipeline._keyword(db, U, "Which product has the article number QZX7-4412-K?", docs + [target], 20, mode="cover_density")
        new = await pipeline._keyword(db, U, "Which product has the article number QZX7-4412-K?", docs + [target], 20, mode="idf")
        ids = {c.id: c.document_id for c in (await db.scalars(select(Chunk).where(Chunk.document_id.in_(docs + [target])))).all()}
    assert target in [ids[c] for c, _ in new][:1]
    assert target not in [ids[c] for c, _ in old], "if this ever passes, the original ranking no longer has the weakness the evaluation found"
