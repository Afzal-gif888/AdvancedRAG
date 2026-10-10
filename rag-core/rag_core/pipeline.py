"""Ingestion and hybrid retrieval.

Ingest:   parse -> chunk -> embed -> vector upsert + chunk rows (idempotent per document; the
          lifecycle - lock, background run, heartbeat, cleanup - lives in rag_core.ingestion)
Retrieve: semantic (vector store) + keyword (PostgreSQL full-text) -> RRF -> rerank -> context
"""
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import anyio
from sqlalchemy import Text, cast, delete, func, literal_column, select
from sqlalchemy.ext.asyncio import AsyncSession

from advancerag_common.observability import timed
from rag_core import errors
from rag_core.chunking import chunk_pages, fit_to_token_limit
from rag_core.config import Settings
from rag_core.db import Chunk, IndexedDocument, IngestionAttempt, TableData, chunk_id
from rag_core.embeddings import DOCUMENT, QUERY, EmbeddingProvider
from rag_core.parsing import CSV, parse, parse_table
from rag_core.rerank import Reranker
from rag_core.vectorstore import VectorRecord, VectorStore

log = logging.getLogger("advancerag.rag.pipeline")


@dataclass
class Services:
    settings: Settings
    embeddings: EmbeddingProvider
    vectors: VectorStore
    reranker: Reranker


# ---------------------------------------------------------------- ingestion

class IngestionCancelled(Exception):
    """The attempt no longer owns the document (document deleted, or superseded by a newer attempt)."""


class _NoControl:
    """Default lifecycle hooks (direct calls, tests). rag_core.ingestion supplies the real ones."""

    async def stage(self, name: str, **info) -> None:
        return None

    async def record_extent(self, extent: int) -> int:
        return extent

    async def before_commit(self, db: AsyncSession, *, chunks: int, timings: dict) -> None:
        return None

    async def may_discard(self) -> bool:
        return True

    async def still_owned(self) -> bool:
        return True


async def ingest(db: AsyncSession, svc: Services, *, document_id: uuid.UUID, user_id: uuid.UUID, filename: str,
                 mime_type: str, data: bytes, control=None) -> dict:
    """parse -> chunk -> embed -> (record extent) -> chunk rows + vector upsert -> commit.

    Failure safety: the vector extent is recorded (committed by the control) before any vector is
    written, and if anything fails once vectors have been written, this document's vectors and
    index rows are removed before the error propagates: a failed ingestion leaves no partial state.
    Re-ingesting the same document is idempotent (deterministic ids overwrite)."""
    control = control or _NoControl()
    cfg = svc.settings
    timings: dict = {}
    if len(data) > cfg.MAX_FILE_BYTES:
        raise errors.ServiceError(errors.DOCUMENT_TOO_LARGE, "The document is too large to index.", 413)

    existing = await db.get(IndexedDocument, document_id)
    if existing is not None and existing.user_id != user_id:
        # Document ids are backend-generated UUIDs; a mismatch is a caller bug or an attack.
        raise errors.ServiceError("DOCUMENT_OWNERSHIP_MISMATCH", "Document belongs to another user.", 409)

    await control.stage("parsing")
    t = time.perf_counter()
    async with timed("parse"):
        pages = await anyio.to_thread.run_sync(parse, data, mime_type)
    chunks = chunk_pages(pages, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
    if svc.embeddings.max_tokens:
        # Never let the model silently truncate a chunk: unembedded text would be invisible to
        # semantic search while still looking indexed.
        chunks = await anyio.to_thread.run_sync(fit_to_token_limit, chunks, svc.embeddings.count_tokens, svc.embeddings.max_tokens)
    if len(chunks) > cfg.MAX_CHUNKS_PER_DOCUMENT:
        raise errors.ServiceError(errors.DOCUMENT_TOO_LARGE, "The document has too much text to index.", 413)
    timings["parse_s"] = round(time.perf_counter() - t, 2)
    timings["pages"] = len(pages)
    # CSVs also keep the complete table for exact whole-dataset analysis (tables.py). Chunking
    # and embedding above are unchanged; this is an additional, bounded representation.
    table = None
    if mime_type == CSV:
        table = await anyio.to_thread.run_sync(parse_table, data, cfg.TABLE_MAX_ROWS, cfg.TABLE_MAX_COLUMNS)
        timings["table_rows"] = len(table.rows) if table else 0

    # Embedding in slices only adds progress/cancel checkpoints; the provider still batches
    # internally exactly as before, so the vectors are identical.
    t = time.perf_counter()
    vectors: List[List[float]] = []
    for i in range(0, len(chunks), cfg.INGEST_EMBED_SLICE):
        await control.stage("embedding", done=len(vectors), total=len(chunks))
        vectors.extend(await svc.embeddings.embed([c.text for c in chunks[i : i + cfg.INGEST_EMBED_SLICE]], DOCUMENT))
    timings["embed_s"] = round(time.perf_counter() - t, 2)

    # Every vector id this document may have (earlier versions, or a crashed attempt's partial
    # writes); recorded durably before the first vector is written.
    extent = await control.record_extent(max(existing.max_chunk_count if existing else 0, len(chunks)))
    wrote_vectors = False
    try:
        await db.execute(delete(Chunk).where(Chunk.document_id == document_id))
        doc = existing or IndexedDocument(document_id=document_id, user_id=user_id)
        doc.filename, doc.mime_type = filename, mime_type
        doc.chunk_count = len(chunks)
        doc.max_chunk_count = extent
        doc.page_count = max((p.number or 0) for p in pages) or None
        doc.embedding_model, doc.embedding_dimensions = svc.embeddings.model, svc.embeddings.dimensions
        db.add(doc)
        await db.flush()
        db.add_all(
            Chunk(id=chunk_id(document_id, c.index), document_id=document_id, user_id=user_id, chunk_index=c.index, page=c.page, text=c.text)
            for c in chunks
        )
        await db.execute(delete(TableData).where(TableData.document_id == document_id))
        if table is not None:
            db.add(TableData(document_id=document_id, user_id=user_id, columns=table.columns, rows=table.rows, row_count=len(table.rows)))
        await db.flush()
        records = [
            VectorRecord(
                id=chunk_id(document_id, c.index),
                values=v,
                # No chunk text in vector metadata: text stays in RAG Core's database.
                metadata={"document_id": str(document_id), "chunk_index": c.index, **({"page": c.page} if c.page else {})},
            )
            for c, v in zip(chunks, vectors)
        ]
        t = time.perf_counter()
        for i in range(0, len(records), cfg.INGEST_UPSERT_SLICE):
            await control.stage("upserting", done=i, total=len(records))
            wrote_vectors = True
            await svc.vectors.upsert(user_id, records[i : i + cfg.INGEST_UPSERT_SLICE], db=db)
        # Ids beyond the new chunk count: a longer earlier version, or a crashed attempt's writes.
        await svc.vectors.delete(user_id, [chunk_id(document_id, i) for i in range(len(chunks), extent)], db=db)
        timings["upsert_s"] = round(time.perf_counter() - t, 2)
        await control.stage("verifying")
        await _verify_vectors(svc, control, user_id, document_id, len(chunks), db)
        await control.stage("committing")
        await control.before_commit(db, chunks=len(chunks), timings=timings)
        await db.commit()
    except BaseException:
        await db.rollback()
        if wrote_vectors and await asyncio.shield(control.may_discard()):
            # Shielded: a cancelled task must still finish removing what it wrote.
            await asyncio.shield(discard_index(db, svc, document_id=document_id, user_id=user_id, extent=extent))
        raise
    log.info("document_indexed", extra={"document_id": str(document_id), "chunks": len(chunks), "pages": len(pages), **timings})
    return {"document_id": str(document_id), "chunks": len(chunks), "pages": len(pages), "embedding_model": svc.embeddings.model,
            "timings": timings}


def verification_ids(document_id: uuid.UUID, count: int) -> List[str]:
    """First, middle and last vector plus the first vector of every 100-vector upsert request, so a
    request that did not land cannot go unnoticed. At most 54 ids for 5000 chunks: one fetch call."""
    return [chunk_id(document_id, i) for i in sorted({0, count // 2, count - 1, *range(0, count, 100)})]


async def _verify_vectors(svc: Services, control, user_id: uuid.UUID, document_id: uuid.UUID, count: int, db: AsyncSession) -> None:
    """A document is only reported indexed once its vectors can be read back from the store
    (see verification_ids). Pinecone is eventually consistent, so this waits a bounded time.
    A miss caused by a concurrent delete (it removed vectors this attempt wrote) stops at once."""
    sample = verification_ids(document_id, count)
    deadline = time.monotonic() + svc.settings.INGEST_VERIFY_SECONDS
    while True:
        if await svc.vectors.count_existing(user_id, sample, db=db) == len(sample):
            return
        if not await control.still_owned():
            raise IngestionCancelled()
        if time.monotonic() >= deadline:
            raise errors.ServiceError("VECTOR_INDEX_UNVERIFIED", "The document's vectors could not be verified in the index.", 503,
                                      retryable=True)
        await asyncio.sleep(1.0)


async def discard_index(db: AsyncSession, svc: Services, *, document_id: uuid.UUID, user_id: uuid.UUID, extent: int) -> None:
    """Remove this document's vectors (ids 0..extent-1, in the owner's namespace) and index rows.
    Scoped to one document of one user; never touches anything else."""
    await svc.vectors.delete(user_id, [chunk_id(document_id, i) for i in range(extent)], db=db)
    await db.execute(delete(Chunk).where(Chunk.document_id == document_id, Chunk.user_id == user_id))
    await db.execute(delete(IndexedDocument).where(IndexedDocument.document_id == document_id, IndexedDocument.user_id == user_id))
    await db.commit()


async def delete_document(db: AsyncSession, svc: Services, *, document_id: uuid.UUID, user_id: uuid.UUID) -> bool:
    """Delete a document's index, also when an ingestion is still running or died half-way.
    Removing the attempt row takes the document away from a running attempt: it stops at its next
    checkpoint, cannot commit (rag_core.ingestion verifies ownership under the same row lock) and
    discards what it wrote."""
    attempt = await db.scalar(
        select(IngestionAttempt).where(IngestionAttempt.document_id == document_id, IngestionAttempt.user_id == user_id).with_for_update()
    )
    doc = await db.get(IndexedDocument, document_id)
    if doc is not None and doc.user_id != user_id:
        doc = None
    if doc is None and attempt is None:
        return False
    extent = max(doc.max_chunk_count if doc else 0, attempt.vector_extent if attempt else 0)
    await svc.vectors.delete(user_id, [chunk_id(document_id, i) for i in range(extent)], db=db)
    if attempt is not None:
        await db.delete(attempt)
    if doc is not None:
        await db.delete(doc)  # chunks cascade
    await db.commit()
    return True


# ---------------------------------------------------------------- retrieval

async def _keyword(db: AsyncSession, user_id: uuid.UUID, query: str, document_ids: Sequence[uuid.UUID], limit: int) -> List[Tuple[str, float]]:
    if db.bind.dialect.name == "postgresql":
        # OR semantics (any query term may match; ts_rank_cd orders by how well), like BM25.
        # plainto_tsquery parses safely and ANDs terms; swapping & for | turns it into an OR query.
        anded = func.plainto_tsquery(literal_column("'english'"), query)
        tsquery = func.to_tsquery(literal_column("'english'"), func.replace(cast(anded, Text), "&", "|"))
        vector = func.to_tsvector(literal_column("'english'"), Chunk.text)
        rank = func.ts_rank_cd(vector, tsquery)
        rows = await db.execute(
            select(Chunk.id, rank.label("rank"))
            .where(Chunk.user_id == user_id, Chunk.document_id.in_(list(document_ids)), vector.op("@@")(tsquery))
            .order_by(rank.desc())
            .limit(limit)
        )
        return [(r[0], float(r[1])) for r in rows.all()]
    # Portable fallback (tests on SQLite): term-overlap scoring.
    terms = {t for t in (w.strip(".,;:!?()\"'").lower() for w in query.split()) if len(t) > 2}
    rows = await db.execute(select(Chunk.id, Chunk.text).where(Chunk.user_id == user_id, Chunk.document_id.in_(list(document_ids))))
    scored = []
    for cid, text in rows.all():
        words = text.lower()
        score = sum(words.count(t) for t in terms)
        if score:
            scored.append((cid, float(score)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]


def rrf(rankings: List[List[str]], k: int) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


async def retrieve(db: AsyncSession, svc: Services, *, user_id: uuid.UUID, query: str, document_ids: Sequence[uuid.UUID],
                   top_k: Optional[int] = None) -> dict:
    cfg = svc.settings
    started = time.perf_counter()
    top_k = top_k or cfg.RETRIEVAL_TOP_K

    # Only documents this user actually has indexed.
    indexed = set(
        (await db.scalars(select(IndexedDocument.document_id).where(IndexedDocument.user_id == user_id, IndexedDocument.document_id.in_(list(document_ids))))).all()
    )
    meta = {"retrieval_strategy": "hybrid_rrf_rerank", "documents_searched": len(indexed), "embedding_model": svc.embeddings.model}
    if not indexed:
        meta.update(retrieved_chunks=0, retrieval_latency_ms=round((time.perf_counter() - started) * 1000, 2))
        return {"chunks": [], "context": "", "metadata": meta}

    query_vector = (await svc.embeddings.embed([query], QUERY))[0]
    try:
        semantic = await svc.vectors.query(user_id, query_vector, cfg.RETRIEVAL_CANDIDATES, list(indexed), db=db)
    except errors.ServiceError as exc:
        if not exc.retryable:
            raise
        # Vector store down: answer from the keyword index alone rather than failing the turn.
        log.warning("vector_search_failed_using_keyword_only", extra={"code": exc.code})
        semantic = []
        meta["retrieval_strategy"] = "keyword_only_degraded"
    async with timed("keyword_search"):
        keyword = await _keyword(db, user_id, query, list(indexed), cfg.RETRIEVAL_CANDIDATES)
    if not semantic and not keyword and meta["retrieval_strategy"] == "keyword_only_degraded":
        raise errors.vector_store_unavailable()
    fused = rrf([[cid for cid, _ in semantic], [cid for cid, _ in keyword]], cfg.RRF_K)[: cfg.RETRIEVAL_CANDIDATES]
    meta.update(semantic_candidates=len(semantic), keyword_candidates=len(keyword))

    rows = {c.id: c for c in (await db.scalars(select(Chunk).where(Chunk.id.in_([cid for cid, _ in fused]), Chunk.user_id == user_id))).all()}
    fused = [(cid, s) for cid, s in fused if cid in rows]

    rerank_started = time.perf_counter()
    try:
        ranked = await svc.reranker.rerank(query, [(cid, rows[cid].text) for cid, _ in fused], top_k) if fused else []
        meta["reranker"] = svc.reranker.name
        # No score cut-off on purpose: measured with bge-reranker-v2-m3, correct paraphrased
        # answers scored as low as 0.007 while on-topic unanswerable questions reached 0.38, so
        # any threshold drops real answers. "Not in the documents" is decided by the grounded
        # prompt instead (Agent Core).
    except Exception:
        # Degrade to RRF order rather than failing retrieval. Recorded in metadata.
        log.warning("rerank_failed_using_rrf_order")
        ranked = fused[:top_k]
        meta["reranker"] = "skipped"
    meta["rerank_latency_ms"] = round((time.perf_counter() - rerank_started) * 1000, 2)

    docs = {d.document_id: d for d in (await db.scalars(select(IndexedDocument).where(IndexedDocument.document_id.in_(list(indexed))))).all()}
    chunks, context_parts, used = [], [], 0
    for cid, score in ranked:
        row = rows[cid]
        header = f"[{len(chunks) + 1}] {docs[row.document_id].filename}" + (f" (page {row.page})" if row.page else "")
        block = f"{header}\n{row.text}"
        if used + len(block) > cfg.MAX_CONTEXT_CHARS and chunks:
            break
        used += len(block)
        context_parts.append(block)
        chunks.append({
            "source_id": len(chunks) + 1,
            "chunk_id": cid,
            "document_id": str(row.document_id),
            "filename": docs[row.document_id].filename,
            "page": row.page,
            "chunk_index": row.chunk_index,
            "text": row.text,
            "score": round(score, 6),
        })
    meta.update(retrieved_chunks=len(chunks), retrieval_latency_ms=round((time.perf_counter() - started) * 1000, 2))
    return {"chunks": chunks, "context": "\n\n".join(context_parts), "metadata": meta}
