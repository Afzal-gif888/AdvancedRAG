"""Ingestion and hybrid retrieval.

Ingest:   parse -> chunk -> embed -> vector upsert + chunk rows (idempotent per document; the
          lifecycle - lock, background run, heartbeat, cleanup - lives in rag_core.ingestion)
Retrieve: semantic (vector store) + keyword (PostgreSQL full-text) -> RRF -> rerank -> context
"""
import asyncio
import logging
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
import typing
from typing import Any, Dict, List, Optional, Sequence, Tuple

import anyio
from sqlalchemy import Text, and_, cast, delete, func, literal, literal_column, select, union_all
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from advancerag_common.observability import METRICS, timed
from rag_core import errors
from rag_core.chunking import TextChunk, chunk_pages, chunk_pages_semantic, chunk_pages_structured, fit_to_token_limit
from rag_core.config import Settings
from rag_core.db import Chunk, IndexedDocument, IngestionAttempt, TableData, chunk_id
from rag_core.embeddings import DOCUMENT, QUERY, EmbeddingProvider
from rag_core.parsing import CSV, parse_document, parse_table
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

    async def before_commit(self, db: AsyncSession, *, chunks: int, timings: dict, extraction: Optional[dict] = None) -> None:
        return None

    async def may_discard(self) -> bool:
        return True

    async def still_owned(self) -> bool:
        return True


def embedding_text(chunk: TextChunk, cfg: Settings) -> str:
    """What is embedded for a chunk. With heading-aware/semantic chunking the heading path is prepended
    (a chunk in the middle of a long section otherwise loses what it is about); the stored and returned
    chunk text stays the bare text."""
    if cfg.CHUNKING_STRATEGY != "recursive" and cfg.EMBED_SECTION_CONTEXT and chunk.section:
        return f"{chunk.section}\n{chunk.text}"
    return chunk.text


def _source_size(data: Optional[bytes], path: Optional[Path]) -> int:
    return len(data) if data is not None else path.stat().st_size  # type: ignore[union-attr]


async def ingest(db: AsyncSession, svc: Services, *, document_id: uuid.UUID, user_id: uuid.UUID, filename: str,
                 mime_type: str, data: Optional[bytes] = None, path: Optional[Path] = None, control=None) -> dict:
    """parse -> chunk -> embed -> (record extent) -> chunk rows + vector upsert -> commit.

    The input is either ``data`` (bytes) or ``path`` (a spooled file on disk, parsed without holding a
    second copy of it in memory).

    Failure safety: the vector extent is recorded (committed by the control) before any vector is
    written, and if anything fails once vectors have been written, this document's vectors and
    index rows are removed before the error propagates: a failed ingestion leaves no partial state.
    Re-ingesting the same document is idempotent (deterministic ids overwrite)."""
    control = control or _NoControl()
    cfg = svc.settings
    source = path if path is not None else data
    if source is None:
        raise ValueError("ingest needs data or path")
    timings: dict = {}
    if _source_size(data, path) > cfg.MAX_FILE_BYTES:
        raise errors.ServiceError(errors.DOCUMENT_TOO_LARGE, "The document is too large to index.", 413)

    existing = await db.get(IndexedDocument, document_id)
    if existing is not None and existing.user_id != user_id:
        # Document ids are backend-generated UUIDs; a mismatch is a caller bug or an attack.
        raise errors.ServiceError("DOCUMENT_OWNERSHIP_MISMATCH", "Document belongs to another user.", 409)

    await control.stage("parsing")
    t = time.perf_counter()
    async with timed("parse"):
        parsed = await anyio.to_thread.run_sync(parse_document, source, mime_type, cfg)
    pages, report = parsed.pages, parsed.report
    chunk_info: dict = {"strategy": cfg.CHUNKING_STRATEGY}
    if cfg.CHUNKING_STRATEGY == "semantic":
        async def _embed_sentences(texts: List[str]):
            return await svc.embeddings.embed(texts, DOCUMENT)

        chunks, sem = await chunk_pages_semantic(pages, _embed_sentences, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP, cfg.SEMANTIC_BREAKPOINT_PERCENTILE,
                                                 cfg.SEMANTIC_MIN_CHUNK_CHARS, cfg.SEMANTIC_MAX_SENTENCES)
        chunk_info.update(sem)
        if sem.get("fallback"):
            report.warn(None, "SEMANTIC_CHUNKING_SKIPPED", "The document is too long for semantic chunking; heading-aware chunking was used.")
    elif cfg.CHUNKING_STRATEGY == "structure":
        chunks = chunk_pages_structured(pages, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
    else:
        chunks = chunk_pages(pages, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
    if svc.embeddings.max_tokens:
        # Never let the model silently truncate a chunk: unembedded text would be invisible to
        # semantic search while still looking indexed.
        chunks = await anyio.to_thread.run_sync(fit_to_token_limit, chunks, svc.embeddings.count_tokens, svc.embeddings.max_tokens)
    if len(chunks) > cfg.MAX_CHUNKS_PER_DOCUMENT:
        approx_mb = cfg.MAX_CHUNKS_PER_DOCUMENT * max(cfg.CHUNK_SIZE - cfg.CHUNK_OVERLAP, 1) // 1_000_000
        raise errors.ServiceError(
            errors.DOCUMENT_TOO_LARGE,
            f"The document produces {len(chunks)} chunks; the limit is {cfg.MAX_CHUNKS_PER_DOCUMENT} (roughly {approx_mb} MB of text). Split the document.",
            413,
        )
    timings["parse_s"] = round(time.perf_counter() - t, 2)
    timings["pages"] = len(pages)
    if report.ocr_pages:
        timings["ocr_pages"] = report.ocr_pages
    # CSVs also keep the complete table for exact whole-dataset analysis (tables.py). Chunking
    # and embedding above are unchanged; this is an additional, bounded representation.
    table = None
    if mime_type == CSV:
        table = await anyio.to_thread.run_sync(parse_table, source, cfg.TABLE_MAX_ROWS, cfg.TABLE_MAX_COLUMNS)
        timings["table_rows"] = len(table.rows) if table else 0

    # Embedding in slices only adds progress/cancel checkpoints; the provider still batches
    # internally exactly as before, so the vectors are identical.
    t = time.perf_counter()
    vectors: List[List[float]] = []
    embed_texts = [embedding_text(c, cfg) for c in chunks]
    for i in range(0, len(chunks), cfg.INGEST_EMBED_SLICE):
        await control.stage("embedding", done=len(vectors), total=len(chunks))
        vectors.extend(await svc.embeddings.embed(embed_texts[i : i + cfg.INGEST_EMBED_SLICE], DOCUMENT))
    timings["embed_s"] = round(time.perf_counter() - t, 2)
    if "embedding_texts" in chunk_info:
        timings["chunking_embedding_texts"] = chunk_info["embedding_texts"]

    extraction = report.to_dict()
    extraction["chunking"] = chunk_info
    pipeline_info = {
        "chunking": cfg.CHUNKING_STRATEGY, "chunk_size": cfg.CHUNK_SIZE, "chunk_overlap": cfg.CHUNK_OVERLAP,
        "embed_section_context": cfg.EMBED_SECTION_CONTEXT and cfg.CHUNKING_STRATEGY != "recursive",
        "text_cleanup": cfg.TEXT_CLEANUP, "ocr_enabled": cfg.OCR_ENABLED, "embedding_model": svc.embeddings.model,
    }

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
        doc.extraction, doc.pipeline_info = extraction, pipeline_info
        db.add(doc)
        await db.flush()
        db.add_all(
            Chunk(id=chunk_id(document_id, c.index), document_id=document_id, user_id=user_id, chunk_index=c.index, page=c.page, text=c.text,
                  section=c.section, extraction_method=c.method)
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
        await control.before_commit(db, chunks=len(chunks), timings=timings, extraction=extraction)
        await db.commit()
    except BaseException:
        await db.rollback()
        if wrote_vectors and await asyncio.shield(control.may_discard()):
            # Shielded: a cancelled task must still finish removing what it wrote.
            await asyncio.shield(discard_index(db, svc, document_id=document_id, user_id=user_id, extent=extent))
        raise
    log.info("document_indexed", extra={"document_id": str(document_id), "chunks": len(chunks), "pages": len(pages),
                                        "extraction_method": report.method, "partial": report.partial, **timings})
    return {"document_id": str(document_id), "chunks": len(chunks), "pages": len(pages), "embedding_model": svc.embeddings.model,
            "timings": timings, "extraction": extraction}


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


async def reindex_document(db: AsyncSession, svc: Services, *, document_id: uuid.UUID, user_id: uuid.UUID) -> dict:
    """Re-embed a document from its stored chunk rows (no original file needed) and overwrite its vectors.

    For repairing an index whose vectors were lost or were written by another embedding model. The vector
    ids are the same deterministic "<document_id>:<n>" ids, so this replaces in place: it never deletes
    anything, never touches another document or user, and never leaves two models' vectors for one chunk.
    The text embedded follows the settings recorded when the document was indexed (heading context), so
    a reindex reproduces the original embedding input. Refused while an ingestion of the document runs."""
    doc = await db.get(IndexedDocument, document_id)
    if doc is None or doc.user_id != user_id:
        raise errors.ServiceError("NOT_FOUND", "Document not indexed.", 404)
    attempt = await db.scalar(select(IngestionAttempt).where(IngestionAttempt.document_id == document_id, IngestionAttempt.user_id == user_id))
    if attempt is not None and attempt.status == "running":
        raise errors.ServiceError("INGESTION_IN_PROGRESS", "The document is being ingested; try again when it has finished.", 409, retryable=True)
    rows = (await db.scalars(select(Chunk).where(Chunk.document_id == document_id, Chunk.user_id == user_id).order_by(Chunk.chunk_index))).all()
    if not rows:
        raise errors.ServiceError("NOT_FOUND", "The document has no stored chunks to reindex.", 404)
    info = doc.pipeline_info or {}
    with_context = bool(info.get("embed_section_context"))
    texts = [f"{c.section}\n{c.text}" if with_context and c.section else c.text for c in rows]
    started = time.perf_counter()
    vectors: List[List[float]] = []
    for i in range(0, len(texts), svc.settings.INGEST_EMBED_SLICE):
        vectors.extend(await svc.embeddings.embed(texts[i : i + svc.settings.INGEST_EMBED_SLICE], DOCUMENT))
    records = [
        VectorRecord(id=c.id, values=v,
                     metadata={"document_id": str(document_id), "chunk_index": c.chunk_index, **({"page": c.page} if c.page else {})})
        for c, v in zip(rows, vectors)
    ]
    for i in range(0, len(records), svc.settings.INGEST_UPSERT_SLICE):
        await svc.vectors.upsert(user_id, records[i : i + svc.settings.INGEST_UPSERT_SLICE], db=db)
    sample = verification_ids(document_id, len(rows))
    deadline = time.monotonic() + svc.settings.INGEST_VERIFY_SECONDS
    while await svc.vectors.count_existing(user_id, sample, db=db) != len(sample):
        if time.monotonic() >= deadline:
            await db.rollback()
            raise errors.ServiceError("VECTOR_INDEX_UNVERIFIED", "The reindexed vectors could not be verified in the index.", 503, retryable=True)
        await asyncio.sleep(1.0)
    previous = doc.embedding_model
    doc.embedding_model, doc.embedding_dimensions = svc.embeddings.model, svc.embeddings.dimensions
    doc.pipeline_info = {**info, "embedding_model": svc.embeddings.model}
    await db.commit()
    METRICS.inc("reindexes", outcome="succeeded")
    return {"document_id": str(document_id), "chunks": len(rows), "embedding_model": svc.embeddings.model, "previous_embedding_model": previous,
            "seconds": round(time.perf_counter() - started, 2)}


async def audit_document(db: AsyncSession, svc: Services, *, document_id: uuid.UUID, user_id: uuid.UUID) -> dict:
    """Read-only consistency check of one document between the database and the vector store.

    Compares the chunk rows (the source of truth for text) with the vectors that can actually be read back, and the model recorded
    on the document with the current embedding model. It writes nothing. The repair for missing or foreign-model vectors is
    `reindex_document`; missing chunk ROWS cannot be rebuilt from vectors (Pinecone holds no text) and need a reprocess of the file."""
    doc = await db.get(IndexedDocument, document_id)
    if doc is None or doc.user_id != user_id:
        raise errors.ServiceError("NOT_FOUND", "Document not indexed.", 404)
    rows = (await db.execute(select(Chunk.id, Chunk.chunk_index).where(Chunk.document_id == document_id, Chunk.user_id == user_id)
                             .order_by(Chunk.chunk_index))).all()
    ids = [r[0] for r in rows]
    present = await svc.vectors.existing_ids(user_id, ids, db=db) if ids else set()
    missing = [cid for cid in ids if cid not in present]
    indexes = [r[1] for r in rows]
    problems = []
    if len(rows) != doc.chunk_count:
        problems.append("CHUNK_ROWS_MISSING" if len(rows) < doc.chunk_count else "CHUNK_ROWS_UNEXPECTED")
    if indexes != list(range(len(indexes))):
        problems.append("CHUNK_INDEX_GAPS")
    if missing:
        problems.append("VECTORS_MISSING")
    if doc.embedding_model != svc.embeddings.model or doc.embedding_dimensions != svc.embeddings.dimensions:
        problems.append("EMBEDDING_MODEL_MISMATCH")
    repair = None
    if "CHUNK_ROWS_MISSING" in problems or "CHUNK_INDEX_GAPS" in problems:
        repair = "reprocess the original file (chunk text cannot be rebuilt from vectors)"
    elif problems:
        repair = "POST /api/v1/rag/documents/{id}/reindex"
    return {"document_id": str(document_id), "ok": not problems, "problems": problems, "repair": repair,
            "chunks_expected": doc.chunk_count, "chunks_in_database": len(rows), "vectors_present": len(present),
            "vectors_missing": len(missing), "missing_sample": missing[:10],
            "embedding_model": doc.embedding_model, "current_embedding_model": svc.embeddings.model}


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

async def _keyword_idf(db: AsyncSession, user_id: uuid.UUID, query: str, document_ids: Sequence[uuid.UUID], limit: int,
                        k1: float = 1.2, b: float = 0.75, pool: int = 300) -> List[Tuple[str, float]]:
    """BM25 over the OR candidates. PostgreSQL's ts_rank_cd ignores how rare a term is; this restores that.

    Lexemes come from PostgreSQL's own analyser (the same stemming as the index), term frequencies from the candidates'
    tsvectors, document frequencies from index-assisted counts over this user's chunks, so the ranking uses statistics of the
    collection being searched. Length is measured in characters. Cost: four database round trips per query (lexemes, OR
    candidates, collection statistics + per-term document frequencies in one UNION, term frequencies)."""
    docs = list(document_ids)
    base = and_(Chunk.user_id == user_id, Chunk.document_id.in_(docs))
    lexemes = [r[0] for r in (await db.execute(sql_text("SELECT unnest(tsvector_to_array(to_tsvector('english', :q)))"), {"q": query})).all()]
    if not lexemes:
        return []
    ors = func.to_tsquery(literal_column("'english'"), " | ".join(f"'{l.replace(chr(39), '')}'" for l in lexemes))
    vector = func.to_tsvector(literal_column("'english'"), Chunk.text)
    cands = (await db.execute(
        select(Chunk.id, func.length(Chunk.text)).where(base, vector.op("@@")(ors)).order_by(func.ts_rank_cd(vector, ors).desc()).limit(pool)
    )).all()
    if not cands:
        return []
    # collection statistics and every term's document frequency in ONE round trip (each branch is index-assisted)
    branches = [select(literal_column("''").label("k"), func.count().label("c"), func.avg(func.length(Chunk.text)).label("a")).where(base)]
    for lex in lexemes:
        one = func.to_tsquery(literal_column("'english'"), f"'{lex.replace(chr(39), '')}'")
        branches.append(select(literal(lex).label("k"), func.count().label("c"), literal_column("NULL::numeric").label("a")).where(base, vector.op("@@")(one)))
    stat_rows = (await db.execute(union_all(*branches))).all()
    stats = {r[0]: r for r in stat_rows}
    n_docs = int(stats[""][1]) or 1
    avg_len = float(stats[""][2] or 1.0)
    dfs = {lex: int(stats[lex][1]) for lex in lexemes}
    tf_rows = (await db.execute(
        sql_text("SELECT c.id, u.lexeme, cardinality(u.positions) FROM rag_chunks c, LATERAL unnest(to_tsvector('english', c.text)) u "
             "WHERE c.id = ANY(:ids) AND u.lexeme = ANY(:lex)"), {"ids": [c[0] for c in cands], "lex": lexemes})).all()
    tf: Dict[str, Dict[str, int]] = {}
    for cid, lex, count in tf_rows:
        tf.setdefault(cid, {})[lex] = count
    scored = []
    for cid, length in cands:
        score = 0.0
        for lex in lexemes:
            f = tf.get(cid, {}).get(lex, 0)
            if f:
                idf = math.log(1.0 + (n_docs - dfs[lex] + 0.5) / (dfs[lex] + 0.5))
                score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * length / avg_len))
        scored.append((cid, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]


async def _keyword(db: AsyncSession, user_id: uuid.UUID, query: str, document_ids: Sequence[uuid.UUID], limit: int,
                   mode: str = "cover_density") -> List[Tuple[str, float]]:
    dialect_name = typing.cast(Any, db.bind).dialect.name
    if dialect_name == "postgresql" and mode == "idf":
        return await _keyword_idf(db, user_id, query, document_ids, limit)
    if dialect_name == "postgresql":
        # OR semantics (any query term may match; ts_rank_cd orders by how well), like BM25.
        # plainto_tsquery parses safely and ANDs terms; swapping & for | turns it into an OR query.
        anded = func.plainto_tsquery(literal_column("'english'"), query)
        ored = func.to_tsquery(literal_column("'english'"), func.replace(cast(anded, Text), "&", "|"))
        vector = func.to_tsvector(literal_column("'english'"), Chunk.text)
        rank_fn = func.ts_rank if mode == "tf" else func.ts_rank_cd

        async def run(tsquery):
            rank = rank_fn(vector, tsquery)
            rows = await db.execute(
                select(Chunk.id, rank.label("rank"))
                .where(Chunk.user_id == user_id, Chunk.document_id.in_(list(document_ids)), vector.op("@@")(tsquery))
                .order_by(rank.desc())
                .limit(limit)
            )
            return [(r[0], float(r[1])) for r in rows.all()]

        if mode == "and_then_or":
            hits = await run(anded)
            if len(hits) >= limit:
                return hits
            seen = {c for c, _ in hits}
            return (hits + [h for h in await run(ored) if h[0] not in seen])[:limit]
        return await run(ored)
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


def rrf(rankings: List[List[str]], k: int, weights: Optional[Sequence[float]] = None) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for n, ranking in enumerate(rankings):
        w = 1.0 if weights is None else weights[n]
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


async def retrieve(db: AsyncSession, svc: Services, *, user_id: uuid.UUID, query: str, document_ids: Sequence[uuid.UUID],
                   top_k: Optional[int] = None) -> dict:
    cfg = svc.settings
    started = time.perf_counter()
    top_k = top_k or cfg.RETRIEVAL_TOP_K

    # Only documents this user actually has indexed.
    rows_ = (await db.execute(
        select(IndexedDocument.document_id, IndexedDocument.embedding_model, IndexedDocument.embedding_dimensions)
        .where(IndexedDocument.user_id == user_id, IndexedDocument.document_id.in_(list(document_ids)))
    )).all()
    indexed = {r[0] for r in rows_}
    # A vector from another embedding model lives in a different space: comparing it with this model's
    # query vector is meaningless even when the dimension matches (the dimension checks cannot see that).
    # Such documents are excluded and reported; if nothing compatible is left the request fails loudly.
    incompatible = {r[0] for r in rows_ if r[1] != svc.embeddings.model or r[2] != svc.embeddings.dimensions}
    if incompatible:
        log.error("embedding_model_mismatch", extra={"documents": len(incompatible), "current_model": svc.embeddings.model})
        if incompatible == indexed:
            raise errors.ServiceError(
                errors.EMBEDDING_MODEL_MISMATCH,
                "These documents were indexed with a different embedding model; they must be re-indexed before they can be searched.",
                409,
            )
        indexed -= incompatible
    meta = {"retrieval_strategy": "hybrid_rrf_rerank", "documents_searched": len(indexed), "embedding_model": svc.embeddings.model}
    if incompatible:
        meta["model_mismatch_documents"] = len(incompatible)
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
        METRICS.inc("retrieval_degraded", strategy="keyword_only")
    async with timed("keyword_search"):
        keyword = await _keyword(db, user_id, query, list(indexed), cfg.RETRIEVAL_CANDIDATES, mode=cfg.KEYWORD_RANKING)
    if not semantic and not keyword and meta["retrieval_strategy"] == "keyword_only_degraded":
        raise errors.vector_store_unavailable()
    fused = rrf([[cid for cid, _ in semantic], [cid for cid, _ in keyword]], cfg.RRF_K,
                None if cfg.RRF_KEYWORD_WEIGHT == 1.0 else [1.0, cfg.RRF_KEYWORD_WEIGHT])[: cfg.RETRIEVAL_CANDIDATES]
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
        METRICS.inc("retrieval_degraded", strategy="rerank_skipped")
    meta["rerank_latency_ms"] = round((time.perf_counter() - rerank_started) * 1000, 2)

    docs = {d.document_id: d for d in (await db.scalars(select(IndexedDocument).where(IndexedDocument.document_id.in_(list(indexed))))).all()}
    chunks, context_parts, used = [], [], 0
    for cid, score in ranked:
        row = rows[cid]
        header = (f"[{len(chunks) + 1}] {docs[row.document_id].filename}" + (f" (page {row.page})" if row.page else "")
                  + (f" - {row.section}" if row.section else ""))
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
            "section": row.section,
            "extraction_method": row.extraction_method,
            "text": row.text,
            "score": round(score, 6),
        })
    meta.update(retrieved_chunks=len(chunks), retrieval_latency_ms=round((time.perf_counter() - started) * 1000, 2))
    return {"chunks": chunks, "context": "\n\n".join(context_parts), "metadata": meta}
