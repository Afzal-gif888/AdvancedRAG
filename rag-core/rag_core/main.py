"""RAG Core service (internal; Bearer INTERNAL_SERVICE_TOKEN).

  POST   /api/v1/rag/ingestions                multipart: file + metadata JSON -> 202, ingestion runs in the background
  GET    /api/v1/rag/ingestions/{document_id}?user_id=   attempt status (running|succeeded|failed|stale)
  POST   /api/v1/rag/ingest                    synchronous variant of /ingestions (same lock and cleanup guarantees)
  DELETE /api/v1/rag/documents/{document_id}?user_id=
  POST   /api/v1/rag/retrieve                  {user_id, query, document_ids, top_k?}
  POST   /api/v1/rag/tables/describe           {user_id, document_ids} -> CSV table schemas
  POST   /api/v1/rag/tables/query              {user_id, document_id, queries[]} -> exact whole-table results
  GET    /health, /ready, /metrics
"""
import asyncio
import json
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional, Union

import anyio

from fastapi import Depends, FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from advancerag_common.observability import RequestContextMiddleware, configure_logging
from advancerag_common.service import ServiceError, install_error_handlers, internal_auth, metrics_endpoint
from rag_core import db as rag_db
from rag_core import ingestion, pipeline, tables
from rag_core.config import settings
from rag_core.db import IndexedDocument, TableData
from rag_core.embeddings import build_embeddings
from rag_core.rerank import build_reranker
from rag_core.vectorstore import build_vector_store, namespace_for

log = logging.getLogger("advancerag.rag")

_services: Optional[pipeline.Services] = None


def get_services() -> pipeline.Services:
    global _services
    if _services is None:
        embeddings = build_embeddings(settings)
        _services = pipeline.Services(settings, embeddings, build_vector_store(settings, embeddings.dimensions), build_reranker(settings))
    return _services


async def _warm_up(svc: pipeline.Services) -> None:
    try:
        await svc.embeddings.warm_up()
    except Exception:
        log.exception("embedding_model_warm_up_failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging("rag-core", settings.LOG_LEVEL)
    if rag_db.engine is None:
        rag_db.configure_engine(settings.database_url)
    svc = get_services()
    log.info("startup", extra={"environment": settings.ENVIRONMENT, "embeddings": svc.embeddings.model,
                                "dimensions": svc.embeddings.dimensions, "vector_store": settings.VECTOR_STORE})
    # Load (and on first run download) the embedding model in the background; /ready reports it.
    warm = asyncio.create_task(_warm_up(svc))
    yield
    warm.cancel()
    await ingestion.shutdown()
    if _services:
        for c in (_services.embeddings, _services.vectors, _services.reranker):
            await c.aclose()
    if rag_db.engine is not None:
        await rag_db.engine.dispose()


app = FastAPI(title="AdvancedRAG RAG Core", lifespan=lifespan, docs_url=None if settings.ENVIRONMENT == "production" else "/docs")
app.add_middleware(RequestContextMiddleware)
install_error_handlers(app)
metrics_endpoint(app, settings.internal_token)
auth = Depends(internal_auth(settings.internal_token))


class IngestMetadata(BaseModel):
    document_id: uuid.UUID
    user_id: uuid.UUID
    filename: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(min_length=1, max_length=100)


class RetrieveRequest(BaseModel):
    user_id: uuid.UUID
    query: str = Field(min_length=1, max_length=8000)
    document_ids: List[uuid.UUID] = Field(min_length=1, max_length=500)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)


def _metadata(raw: str) -> IngestMetadata:
    try:
        return IngestMetadata.model_validate(json.loads(raw))
    except (ValueError, ValidationError):
        raise ServiceError("INVALID_METADATA", "Invalid ingest metadata.", 422)


async def _spool(file: UploadFile) -> Path:
    """Copy the upload to a temporary file in 1 MiB steps (memory stays flat whatever the size) and refuse
    anything above MAX_FILE_BYTES while copying. The caller owns the returned path."""
    fd, name = tempfile.mkstemp(prefix="ingest-", suffix=".part", dir=settings.SPOOL_DIR or None)
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > settings.MAX_FILE_BYTES:
                    raise ServiceError("DOCUMENT_TOO_LARGE", "The document is too large to index.", 413)
                out.write(chunk)
    except BaseException:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise
    return Path(name)


@app.post("/api/v1/rag/ingestions", status_code=202, dependencies=[auth])
async def start_ingestion(metadata: str = Form(...), file: UploadFile = File(...), svc: pipeline.Services = Depends(get_services)):
    """Starts (or joins) the document's ingestion and returns at once; poll GET /ingestions/{id}."""
    meta = _metadata(metadata)
    path = await _spool(file)
    return await ingestion.IngestionManager(svc).start(document_id=meta.document_id, user_id=meta.user_id,
                                                        filename=meta.filename, mime_type=meta.mime_type, path=path)


@app.get("/api/v1/rag/ingestions/{document_id}", dependencies=[auth])
async def ingestion_status(document_id: uuid.UUID, user_id: uuid.UUID = Query(...), svc: pipeline.Services = Depends(get_services)):
    status = await ingestion.IngestionManager(svc).status(document_id=document_id, user_id=user_id)
    if status is None:
        raise ServiceError("NOT_FOUND", "No ingestion for this document.", 404)
    return status


@app.post("/api/v1/rag/ingest", dependencies=[auth])
async def ingest(metadata: str = Form(...), file: UploadFile = File(...), svc: pipeline.Services = Depends(get_services)):
    meta = _metadata(metadata)
    path = await _spool(file)
    result = await ingestion.IngestionManager(svc).run_sync(document_id=meta.document_id, user_id=meta.user_id,
                                                            filename=meta.filename, mime_type=meta.mime_type, path=path)
    return {k: v for k, v in result.items() if k != "timings"}


@app.delete("/api/v1/rag/documents/{document_id}", dependencies=[auth])
async def delete_document(document_id: uuid.UUID, user_id: uuid.UUID = Query(...), db: AsyncSession = Depends(rag_db.get_db),
                          svc: pipeline.Services = Depends(get_services)):
    if not await pipeline.delete_document(db, svc, document_id=document_id, user_id=user_id):
        raise ServiceError("NOT_FOUND", "Document not indexed.", 404)
    return Response(status_code=204)


@app.get("/api/v1/rag/documents/{document_id}/audit", dependencies=[auth])
async def audit_document(document_id: uuid.UUID, user_id: uuid.UUID = Query(...), db: AsyncSession = Depends(rag_db.get_db),
                         svc: pipeline.Services = Depends(get_services)):
    """Read-only: do the chunk rows and the vectors of this document agree (see pipeline.audit_document)?"""
    return await pipeline.audit_document(db, svc, document_id=document_id, user_id=user_id)


@app.post("/api/v1/rag/documents/{document_id}/reindex", dependencies=[auth])
async def reindex_document(document_id: uuid.UUID, user_id: uuid.UUID = Query(...), db: AsyncSession = Depends(rag_db.get_db),
                           svc: pipeline.Services = Depends(get_services)):
    """Re-embed a document from its stored chunks (see pipeline.reindex_document)."""
    return await pipeline.reindex_document(db, svc, document_id=document_id, user_id=user_id)


@app.post("/api/v1/rag/retrieve", dependencies=[auth])
async def retrieve(body: RetrieveRequest, db: AsyncSession = Depends(rag_db.get_db), svc: pipeline.Services = Depends(get_services)):
    result = await pipeline.retrieve(db, svc, user_id=body.user_id, query=body.query, document_ids=body.document_ids, top_k=body.top_k)
    meta = result["metadata"]
    # Safe diagnostics: ids and counts only (no query text, no chunk text).
    log.info("retrieve_done", extra={
        "user_id": str(body.user_id), "namespace": namespace_for(body.user_id),
        "documents_requested": len(body.document_ids), "documents_searched": meta.get("documents_searched"),
        "retrieved_document_ids": sorted({c["document_id"] for c in result["chunks"]}),
        "retrieved_chunks": len(result["chunks"]), "retrieval_strategy": meta.get("retrieval_strategy"),
    })
    return result


class TablesDescribeRequest(BaseModel):
    user_id: uuid.UUID
    document_ids: List[uuid.UUID] = Field(min_length=1, max_length=50)


class TableFilterIn(BaseModel):
    column: str = Field(min_length=1, max_length=200)
    op: Literal["eq", "ne", "contains", "not_contains", "gt", "gte", "lt", "lte", "in"]
    value: Union[float, str, List[str]]


class TableQueryIn(BaseModel):
    operation: Literal["count", "distribution", "crosstab", "stats", "top_terms"]
    column: Optional[str] = Field(default=None, max_length=200)
    by: Optional[str] = Field(default=None, max_length=200)
    filters: List[TableFilterIn] = Field(default_factory=list, max_length=tables.MAX_FILTERS)
    limit: int = Field(default=20, ge=1, le=tables.MAX_LIMIT)


class TableQueryRequest(BaseModel):
    user_id: uuid.UUID
    document_id: uuid.UUID
    queries: List[TableQueryIn] = Field(min_length=1, max_length=5)


@app.post("/api/v1/rag/tables/describe", dependencies=[auth])
async def describe_tables(body: TablesDescribeRequest, db: AsyncSession = Depends(rag_db.get_db)):
    """Schemas of the caller's CSV tables among `document_ids` (others are silently absent)."""
    rows = (await db.execute(
        select(TableData, IndexedDocument.filename)
        .join(IndexedDocument, IndexedDocument.document_id == TableData.document_id)
        .where(TableData.user_id == body.user_id, TableData.document_id.in_(body.document_ids))
    )).all()
    out = []
    for table, filename in rows:
        columns = await anyio.to_thread.run_sync(tables.describe, table.columns, table.rows)
        out.append({"document_id": str(table.document_id), "filename": filename, "row_count": table.row_count, "columns": columns})
    return {"tables": out}


@app.post("/api/v1/rag/tables/query", dependencies=[auth])
async def query_table(body: TableQueryRequest, db: AsyncSession = Depends(rag_db.get_db)):
    """Exact results over the whole stored table. Ownership is checked before anything runs."""
    table = await db.get(TableData, body.document_id)
    if table is None or table.user_id != body.user_id:
        raise ServiceError("TABLE_NOT_FOUND", "No table for this document.", 404)
    filename = await db.scalar(select(IndexedDocument.filename).where(IndexedDocument.document_id == body.document_id))

    def run_all():
        return [tables.run(table.columns, table.rows, operation=q.operation, column=q.column, by=q.by, limit=q.limit,
                           filters=[tables.Filter(f.column, f.op, f.value) for f in q.filters]) for q in body.queries]

    results = await anyio.to_thread.run_sync(run_all)
    log.info("table_query_done", extra={"user_id": str(body.user_id), "document_id": str(body.document_id),
                                        "operations": [q.operation for q in body.queries], "rows_total": table.row_count})
    return {"document_id": str(body.document_id), "filename": filename, "results": results}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready(svc: pipeline.Services = Depends(get_services)):
    checks = {}
    try:
        async with rag_db.SessionLocal() as s:
            await s.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:
        checks["database"] = False
    vec_ok, vec_detail = await svc.vectors.health()
    checks["vector_store"] = vec_ok
    integrity = {"model_mismatch_documents": None}
    try:
        async with rag_db.SessionLocal() as s:
            integrity["model_mismatch_documents"] = int(await s.scalar(
                select(func.count()).select_from(IndexedDocument).where(
                    (IndexedDocument.embedding_model != svc.embeddings.model) | (IndexedDocument.embedding_dimensions != svc.embeddings.dimensions))
            ) or 0)
    except Exception:
        pass
    checks["embedding_model_loaded"] = svc.embeddings.is_ready()
    ok = all(checks.values())
    return JSONResponse(status_code=200 if ok else 503, content={
        "status": "ready" if ok else "not_ready",
        "checks": checks,
        "embedding": {"model": svc.embeddings.model, "dimensions": svc.embeddings.dimensions},
        "detail": vec_detail,
        # Informational: documents indexed with another model are not searched (they need a re-index);
        # that does not make the service itself unready.
        "index_integrity": integrity,
    })
