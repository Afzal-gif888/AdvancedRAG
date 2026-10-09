"""Vector stores. Pinecone (real) via its REST data plane; `local` keeps vectors in RAG Core's
DB with brute-force cosine search (development/tests only — refused in production).

Isolation: every user gets their own Pinecone namespace, and queries are additionally filtered
to the requested document ids.
"""
import logging
import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import httpx
from sqlalchemy import select

from advancerag_common.observability import timed
from rag_core import errors
from rag_core.config import Settings
from rag_core.db import Chunk

log = logging.getLogger("advancerag.rag.vectors")
PINECONE_API_VERSION = "2025-01"


@dataclass
class VectorRecord:
    id: str
    values: List[float]
    metadata: Dict


def namespace_for(user_id: uuid.UUID) -> str:
    return f"user-{user_id}"


class VectorStore(ABC):
    name: str

    @abstractmethod
    async def upsert(self, user_id: uuid.UUID, records: List[VectorRecord], db=None) -> None: ...

    @abstractmethod
    async def query(self, user_id: uuid.UUID, vector: List[float], top_k: int, document_ids: Sequence[uuid.UUID], db=None) -> List[Tuple[str, float]]: ...

    @abstractmethod
    async def delete(self, user_id: uuid.UUID, ids: List[str], db=None) -> None: ...

    @abstractmethod
    async def count_existing(self, user_id: uuid.UUID, ids: List[str], db=None) -> int:
        """How many of `ids` can be read back (ingestion verification)."""

    async def health(self) -> Tuple[bool, Optional[str]]:
        return True, None

    async def aclose(self) -> None:
        return None


class PineconeStore(VectorStore):
    name = "pinecone"

    def __init__(self, *, api_key: str, index: str, dimensions: int, api_url: str, host: Optional[str] = None,
                 timeout: float = 30.0, transport=None):
        self._key, self.index, self.dimensions = api_key, index, dimensions
        self._host = host
        self._control = httpx.AsyncClient(base_url=api_url.rstrip("/"), timeout=timeout, transport=transport)
        self._data: Optional[httpx.AsyncClient] = None
        self._transport, self._timeout = transport, timeout

    def _headers(self) -> Dict[str, str]:
        return {"Api-Key": self._key, "X-Pinecone-API-Version": PINECONE_API_VERSION}

    async def aclose(self) -> None:
        await self._control.aclose()
        if self._data:
            await self._data.aclose()

    async def describe(self) -> dict:
        try:
            r = await self._control.get(f"/indexes/{self.index}", headers=self._headers())
        except httpx.HTTPError:
            raise errors.vector_store_unavailable()
        if r.status_code in (401, 403):
            raise errors.ServiceError(errors.VECTOR_STORE_MISCONFIGURED, "Pinecone rejected the configured API key.", 503, retryable=True)
        if r.status_code == 404:
            raise errors.ServiceError(errors.VECTOR_STORE_MISCONFIGURED, "The configured Pinecone index does not exist.", 503, retryable=True)
        if not r.is_success:
            raise errors.vector_store_unavailable()
        return r.json()

    async def _client(self) -> httpx.AsyncClient:
        if self._data is None:
            host = self._host
            if not host:
                info = await self.describe()
                if info.get("dimension") != self.dimensions:
                    raise errors.ServiceError(
                        errors.VECTOR_STORE_MISCONFIGURED,
                        f"Pinecone index dimension {info.get('dimension')} does not match the embedding model's "
                        f"{self.dimensions} dimensions; create the index with dimension {self.dimensions}.",
                        503,
                        retryable=True,
                    )
                host = info["host"]
            base = host if host.startswith("http") else f"https://{host}"
            self._data = httpx.AsyncClient(base_url=base, timeout=self._timeout, transport=self._transport)
        return self._data

    async def _post(self, path: str, body: dict) -> dict:
        client = await self._client()
        try:
            r = await client.post(path, json=body, headers=self._headers())
        except httpx.HTTPError:
            raise errors.vector_store_unavailable()
        if not r.is_success:
            log.warning("pinecone_http_error", extra={"status": r.status_code, "path": path})
            if r.status_code in (401, 403):
                raise errors.ServiceError(errors.VECTOR_STORE_MISCONFIGURED, "Pinecone rejected the configured API key.", 503, retryable=True)
            raise errors.vector_store_unavailable()
        return r.json() if r.content else {}

    async def upsert(self, user_id, records, db=None):
        async with timed("vector_upsert", store="pinecone"):
            for i in range(0, len(records), 100):
                batch = records[i : i + 100]
                await self._post("/vectors/upsert", {
                    "namespace": namespace_for(user_id),
                    "vectors": [{"id": r.id, "values": r.values, "metadata": r.metadata} for r in batch],
                })

    async def query(self, user_id, vector, top_k, document_ids, db=None):
        async with timed("vector_query", store="pinecone"):
            data = await self._post("/query", {
                "namespace": namespace_for(user_id),
                "vector": vector,
                "topK": top_k,
                "includeMetadata": False,
                "filter": {"document_id": {"$in": [str(d) for d in document_ids]}},
            })
        return [(m["id"], float(m.get("score", 0.0))) for m in data.get("matches", [])]

    async def delete(self, user_id, ids, db=None):
        if not ids:
            return
        async with timed("vector_delete", store="pinecone"):
            for i in range(0, len(ids), 1000):
                await self._post("/vectors/delete", {"namespace": namespace_for(user_id), "ids": ids[i : i + 1000]})

    async def count_existing(self, user_id, ids, db=None):
        client = await self._client()
        found = 0
        for i in range(0, len(ids), 100):
            params = [("namespace", namespace_for(user_id))] + [("ids", x) for x in ids[i : i + 100]]
            try:
                r = await client.get("/vectors/fetch", params=params, headers=self._headers())
            except httpx.HTTPError:
                raise errors.vector_store_unavailable()
            if not r.is_success:
                raise errors.vector_store_unavailable()
            found += len((r.json() or {}).get("vectors") or {})
        return found

    async def health(self):
        try:
            info = await self.describe()
        except errors.ServiceError as exc:
            return False, exc.code
        if info.get("dimension") != self.dimensions:
            return False, f"DIMENSION_MISMATCH: index has {info.get('dimension')}, embedding model produces {self.dimensions}"
        ready = (info.get("status") or {}).get("ready", True)
        return bool(ready), None if ready else "INDEX_NOT_READY"


class LocalVectorStore(VectorStore):
    """Vectors live in rag_chunks.embedding; cosine similarity computed in Python."""

    name = "local"

    async def upsert(self, user_id, records, db=None):
        by_id = {r.id: r.values for r in records}
        rows = (await db.scalars(select(Chunk).where(Chunk.id.in_(list(by_id))))).all()
        for row in rows:
            row.embedding = by_id[row.id]

    async def query(self, user_id, vector, top_k, document_ids, db=None):
        rows = (await db.execute(
            select(Chunk.id, Chunk.embedding).where(Chunk.user_id == user_id, Chunk.document_id.in_(list(document_ids)))
        )).all()
        scored = []
        for cid, emb in rows:
            if emb:
                scored.append((cid, sum(a * b for a, b in zip(vector, emb)) / ((math.sqrt(sum(a * a for a in emb)) or 1.0))))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def delete(self, user_id, ids, db=None):
        return None  # rows are deleted with the chunks

    async def count_existing(self, user_id, ids, db=None):
        rows = (await db.scalars(select(Chunk.embedding).where(Chunk.id.in_(list(ids)), Chunk.user_id == user_id))).all()
        return sum(1 for e in rows if e)


def build_vector_store(cfg: Settings, dimensions: int) -> VectorStore:
    """`dimensions` comes from the embedding model, never from a free-standing setting."""
    if cfg.VECTOR_STORE == "local":
        return LocalVectorStore()
    if not cfg.PINECONE_API_KEY or not cfg.PINECONE_INDEX:
        return _Unconfigured()
    return PineconeStore(
        api_key=cfg.PINECONE_API_KEY.get_secret_value(),
        index=cfg.PINECONE_INDEX,
        dimensions=dimensions,
        api_url=cfg.PINECONE_API_URL,
        host=cfg.PINECONE_INDEX_HOST,
        timeout=cfg.HTTP_TIMEOUT_SECONDS,
    )


class _Unconfigured(VectorStore):
    name = "pinecone"

    def _err(self):
        return errors.ServiceError(errors.VECTOR_STORE_MISCONFIGURED, "PINECONE_API_KEY / PINECONE_INDEX are not configured.", 503, retryable=True)

    async def upsert(self, *a, **k):
        raise self._err()

    async def query(self, *a, **k):
        raise self._err()

    async def delete(self, *a, **k):
        raise self._err()

    async def count_existing(self, *a, **k):
        raise self._err()

    async def health(self):
        return False, "NOT_CONFIGURED"
