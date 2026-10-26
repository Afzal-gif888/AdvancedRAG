"""Vector stores. Pinecone (real) via its REST data plane; `local` keeps vectors in RAG Core's
DB with brute-force cosine search (development/tests only — refused in production).

Isolation: every user gets their own Pinecone namespace, and queries are additionally filtered
to the requested document ids.
"""
import asyncio
import logging
import math
import random
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

    async def existing_ids(self, user_id: uuid.UUID, ids: List[str], db=None) -> set:
        """Which of `ids` can be read back. Generic version on top of `count_existing`: a slice that is complete costs one
        call, an incomplete slice is probed id by id (an audit tool, not a hot path)."""
        present: set = set()
        for i in range(0, len(ids), 100):
            part = ids[i : i + 100]
            if await self.count_existing(user_id, part, db=db) == len(part):
                present.update(part)
            else:
                for one in part:
                    if await self.count_existing(user_id, [one], db=db):
                        present.add(one)
        return present

    async def health(self) -> Tuple[bool, Optional[str]]:
        return True, None

    async def aclose(self) -> None:
        return None


class PineconeStore(VectorStore):
    name = "pinecone"

    def __init__(self, *, api_key: str, index: str, dimensions: int, api_url: str, host: Optional[str] = None,
                 timeout: float = 30.0, transport=None, upsert_concurrency: int = 1, max_attempts: int = 1, metric: str = "cosine"):
        self._key, self.index, self.dimensions, self.metric = api_key, index, dimensions, metric
        self.upsert_concurrency, self.max_attempts = max(1, upsert_concurrency), max(1, max_attempts)
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
                index_metric = info.get("metric")
                if index_metric and index_metric.lower() != self.metric.lower():
                    raise errors.ServiceError(
                        errors.VECTOR_STORE_MISCONFIGURED,
                        f"Pinecone index metric {index_metric!r} does not match the expected {self.metric!r}; "
                        "vectors are unit length and ranked by cosine similarity.",
                        503,
                        retryable=True,
                    )
                host = info["host"]
            base = host if host.startswith("http") else f"https://{host}"
            self._data = httpx.AsyncClient(base_url=base, timeout=self._timeout, transport=self._transport)
        return self._data

    @staticmethod
    async def _backoff(attempt: int, retry_after: Optional[str] = None) -> None:
        """Exponential (0.5 s, 1 s, 2 s ...) plus a little jitter; a numeric Retry-After is honoured up to 5 s."""
        delay = 0.5 * 2 ** (attempt - 1)
        try:
            delay = max(delay, min(float(retry_after), 5.0)) if retry_after else delay
        except ValueError:
            pass
        await asyncio.sleep(delay + random.uniform(0, 0.25))

    async def _send(self, method: str, path: str, attempts: int = 1, **kwargs) -> httpx.Response:
        """One request, retried at most `attempts` times in total, and only for failures that can be
        transient: network errors, 429 and 5xx. Callers pass attempts > 1 only for idempotent calls
        (upsert / fetch / delete by deterministic id)."""
        client = await self._client()
        for attempt in range(1, attempts + 1):
            last = attempt == attempts
            try:
                r = await client.request(method, path, headers=self._headers(), **kwargs)
            except httpx.HTTPError:
                if last:
                    raise errors.vector_store_unavailable()
                log.warning("pinecone_retry", extra={"path": path, "attempt": attempt, "reason": "network"})
                await self._backoff(attempt)
                continue
            if r.is_success:
                return r
            log.warning("pinecone_http_error", extra={"status": r.status_code, "path": path, "attempt": attempt})
            if r.status_code in (401, 403):
                raise errors.ServiceError(errors.VECTOR_STORE_MISCONFIGURED, "Pinecone rejected the configured API key.", 503, retryable=True)
            if (r.status_code == 429 or r.status_code >= 500) and not last:
                await self._backoff(attempt, r.headers.get("retry-after"))
                continue
            raise errors.vector_store_unavailable()
        raise errors.vector_store_unavailable()  # unreachable: the last attempt returns or raises

    async def _post(self, path: str, body: dict, attempts: int = 1) -> dict:
        r = await self._send("POST", path, attempts, json=body)
        return r.json() if r.content else {}

    async def upsert(self, user_id, records, db=None):
        namespace = namespace_for(user_id)
        batches = [records[i : i + 100] for i in range(0, len(records), 100)]
        gate = asyncio.Semaphore(self.upsert_concurrency)
        failed = False

        async def send(batch):
            nonlocal failed
            async with gate:  # bounded: at most `upsert_concurrency` requests in flight
                if failed:
                    return  # a sibling request already failed for good: start nothing more
                try:
                    await self._post("/vectors/upsert", {
                        "namespace": namespace,
                        "vectors": [{"id": r.id, "values": r.values, "metadata": r.metadata} for r in batch],
                    }, attempts=self.max_attempts)
                except BaseException:
                    failed = True
                    raise

        async with timed("vector_upsert", store="pinecone"):
            tasks = [asyncio.create_task(send(b)) for b in batches]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                # One request failed for good: stop the rest. Anything that already landed is inside the
                # vector extent the ingestion recorded first, so its cleanup removes it.
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

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
                await self._post("/vectors/delete", {"namespace": namespace_for(user_id), "ids": ids[i : i + 1000]},
                                 attempts=self.max_attempts)

    async def count_existing(self, user_id, ids, db=None):
        found = 0
        for i in range(0, len(ids), 100):
            params = [("namespace", namespace_for(user_id))] + [("ids", x) for x in ids[i : i + 100]]
            r = await self._send("GET", "/vectors/fetch", self.max_attempts, params=params)
            found += len((r.json() or {}).get("vectors") or {})
        return found

    async def existing_ids(self, user_id, ids, db=None):
        present: set = set()
        for i in range(0, len(ids), 100):
            params = [("namespace", namespace_for(user_id))] + [("ids", x) for x in ids[i : i + 100]]
            r = await self._send("GET", "/vectors/fetch", self.max_attempts, params=params)
            present.update(((r.json() or {}).get("vectors") or {}).keys())
        return present

    async def health(self):
        try:
            info = await self.describe()
        except errors.ServiceError as exc:
            return False, exc.code
        if info.get("dimension") != self.dimensions:
            return False, f"DIMENSION_MISMATCH: index has {info.get('dimension')}, embedding model produces {self.dimensions}"
        if info.get("metric") and info["metric"].lower() != self.metric.lower():
            return False, f"METRIC_MISMATCH: index uses {info['metric']}, expected {self.metric}"
        ready = (info.get("status") or {}).get("ready", True)
        return bool(ready), None if ready else "INDEX_NOT_READY"


class LocalVectorStore(VectorStore):
    """Vectors live in rag_chunks.embedding; cosine similarity computed in Python."""

    name = "local"

    async def upsert(self, user_id, records, db: Any = None):
        by_id = {r.id: r.values for r in records}
        rows = (await db.scalars(select(Chunk).where(Chunk.id.in_(list(by_id))))).all()
        for row in rows:
            row.embedding = by_id[row.id]

    async def query(self, user_id, vector, top_k, document_ids, db: Any = None):
        rows = (await db.execute(
            select(Chunk.id, Chunk.embedding).where(Chunk.user_id == user_id, Chunk.document_id.in_(list(document_ids)))
        )).all()
        scored = []
        for cid, emb in rows:
            if emb:
                scored.append((cid, sum(a * b for a, b in zip(vector, emb)) / ((math.sqrt(sum(a * a for a in emb)) or 1.0))))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def delete(self, user_id, ids, db: Any = None):
        return None  # rows are deleted with the chunks

    async def existing_ids(self, user_id, ids, db: Any = None):
        rows = (await db.execute(select(Chunk.id, Chunk.embedding).where(Chunk.id.in_(list(ids)), Chunk.user_id == user_id))).all()
        return {cid for cid, emb in rows if emb}

    async def count_existing(self, user_id, ids, db: Any = None):
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
        upsert_concurrency=cfg.PINECONE_UPSERT_CONCURRENCY,
        max_attempts=cfg.PINECONE_MAX_ATTEMPTS,
        metric=cfg.PINECONE_METRIC,
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
