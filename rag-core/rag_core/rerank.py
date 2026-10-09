"""Rerankers: Pinecone hosted inference (cross-encoder) or none (keep RRF order)."""
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

import httpx

from advancerag_common.observability import timed
from rag_core.config import Settings
from rag_core.vectorstore import PINECONE_API_VERSION

log = logging.getLogger("advancerag.rag.rerank")


class Reranker(ABC):
    name: str

    @abstractmethod
    async def rerank(self, query: str, docs: List[Tuple[str, str]], top_n: int) -> List[Tuple[str, float]]:
        """docs: [(id, text)] -> [(id, score)] best first. Raise on failure."""

    async def aclose(self) -> None:
        return None


class NoopReranker(Reranker):
    name = "none"

    async def rerank(self, query, docs, top_n):
        return [(doc_id, 1.0 / (i + 1)) for i, (doc_id, _) in enumerate(docs[:top_n])]


class PineconeReranker(Reranker):
    name = "pinecone"

    def __init__(self, *, api_key: str, model: str, api_url: str, timeout: float = 30.0, transport=None):
        self._key, self.model = api_key, model
        self._client = httpx.AsyncClient(base_url=api_url.rstrip("/"), timeout=timeout, transport=transport)

    async def aclose(self):
        await self._client.aclose()

    async def rerank(self, query, docs, top_n):
        async with timed("rerank", provider="pinecone"):
            r = await self._client.post(
                "/rerank",
                headers={"Api-Key": self._key, "X-Pinecone-API-Version": PINECONE_API_VERSION},
                json={
                    "model": self.model,
                    "query": query,
                    "documents": [{"id": doc_id, "text": text} for doc_id, text in docs],
                    "top_n": top_n,
                    "return_documents": False,
                    "parameters": {"truncate": "END"},
                },
            )
            r.raise_for_status()
            data = r.json()["data"]
        return [(docs[item["index"]][0], float(item["score"])) for item in data]


def build_reranker(cfg: Settings) -> Reranker:
    if cfg.RERANKER == "pinecone" and cfg.PINECONE_API_KEY:
        return PineconeReranker(api_key=cfg.PINECONE_API_KEY.get_secret_value(), model=cfg.RERANK_MODEL, api_url=cfg.PINECONE_API_URL, timeout=cfg.HTTP_TIMEOUT_SECONDS)
    return NoopReranker()
