"""Client for RAG Core. RAG Core owns parsing, chunking, embeddings, Pinecone, hybrid search,
RRF, reranking and context selection — none of that lives in the backend.

Contract (see API_CONTRACT.md, "RAG Core integration"):
  POST   /api/v1/rag/ingestions                   multipart: file + metadata JSON -> 202 (runs in the background)
  GET    /api/v1/rag/ingestions/{document_id}?user_id=...   attempt status
  DELETE /api/v1/rag/documents/{document_id}?user_id=...   also cancels a running ingestion
"""
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

import httpx

from app.core.config import settings
from app.integrations.http import get_client, outbound_headers, timed

SERVICE = "rag_core"


class RAGCoreError(Exception):
    """Raised for failed RAG Core calls. `retryable` drives job retries; `code` is safe to store."""

    def __init__(self, code: str, retryable: bool):
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class RAGCoreClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.RAG_CORE_URL

    def _client(self) -> httpx.AsyncClient:
        return get_client(SERVICE, self.base_url, settings.RAG_CORE_TIMEOUT_SECONDS)

    @staticmethod
    def _timeout() -> httpx.Timeout:
        return httpx.Timeout(settings.RAG_CORE_TIMEOUT_SECONDS, connect=5.0)

    @staticmethod
    def _check(response: httpx.Response, ok_statuses=()) -> None:
        if response.status_code in ok_statuses or response.is_success:
            return
        retryable = response.status_code >= 500 or response.status_code in (408, 429)
        code = "RAG_SERVICE_UNAVAILABLE" if retryable else "RAG_REJECTED_DOCUMENT"
        try:
            body = response.json()
            err = body.get("error") if isinstance(body, dict) else None
            if isinstance(err, dict) and isinstance(err.get("code"), str) and not retryable:
                code = err["code"][:64]
        except ValueError:
            pass
        raise RAGCoreError(code, retryable)

    async def start_ingestion(
        self,
        *,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
        filename: str,
        mime_type: str,
        content: Union[bytes, "os.PathLike[str]"],
    ) -> Dict[str, Any]:
        """Hands the file to RAG Core and returns at once with the attempt status. If an ingestion of
        this document is already running, RAG Core joins it instead of starting another one."""
        metadata = {"document_id": str(document_id), "user_id": str(user_id), "filename": filename, "mime_type": mime_type}
        size = len(content) if isinstance(content, (bytes, bytearray)) else Path(content).stat().st_size
        handle = None
        try:
            # A path is streamed from disk by httpx in small blocks: the file is never held in memory here.
            handle = None if isinstance(content, (bytes, bytearray)) else open(content, "rb")
            async with timed(SERVICE, "ingest_start", {"size_bytes": size}):
                response = await self._client().post(
                    "/api/v1/rag/ingestions",
                    data={"metadata": json.dumps(metadata)},
                    files={"file": (filename, handle if handle is not None else content, mime_type)},
                    headers=outbound_headers(),
                    timeout=self._timeout(),
                )
        except httpx.HTTPError as exc:
            raise RAGCoreError("RAG_SERVICE_UNAVAILABLE", retryable=True) from exc
        except OSError as exc:  # the local copy vanished between being prepared and being sent
            raise RAGCoreError("FILE_MISSING", retryable=False) from exc
        finally:
            if handle is not None:
                handle.close()
        self._check(response)
        return response.json()

    async def ingestion_status(self, *, document_id: uuid.UUID, user_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Latest attempt for the document, or None when RAG Core has none."""
        try:
            response = await self._client().get(
                f"/api/v1/rag/ingestions/{document_id}",
                params={"user_id": str(user_id)},
                headers=outbound_headers(),
                timeout=self._timeout(),
            )
        except httpx.HTTPError as exc:
            raise RAGCoreError("RAG_SERVICE_UNAVAILABLE", retryable=True) from exc
        if response.status_code == 404:
            return None
        self._check(response)
        return response.json()

    async def delete_document(self, *, document_id: uuid.UUID, user_id: uuid.UUID) -> None:
        try:
            async with timed(SERVICE, "delete"):
                response = await self._client().delete(
                    f"/api/v1/rag/documents/{document_id}",
                    params={"user_id": str(user_id)},
                    headers=outbound_headers(),
                    timeout=self._timeout(),
                )
        except httpx.HTTPError as exc:
            raise RAGCoreError("RAG_SERVICE_UNAVAILABLE", retryable=True) from exc
        # Already gone counts as success.
        self._check(response, ok_statuses=(404,))


def get_rag_client() -> RAGCoreClient:
    return RAGCoreClient()
