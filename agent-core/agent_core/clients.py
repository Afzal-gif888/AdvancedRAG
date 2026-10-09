"""HTTP clients for RAG Core and the LLM service — the only way Agent Core reaches them.

Both use bounded timeouts, the internal service token and X-Request-ID propagation
(advancerag_common.service.outbound_headers). Responses are validated before use; every failure
becomes an AgentError with a stable public code and a precise internal `reason`.

Agent Core never retries: the LLM service already applies bounded, provider-aware retries, and
RAG retrieval is cheap enough that the user can simply resend. This keeps a failing dependency
from turning into a retry storm.
"""
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from advancerag_common.observability import timed
from advancerag_common.service import outbound_headers
from advancerag_common.sse import Event, parse_events
from agent_core.config import Settings
from agent_core.errors import (  # noqa: F401  (re-exported for callers and tests)
    GENERATION_FAILED,
    LLM_RATE_LIMITED,
    LLM_TIMEOUT,
    LLM_UNAVAILABLE,
    RAG_HTTP_ERROR,
    RAG_INVALID_RESPONSE,
    RAG_SERVICE_UNAVAILABLE,
    RAG_TIMEOUT,
    RAG_UNREACHABLE,
    AgentError,
    llm_error,
    rag_error,
)
from agent_core.schemas import GenerateResult, RetrieveResult, validate_generation, validate_retrieval

LLM_STREAM_INTERRUPTED = "LLM_STREAM_INTERRUPTED"


def _error_code(response: httpx.Response) -> Optional[str]:
    try:
        err = response.json().get("error")
        return err.get("code") if isinstance(err, dict) else None
    except Exception:
        return None


class RAGClient:
    def __init__(self, cfg: Settings, transport: Optional[httpx.AsyncBaseTransport] = None):
        self._token = cfg.internal_token
        self._client = httpx.AsyncClient(
            base_url=cfg.RAG_CORE_URL,
            timeout=httpx.Timeout(cfg.RAG_TIMEOUT_SECONDS, connect=cfg.CONNECT_TIMEOUT_SECONDS),
            transport=transport,
        )

    async def aclose(self):
        await self._client.aclose()

    async def retrieve(self, *, user_id: uuid.UUID, query: str, document_ids: List[uuid.UUID], top_k: int) -> RetrieveResult:
        body = {"user_id": str(user_id), "query": query, "document_ids": [str(d) for d in document_ids], "top_k": top_k}
        try:
            async with timed("rag_retrieve"):
                r = await self._client.post("/api/v1/rag/retrieve", json=body, headers=outbound_headers(self._token))
        except httpx.TimeoutException as exc:
            raise rag_error(RAG_TIMEOUT) from exc
        except httpx.HTTPError as exc:
            raise rag_error(RAG_UNREACHABLE) from exc
        if not r.is_success:
            raise rag_error(f"{RAG_HTTP_ERROR}_{r.status_code}:{_error_code(r) or 'unknown'}")
        try:
            raw = r.json()
        except ValueError as exc:
            raise rag_error(RAG_INVALID_RESPONSE) from exc
        return validate_retrieval(raw, document_ids)

    async def _post_json(self, path: str, body: dict, ok_404: bool = False) -> Optional[dict]:
        try:
            async with timed("rag_tables"):
                r = await self._client.post(path, json=body, headers=outbound_headers(self._token))
        except httpx.TimeoutException as exc:
            raise rag_error(RAG_TIMEOUT) from exc
        except httpx.HTTPError as exc:
            raise rag_error(RAG_UNREACHABLE) from exc
        if ok_404 and r.status_code in (404, 422):
            return None
        if not r.is_success:
            raise rag_error(f"{RAG_HTTP_ERROR}_{r.status_code}:{_error_code(r) or 'unknown'}")
        try:
            return r.json()
        except ValueError as exc:
            raise rag_error(RAG_INVALID_RESPONSE) from exc

    async def describe_tables(self, *, user_id: uuid.UUID, document_ids: List[uuid.UUID]) -> List[dict]:
        data = await self._post_json("/api/v1/rag/tables/describe",
                                     {"user_id": str(user_id), "document_ids": [str(d) for d in document_ids][:50]}, ok_404=True)
        tables = (data or {}).get("tables")
        allowed = {str(d) for d in document_ids}
        return [t for t in tables if isinstance(t, dict) and t.get("document_id") in allowed] if isinstance(tables, list) else []

    async def query_table(self, *, user_id: uuid.UUID, document_id: uuid.UUID, queries: List[dict]) -> Optional[dict]:
        """Computed results, or None when RAG Core rejects the spec (422) / has no such table (404)."""
        return await self._post_json("/api/v1/rag/tables/query",
                                     {"user_id": str(user_id), "document_id": str(document_id), "queries": queries}, ok_404=True)

    async def ready(self) -> bool:
        try:
            r = await self._client.get("/ready", headers=outbound_headers(self._token), timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False


class LLMClient:
    def __init__(self, cfg: Settings, transport: Optional[httpx.AsyncBaseTransport] = None):
        self._token = cfg.internal_token
        self._client = httpx.AsyncClient(
            base_url=cfg.LLM_SERVICE_URL,
            timeout=httpx.Timeout(cfg.LLM_TIMEOUT_SECONDS, connect=cfg.CONNECT_TIMEOUT_SECONDS),
            transport=transport,
        )

    async def aclose(self):
        await self._client.aclose()

    @staticmethod
    def _body(messages, model, temperature, max_tokens) -> Dict[str, Any]:
        body = {"messages": messages, "model": model, "temperature": temperature, "max_tokens": max_tokens}
        return {k: v for k, v in body.items() if v is not None}

    async def generate(self, messages: List[Dict[str, str]], *, model: Optional[str] = None, temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> GenerateResult:
        try:
            async with timed("llm_generate"):
                r = await self._client.post("/v1/generate", json=self._body(messages, model, temperature, max_tokens),
                                            headers=outbound_headers(self._token))
        except httpx.TimeoutException as exc:
            raise llm_error(LLM_TIMEOUT) from exc
        except httpx.HTTPError as exc:
            raise llm_error("llm_service_unreachable") from exc
        if not r.is_success:
            raise llm_error(_error_code(r) or f"llm_service_http_{r.status_code}")
        try:
            raw = r.json()
        except ValueError as exc:
            raise llm_error("LLM_INVALID_RESPONSE") from exc
        return validate_generation(raw)

    async def stream(self, messages: List[Dict[str, str]], *, model: Optional[str] = None, temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None) -> AsyncIterator[Event]:
        """Yields `token` events, then exactly one `done`. A stream that ends without `done` is an
        interrupted generation, never a complete answer. Closing this generator (client
        disconnect / cancellation) closes the upstream HTTP stream."""
        try:
            async with self._client.stream("POST", "/v1/stream", json=self._body(messages, model, temperature, max_tokens),
                                           headers=outbound_headers(self._token)) as r:
                if not r.is_success:
                    await r.aread()
                    raise llm_error(_error_code(r) or f"llm_service_http_{r.status_code}")
                done = False
                async for ev in parse_events(r):
                    if ev.event == "error":
                        raise llm_error(ev.data.get("code") if isinstance(ev.data.get("code"), str) else None)
                    if ev.event == "token":
                        if not isinstance(ev.data.get("delta"), str):
                            raise llm_error("LLM_INVALID_RESPONSE")
                    elif ev.event == "done":
                        done = True
                    else:
                        continue  # unknown event types are ignored
                    yield ev
                    if done:
                        return
                raise llm_error(LLM_STREAM_INTERRUPTED)
        except httpx.TimeoutException as exc:
            raise llm_error(LLM_TIMEOUT) from exc
        except httpx.HTTPError as exc:
            raise llm_error("llm_service_unreachable") from exc

    async def ready(self) -> bool:
        try:
            r = await self._client.get("/ready", headers=outbound_headers(self._token), timeout=8.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False
