"""Client for Agent Core. The backend supplies authenticated context; Agent Core decides
retrieval, tools, planning and generation (via RAG Core and the LLM layer).

Contract (see API_CONTRACT.md, "Agent Core integration"):
  POST /api/v1/agent/invoke  -> JSON {content, citations[], metadata{}}
  POST /api/v1/agent/stream  -> text/event-stream with events listed in STREAM_EVENTS
"""
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from app.core.config import settings
from app.core.exceptions import AppError, ServiceUnavailable
from app.integrations.http import get_client, outbound_headers, timed

SERVICE = "agent_core"

# Events forwarded to the frontend. Anything else (e.g. internal reasoning) is dropped.
STREAM_EVENTS = {
    "agent.started",
    "retrieval.started",
    "retrieval.completed",
    "generation.started",
    "content.delta",
    "content.replace",
    "generation.completed",
    "error",
}

# Metadata keys safe to persist/expose. Never chain-of-thought or prompts.
SAFE_METADATA_KEYS = {
    "model",
    "retrieval_strategy",
    "retrieved_chunks",
    "retrieval_latency_ms",
    "rerank_latency_ms",
    "generation_latency_ms",
    "total_latency_ms",
    "token_usage",
    "citation_count",
    "confidence",
    "verification_status",
    "retrieval_scope",
    "answer_mode",
    "evidence_grade",
    "correction_attempts",
    "answer_replaced",
    "table_analysis",
    "followup",
    # Exact table-analysis results behind an answer (server-side only: never sent to the browser;
    # reused by later follow-ups in the same conversation). Size-capped in safe_metadata.
    "structured_result",
}
MAX_STRUCTURED_RESULT_BYTES = 64_000


# Agent Core error codes that are safe to surface to the browser unchanged.
PASSTHROUGH_CODES = {"RAG_SERVICE_UNAVAILABLE", "LLM_UNAVAILABLE", "LLM_TIMEOUT", "LLM_RATE_LIMITED", "GENERATION_FAILED"}


def _error_code(response: httpx.Response) -> Optional[str]:
    try:
        err = response.json().get("error")
        return err.get("code") if isinstance(err, dict) else None
    except ValueError:
        return None


class AgentServiceUnavailable(ServiceUnavailable):
    code = "AGENT_SERVICE_UNAVAILABLE"
    message = "The AI service is temporarily unavailable."


class AgentServiceError(AppError):
    status_code = 502
    code = "AGENT_SERVICE_ERROR"
    message = "The AI service could not complete this request."


@dataclass
class AgentRequest:
    user_id: uuid.UUID
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    message: str
    document_ids: List[uuid.UUID]
    history: List[Dict[str, Any]]  # {role, content[, citations, structured]}
    context: Dict[str, Any] = field(default_factory=dict)
    # The user's Knowledge Base: every ready document they own, [{"document_id", "filename"}].
    library_documents: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "user_id": str(self.user_id),
            "conversation_id": str(self.conversation_id),
            "message_id": str(self.message_id),
            "message": self.message,
            "document_ids": [str(d) for d in self.document_ids],
            "history": self.history,
            "context": self.context,
            "library_documents": self.library_documents,
        }


@dataclass
class AgentResult:
    content: str
    citations: List[dict]
    metadata: Dict[str, Any]


@dataclass
class StreamEvent:
    event: str
    data: Dict[str, Any]


def safe_metadata(metadata: Optional[dict]) -> Dict[str, Any]:
    out = {k: v for k, v in (metadata or {}).items() if k in SAFE_METADATA_KEYS}
    result = out.get("structured_result")
    if result is not None and (not isinstance(result, dict) or len(json.dumps(result, default=str)) > MAX_STRUCTURED_RESULT_BYTES):
        out.pop("structured_result")
    return out


def safe_citations(citations: Any) -> List[dict]:
    if not isinstance(citations, list):
        return []
    allowed = {"source_id", "document_id", "filename", "page", "chunk_id", "snippet"}
    out = []
    for c in citations[:50]:
        if not isinstance(c, dict):
            continue
        kept = {k: v for k, v in c.items() if k in allowed}
        sid = kept.get("source_id")
        if not (isinstance(sid, int) and not isinstance(sid, bool) and 1 <= sid <= 1000):
            kept.pop("source_id", None)  # never let a malformed marker number break message rendering
        out.append(kept)
    return out


def _decode_event(event: str, data_lines: List[str]) -> StreamEvent:
    raw = "\n".join(data_lines)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"delta": raw} if event == "content.delta" else {}
    return StreamEvent(event, data if isinstance(data, dict) else {})


async def _parse_sse(response: httpx.Response) -> AsyncIterator[StreamEvent]:
    event, data_lines = "message", []
    async for line in response.aiter_lines():
        if line == "":
            if data_lines:
                yield _decode_event(event, data_lines)
            event, data_lines = "message", []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield _decode_event(event, data_lines)


class AgentCoreClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.AGENT_CORE_URL

    def _client(self) -> httpx.AsyncClient:
        return get_client(SERVICE, self.base_url, settings.AGENT_CORE_TIMEOUT_SECONDS)

    @staticmethod
    def _timeout() -> httpx.Timeout:
        # Explicit per request: never inherit a pooled client's defaults. For streams the read
        # timeout bounds the silence between events (e.g. a slow follow-up rewrite).
        return httpx.Timeout(settings.AGENT_CORE_TIMEOUT_SECONDS, connect=5.0)

    async def invoke(self, req: AgentRequest) -> AgentResult:
        try:
            async with timed(SERVICE, "invoke"):
                response = await self._client().post("/api/v1/agent/invoke", json=req.to_json(), headers=outbound_headers(),
                                                    timeout=self._timeout())
        except httpx.HTTPError as exc:
            raise AgentServiceUnavailable() from exc
        if not response.is_success:
            code = _error_code(response)
            if code in PASSTHROUGH_CODES:
                raise AgentServiceUnavailable(code=code)
            if response.status_code >= 500 or response.status_code in (408, 429):
                raise AgentServiceUnavailable()
            raise AgentServiceError()
        body = response.json()
        return AgentResult(
            content=str(body.get("content", "")),
            citations=safe_citations(body.get("citations")),
            metadata=safe_metadata(body.get("metadata")),
        )

    async def stream(self, req: AgentRequest) -> AsyncIterator[StreamEvent]:
        """Yields only whitelisted events. Raises AgentServiceUnavailable/AgentServiceError."""
        try:
            async with timed(SERVICE, "stream"):
                async with self._client().stream(
                    "POST",
                    "/api/v1/agent/stream",
                    json=req.to_json(),
                    headers={**outbound_headers(), "Accept": "text/event-stream"},
                    timeout=self._timeout(),
                ) as response:
                    if response.status_code >= 500 or response.status_code in (408, 429):
                        raise AgentServiceUnavailable()
                    if response.status_code >= 400:
                        raise AgentServiceError()
                    async for ev in _parse_sse(response):
                        if ev.event in STREAM_EVENTS:
                            yield ev
        except httpx.HTTPError as exc:
            raise AgentServiceUnavailable() from exc


def get_agent_client() -> AgentCoreClient:
    """FastAPI dependency (overridden in tests)."""
    return AgentCoreClient()
