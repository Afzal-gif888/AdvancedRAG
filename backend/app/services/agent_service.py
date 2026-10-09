"""Runs a chat turn through Agent Core and persists the outcome.

Assistant message lifecycle:  pending -> streaming -> completed | failed | cancelled
- Partial output is flushed to the DB while streaming, so a crash leaves recoverable content
  (stale pending/streaming rows are later marked failed/INTERRUPTED by the reconciler).
- On failure, any partial output is kept but the message is `failed` with a safe error code,
  so it is visibly incomplete and can be retried. Nothing is ever fabricated.
"""
import asyncio
import json
import logging
import time
from typing import AsyncIterator, Optional

from app.core.config import settings
from app.core.exceptions import AppError
from app.db.models import message as msg_status
from app.db.session import session_scope
from app.integrations.agent_core import (
    AgentCoreClient,
    AgentServiceError,
    safe_citations,
    safe_metadata,
)
from app.services import message_service
from app.services.message_service import PreparedTurn

logger = logging.getLogger("advancerag.agent")

SAFE_ERROR_MESSAGES = {
    "AGENT_SERVICE_UNAVAILABLE": "The AI service is temporarily unavailable.",
    "AGENT_SERVICE_ERROR": "The AI service could not complete this request.",
    "RAG_SERVICE_UNAVAILABLE": "Document search is temporarily unavailable.",
    "LLM_UNAVAILABLE": "The language model is temporarily unavailable.",
    "LLM_TIMEOUT": "The language model took too long to respond.",
    "LLM_RATE_LIMITED": "The language model is busy. Please try again shortly.",
    "EMPTY_RESPONSE": "The AI service returned an empty response.",
    "GENERATION_FAILED": "The response could not be generated.",
    "INTERRUPTED": "The response was interrupted.",
}
# Error codes from Agent Core that may be shown as-is; anything else becomes GENERATION_FAILED.
_AGENT_CODES = {"RAG_SERVICE_UNAVAILABLE", "LLM_UNAVAILABLE", "LLM_TIMEOUT", "LLM_RATE_LIMITED", "GENERATION_FAILED"}

# Fields from intermediate events that are safe to pass to the browser.
_STATUS_FIELDS = {"retrieval.completed": {"retrieved_chunks"}}


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def safe_agent_code(code) -> str:
    return code if code in _AGENT_CODES else "GENERATION_FAILED"


async def _persist(message_id, **kwargs):
    # Shielded so a client disconnect cannot leave the message half-written.
    async def _run():
        async with session_scope() as db:
            return await message_service.finalize_assistant(db, message_id, **kwargs)

    return await asyncio.shield(_run())


async def _flush_partial(message_id, content: str) -> None:
    async with session_scope() as db:
        await message_service.update_streaming(db, message_id, content)


async def run_turn(agent: AgentCoreClient, turn: PreparedTurn):
    """Non-streaming: returns the finalized assistant message or raises a safe AppError."""
    started = time.perf_counter()
    try:
        result = await agent.invoke(turn.agent_request)
    except AppError as exc:
        await _persist(turn.assistant_message.id, status=msg_status.FAILED, error_code=exc.code)
        raise
    if not result.content.strip():
        await _persist(turn.assistant_message.id, status=msg_status.FAILED, error_code="EMPTY_RESPONSE")
        raise AgentServiceError("The AI service returned an empty response.", code="EMPTY_RESPONSE")
    metadata = {**result.metadata, "backend_latency_ms": round((time.perf_counter() - started) * 1000, 2)}
    return await _persist(
        turn.assistant_message.id,
        status=msg_status.COMPLETED,
        content=result.content,
        citations=result.citations,
        metadata=metadata,
    )


async def stream_turn(agent: AgentCoreClient, turn: PreparedTurn) -> AsyncIterator[str]:
    """SSE lifecycle: message.created -> [agent/retrieval/generation status] -> content.delta* ->
    [content.replace: the verified answer replaces the draft] ->
    message.completed | error. Unknown/internal events from Agent Core are never forwarded."""
    started = time.perf_counter()
    assistant_id = turn.assistant_message.id
    parts = []
    citations, metadata = [], {}
    error_code: Optional[str] = None
    streaming_marked = False
    last_flush = time.monotonic()

    yield sse(
        "message.created",
        {
            "user_message": message_service.to_response(turn.user_message).model_dump(mode="json"),
            "assistant_message": message_service.to_response(turn.assistant_message).model_dump(mode="json"),
        },
    )

    try:
        async for ev in agent.stream(turn.agent_request):
            if ev.event == "content.delta":
                delta = ev.data.get("delta")
                if isinstance(delta, str) and delta:
                    parts.append(delta)
                    yield sse("content.delta", {"delta": delta})
                    now = time.monotonic()
                    if not streaming_marked or now - last_flush >= settings.STREAM_FLUSH_SECONDS:
                        await _flush_partial(assistant_id, "".join(parts))
                        streaming_marked, last_flush = True, now
            elif ev.event == "content.replace":
                # The verified/corrected answer replaces the streamed draft; only the final text is stored.
                final = ev.data.get("content")
                if isinstance(final, str) and final.strip() and len(final) <= 200_000:
                    parts = [final]
                    yield sse("content.replace", {"content": final})
                    await _flush_partial(assistant_id, final)
                    streaming_marked, last_flush = True, time.monotonic()
            elif ev.event == "generation.completed":
                citations = safe_citations(ev.data.get("citations"))
                metadata = safe_metadata(ev.data.get("metadata"))
                yield sse("generation.completed", {})
            elif ev.event == "error":
                error_code = safe_agent_code(ev.data.get("code"))
                break
            else:
                allowed = _STATUS_FIELDS.get(ev.event, set())
                yield sse(ev.event, {k: v for k, v in ev.data.items() if k in allowed})
    except AppError as exc:
        error_code = exc.code
    except (asyncio.CancelledError, GeneratorExit):
        # Client went away: keep what was generated, mark it cancelled, stop.
        await _persist(assistant_id, status=msg_status.CANCELLED, content="".join(parts), error_code="CANCELLED")
        raise
    except Exception:
        logger.exception("stream_turn_failed")
        error_code = "GENERATION_FAILED"

    content = "".join(parts)
    if error_code is None and not content.strip():
        error_code = "EMPTY_RESPONSE"

    if error_code:
        # Partial output (if any) is kept but the message is marked failed + retryable.
        saved = await _persist(assistant_id, status=msg_status.FAILED, content=content, error_code=error_code)
        payload = {"code": error_code, "message": SAFE_ERROR_MESSAGES.get(error_code, SAFE_ERROR_MESSAGES["GENERATION_FAILED"])}
        if saved is not None:
            payload["assistant_message"] = message_service.to_response(saved).model_dump(mode="json")
        yield sse("error", payload)
        return

    metadata["backend_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    saved = await _persist(assistant_id, status=msg_status.COMPLETED, content=content, citations=citations, metadata=metadata)
    yield sse(
        "message.completed",
        {"assistant_message": message_service.to_response(saved).model_dump(mode="json") if saved else None},
    )
