"""Typed contracts at Agent Core's three boundaries.

- AgentRequest:   Backend -> Agent Core (unchanged wire format; ids are validated as UUIDs)
- RetrieveResult: RAG Core -> Agent Core (validated; nothing is trusted blindly)
- GenerateResult: LLM service -> Agent Core (validated)
"""
import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from agent_core.errors import GENERATION_FAILED, RAG_INVALID_RESPONSE, AgentError, llm_error, rag_error


# ---------------------------------------------------------------- Backend -> Agent Core

class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=200_000)
    # Assistant turns only (backend, from the stored message): the citations the answer carried and,
    # for exact table analysis, the computed results. Used to reuse an earlier answer (followup.py).
    citations: List[Dict[str, Any]] = Field(default_factory=list, max_length=50)
    structured: Optional[Dict[str, Any]] = None


class LibraryDocument(BaseModel):
    """One ready document of the user's Knowledge Base."""
    document_id: uuid.UUID
    filename: str = Field(min_length=1, max_length=255)
    recent: bool = False  # accepted for compatibility; scope no longer depends on it


class AgentRequest(BaseModel):
    user_id: uuid.UUID
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    message: str = Field(min_length=1, max_length=100_000)
    # Backend-verified: owned by user_id and status ready. RAG Core re-checks ownership.
    document_ids: List[uuid.UUID] = Field(default_factory=list, max_length=500)
    history: List[HistoryMessage] = Field(default_factory=list, max_length=500)
    context: Dict[str, Any] = Field(default_factory=dict)
    # The user's Knowledge Base: every ready document they own (backend-verified like
    # document_ids). Searched for every non-small-talk question; a question naming files narrows
    # the search to them (library.py). Retrieval ranks chunks; nothing is injected wholesale.
    library_documents: List[LibraryDocument] = Field(default_factory=list, max_length=500)

    @field_validator("document_ids")
    @classmethod
    def _dedupe(cls, v: List[uuid.UUID]) -> List[uuid.UUID]:
        return list(dict.fromkeys(v))

    @model_validator(mode="after")
    def _library_excludes_attached(self) -> "AgentRequest":
        attached, seen, kept = set(self.document_ids), set(), []
        for d in self.library_documents:
            if d.document_id not in attached and d.document_id not in seen:
                seen.add(d.document_id)
                kept.append(d)
        self.library_documents = kept
        return self

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v

    @property
    def pending_document_count(self) -> int:
        try:
            return max(int((self.context or {}).get("pending_document_count") or 0), 0)
        except (TypeError, ValueError):
            return 0


# ---------------------------------------------------------------- RAG Core -> Agent Core

class RetrievedChunk(BaseModel):
    source_id: int = Field(ge=1)
    chunk_id: str = Field(min_length=1, max_length=200)
    document_id: uuid.UUID
    filename: str = Field(min_length=1, max_length=255)
    page: Optional[int] = Field(default=None, ge=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    score: float


class RetrieveResult(BaseModel):
    chunks: List[RetrievedChunk] = Field(default_factory=list, max_length=50)
    context: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


def validate_retrieval(raw: Any, requested_document_ids: List[uuid.UUID]) -> RetrieveResult:
    """Parse RAG Core's response and enforce the invariants Agent Core relies on:
    sources numbered 1..n in rank order, unique chunk ids, chunk id bound to its document, and
    every chunk belonging to a document that was actually requested (defence in depth on top of
    RAG Core's own per-user scoping). Any violation rejects the whole result."""
    try:
        result = RetrieveResult.model_validate(raw)
    except ValidationError as exc:
        raise rag_error(RAG_INVALID_RESPONSE) from exc
    allowed = set(requested_document_ids)
    seen_chunks = set()
    for position, chunk in enumerate(result.chunks, start=1):
        if (chunk.source_id != position or chunk.chunk_id in seen_chunks or chunk.document_id not in allowed
                or not chunk.chunk_id.startswith(f"{chunk.document_id}:")):
            raise rag_error(RAG_INVALID_RESPONSE)
        seen_chunks.add(chunk.chunk_id)
    return result


# ---------------------------------------------------------------- LLM service -> Agent Core

class GenerateResult(BaseModel):
    content: str
    model: Optional[str] = None
    finish_reason: Optional[str] = None


def validate_generation(raw: Any) -> GenerateResult:
    try:
        return GenerateResult.model_validate(raw)
    except ValidationError as exc:
        raise llm_error("LLM_INVALID_RESPONSE") from exc


__all__ = ["AgentRequest", "HistoryMessage", "LibraryDocument", "RetrievedChunk", "RetrieveResult", "GenerateResult",
           "validate_retrieval", "validate_generation", "AgentError", "GENERATION_FAILED"]
