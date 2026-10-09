"""Deterministic request classification and retrieval scope (no I/O, no model calls).

Product contract (docs/agent-core-architecture.md):
- The user's Knowledge Base (their ready, indexed documents; sent by the backend with ownership
  and status verified) is the source for document-grounded answers. Nothing has to be attached.
- Every non-small-talk question is searched against the Knowledge Base; whether the evidence is
  relevant and sufficient is decided after retrieval (grading.py), so a general question with no
  relevant passages is still answered generally, and a document question without sufficient
  evidence gets an explicit "not in your Knowledge Base" answer.
- A question that names files, or a file type, of the Knowledge Base searches only those.
  Several documents in scope is never a reason to ask the user which one: retrieval and
  reranking pick the passages, and differing evidence is handled when answering.
- Quantitative questions over a CSV in the Knowledge Base are computed over the whole table
  (structured.py) instead of from retrieved chunks.
- Pure small talk never triggers retrieval.
"""
import re
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from agent_core import library, structured
from agent_core.schemas import AgentRequest

_SMALLTALK = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|thx|ok|okay|cool|great|bye|good (morning|evening|night))[\s!.?]*$", re.I
)


class RequestType(str, Enum):
    SMALL_TALK = "small_talk"
    GENERAL = "general"                    # empty Knowledge Base (or referenced files it lacks)
    DOCUMENT_QUESTION = "document"         # Knowledge Base search, first turn
    FOLLOW_UP = "follow_up"                # Knowledge Base search, conversation history present
    STRUCTURED = "structured"              # exact statistics over a stored CSV table


class RetrievalDecision(str, Enum):
    REQUIRED = "required"
    NOT_REQUIRED = "not_required"


@dataclass(frozen=True)
class Classification:
    request_type: RequestType
    retrieval: RetrievalDecision
    reason: str
    document_ids: Tuple[uuid.UUID, ...] = ()       # exactly what may be searched
    scope: str = "none"                            # knowledge_base | named | none
    referenced: bool = False                       # the question explicitly refers to the user's documents
    missing_type: Optional[str] = None             # referenced a document type the Knowledge Base lacks
    table_ids: Tuple[uuid.UUID, ...] = ()          # CSVs in scope for exact analysis (STRUCTURED)

    @property
    def retrieve(self) -> bool:
        return self.retrieval is RetrievalDecision.REQUIRED


def is_small_talk(message: str) -> bool:
    return bool(_SMALLTALK.match(message))


def classify(req: AgentRequest) -> Classification:
    if is_small_talk(req.message):
        return Classification(RequestType.SMALL_TALK, RetrievalDecision.NOT_REQUIRED, "small_talk")
    candidates = [library.Candidate(d.document_id, d.filename) for d in req.library_documents]
    selection = library.select(req.message, candidates)
    explicit = list(req.document_ids)  # older clients; part of the user's verified documents

    if not candidates and not explicit:
        if req.pending_document_count:
            reason = "documents_still_processing"
        elif selection.referenced:
            reason = "referenced_documents_not_in_knowledge_base"
        else:
            reason = "no_documents_attached"
        return Classification(RequestType.GENERAL, RetrievalDecision.NOT_REQUIRED, reason,
                              missing_type=selection.missing_type, referenced=selection.referenced)
    if selection.referenced and not selection.document_ids and not (selection.kind == "generic" and explicit):
        return Classification(RequestType.GENERAL, RetrievalDecision.NOT_REQUIRED, "referenced_documents_not_in_knowledge_base",
                              missing_type=selection.missing_type, referenced=True)

    if selection.kind in ("named", "typed") and selection.document_ids:
        ids: List[uuid.UUID] = list(dict.fromkeys([*explicit, *selection.document_ids]))
        scope, reason = "named", f"knowledge_base_{selection.kind}_reference"
    else:
        ids = list(dict.fromkeys([*explicit, *(c.document_id for c in candidates)]))
        scope, reason = "knowledge_base", "knowledge_base_search"

    if structured.is_quantitative(req.message):
        names = {c.document_id: c.filename for c in candidates}
        csv_ids = tuple(i for i in ids if names.get(i, "").lower().endswith(".csv"))
        if csv_ids:
            # document_ids stays the whole scope: text retrieval is the fallback when no table answers.
            return Classification(RequestType.STRUCTURED, RetrievalDecision.REQUIRED, "quantitative_question_over_csv",
                                  tuple(ids), scope, selection.referenced, table_ids=csv_ids)

    rtype = RequestType.FOLLOW_UP if req.history else RequestType.DOCUMENT_QUESTION
    return Classification(rtype, RetrievalDecision.REQUIRED, reason, tuple(ids), scope, selection.referenced)
