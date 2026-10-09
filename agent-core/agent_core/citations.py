"""Citations: map the model's source markers back to the chunks RAG Core actually returned.

- Markers as models write them: [2], [1, 3], [1][4] and full-width 【2】.
- A number that was not supplied as a source is ignored: citations can never be invented.
- Each citation carries the source number the model cited (`source_id`, i.e. the n in [n]), RAG
  Core's metadata only (document, filename, page when known, chunk id) and a short snippet.
"""
import re
from typing import Any, Dict, List, Set

from agent_core.schemas import RetrievedChunk

_CITATION = re.compile(r"[\[【]\s*(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\s*[\]】]")
SNIPPET_CHARS = 300

# verification_status values (a key the backend already persists)
CITED = "cited"                     # grounded answer that cites supplied sources
UNCITED = "uncited"                 # sources were supplied but the answer cites none of them
NO_EVIDENCE = "no_relevant_documents"
NOT_APPLICABLE = "not_applicable"   # no retrieval for this turn (general / small talk)
# Self-correcting RAG outcomes (grading.py)
VERIFIED = "verified"               # every claim checked against the cited passages and supported
REVISED = "revised"                 # unsupported claims were removed; the revised answer replaced the draft
INSUFFICIENT = "insufficient_evidence"  # the Knowledge Base does not support an answer: abstained
UNVERIFIED = "unverified"           # cited answer, but the verification check could not run
COMPUTED = "computed"               # exact figures computed over a whole stored table
REFORMATTED = "reformatted"         # an earlier answer of this conversation reused (no new facts)


def extract_citations(answer: str, sources: List[RetrievedChunk]) -> List[Dict[str, Any]]:
    by_number = {c.source_id: c for c in sources}
    seen, out = set(), []
    for match in _CITATION.finditer(answer):
        for n in (int(x) for x in re.findall(r"\d+", match.group(1))):
            chunk = by_number.get(n)
            if chunk is None or n in seen:
                continue
            seen.add(n)
            out.append({
                "source_id": chunk.source_id,  # the [n] the model wrote, so the UI can link marker -> source
                "document_id": str(chunk.document_id),
                "filename": chunk.filename,
                "page": chunk.page,
                "chunk_id": chunk.chunk_id,
                "snippet": chunk.text[:SNIPPET_CHARS],
            })
    return out


def strip_invalid_markers(answer: str, valid: Set[int]) -> str:
    """Remove citation numbers that do not name a supplied source ([7] -> "", [1, 7] -> [1])."""
    def fix(match: "re.Match[str]") -> str:
        start = match.start()
        if start and (answer[start - 1].isalnum() or answer[start - 1] == "_"):
            return match.group(0)  # indexing in code (array[0]), not a citation
        numbers = [n for n in re.findall(r"\d+", match.group(1)) if int(n) in valid]
        return f"[{', '.join(numbers)}]" if numbers else ""
    out = _CITATION.sub(fix, answer)
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", out) if out != answer else answer


def verification_status(retrieved: bool, sources: List[RetrievedChunk], citations: List[Dict[str, Any]]) -> str:
    if not retrieved:
        return NOT_APPLICABLE
    if not sources:
        return NO_EVIDENCE
    return CITED if citations else UNCITED
