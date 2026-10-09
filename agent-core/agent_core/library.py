"""Search scope inside the user's Knowledge Base, from what the question explicitly says.

Deterministic (no I/O, no model call). The candidates are what the backend sent: the user's own
ready documents (ownership and status verified by the backend; RAG Core re-checks ownership).

  named    the question names a file ("what does pricing-2025.csv say about ...") -> only it
  typed    "the csv", "my pdfs", "the spreadsheet" -> only files of that type (all of them)
  generic  "my documents", "my Knowledge Base", "the file I uploaded" -> the whole Knowledge Base
  none     no reference -> the whole Knowledge Base

Documents are never chosen because a filename merely shares a word with the question, and the
user is never asked to pick a document: when several documents are in scope, retrieval and
reranking find the passages that answer the question, and the answer step handles evidence that
differs between documents (prompts.GROUNDING_RULES).
"""
import re
import uuid
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# Determiner/possessive context keeps general questions ("what is a PDF?") out of retrieval.
_DET = r"(?:my|our|the|this|that|these|those|uploaded|attached|your)"
_GAP = r"(?:[\w'-]+\s+){0,2}?"
_TYPES = {
    "csv": r"(?:csvs?|spreadsheets?|excel(?: files?| sheets?)?)",
    "pdf": r"pdfs?",
    "docx": r"(?:docx|word (?:documents?|files?|docs?))",
    "md": r"(?:markdown(?: files?)?)",
    "txt": r"(?:text files?|txt files?)",
}
_TYPE_RES = {ext: re.compile(rf"\b{_DET}\s+{_GAP}{pat}\b|\b{pat}\s+(?:file|document)s?\s+(?:i|we)\s+(?:uploaded|attached)\b", re.I)
             for ext, pat in _TYPES.items()}
_GENERIC_NOUN = r"(?:documents?|docs|files?|uploads?|attachments?|datasets?|data|reports?|manuals?|datasheets?|notes|sheets?|records|knowledge base|library)"
_GENERIC_RES = (
    re.compile(rf"\b(?:my|our|uploaded|attached|these|those)\s+{_GAP}{_GENERIC_NOUN}\b", re.I),
    re.compile(rf"\b(?:the|this|that)\s+{_GAP}(?:documents?|docs|files?|uploads?|attachments?|datasets?|reports?|manuals?|datasheets?)\b", re.I),
    re.compile(r"\b(?:i|we)\s+(?:have\s+)?(?:uploaded|attached|shared|sent)\b", re.I),
    re.compile(r"\b(?:in|from|according to|based on)\s+(?:my|our|the)\s+(?:library|knowledge base|uploads?)\b", re.I),
)
# Words too generic to identify a file by name.
_FILLER = set("""
a an the this that these those my our your i we you me it its is are was were be to of in on at for from by with and or
please can could would will should do does did tell show give me us about what whats which how
analyze analyse analysis summarize summarise summary describe explain review read look check open use go through
uploaded upload attached attach shared sent have has file files document documents doc docs data dataset datasets
csv csvs pdf pdfs docx spreadsheet spreadsheets sheet sheets report reports notes manual manuals datasheet datasheets
library knowledge base contents content say says inside earlier before previously just now again one
""".split())


@dataclass(frozen=True)
class Candidate:
    document_id: uuid.UUID
    filename: str
    recent: bool = False

    @property
    def extension(self) -> str:
        return self.filename.rsplit(".", 1)[-1].lower() if "." in self.filename else ""


@dataclass(frozen=True)
class LibrarySelection:
    kind: str = "none"                                   # named | typed | generic | none
    document_ids: Tuple[uuid.UUID, ...] = ()
    missing_type: Optional[str] = None                   # referenced a type the library lacks
    referenced: bool = False                             # the question refers to the user's documents


def _variants(filename: str) -> List[str]:
    name = filename.lower().strip()
    stem = name.rsplit(".", 1)[0] if "." in name else name
    out = {name, stem, re.sub(r"[-_.]+", " ", stem).strip()}
    # Stems too short or too generic would match ordinary words ("notes", "data").
    return [v for v in out if len(v) >= 4 and v not in _FILLER]


def _named(message: str, candidates: Sequence[Candidate]) -> List[Candidate]:
    text = message.lower()
    hits = []
    for c in candidates:
        for v in _variants(c.filename):
            if re.search(rf"(?<![\w-]){re.escape(v)}(?![\w-])", text):
                hits.append(c)
                break
    return hits


def select(message: str, candidates: Sequence[Candidate]) -> LibrarySelection:
    named = _named(message, candidates)
    if named:
        return LibrarySelection("named", tuple(dict.fromkeys(c.document_id for c in named)), referenced=True)

    types = [ext for ext, rx in _TYPE_RES.items() if rx.search(message)]
    if types:
        pool = [c for c in candidates if c.extension in types]
        if not pool:
            return LibrarySelection("typed", referenced=True, missing_type=types[0])
        return LibrarySelection("typed", tuple(c.document_id for c in pool), referenced=True)
    if any(rx.search(message) for rx in _GENERIC_RES):
        # "my documents", "my Knowledge Base", "the file I uploaded": a reference to the user's
        # data, not to one particular file. Whole Knowledge Base; retrieval ranks the passages.
        return LibrarySelection("generic", tuple(c.document_id for c in candidates), referenced=True)
    return LibrarySelection()
