"""Recursive, page-aware chunking with overlap (character based, separator preferring)."""
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from rag_core.parsing import Page

SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


@dataclass
class TextChunk:
    index: int
    page: Optional[int]
    text: str


def _split(text: str, size: int, separators: Sequence[str]) -> List[str]:
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(separators):
        if sep in text:
            parts = text.split(sep)
            pieces: List[str] = []
            for j, part in enumerate(parts):
                piece = part + (sep if j < len(parts) - 1 else "")
                if len(piece) > size:
                    pieces.extend(_split(piece, size, separators[i + 1:]))
                elif piece:
                    pieces.append(piece)
            return pieces
    return [text[k : k + size] for k in range(0, len(text), size)]


def _merge(pieces: List[str], size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > size:
            chunks.append(current.strip())
            # Carry the tail forward for context continuity, starting at a line boundary when the
            # tail spans one (a table/CSV row or list item is never cut in half), else a word.
            tail = current[-overlap:] if overlap else ""
            if tail and "\n" in tail.rstrip("\n"):
                tail = tail[tail.index("\n") + 1:]
            elif tail and " " in tail:
                tail = tail[tail.index(" ") + 1:]
            current = tail
        current += piece
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


def fit_to_token_limit(chunks: List[TextChunk], count_tokens: Callable[[str], int], max_tokens: int) -> List[TextChunk]:
    """Split any chunk the embedding model would silently truncate (dense numbers, codes and
    tables tokenize to far more tokens per character than prose). Re-numbers chunk indexes."""

    def fit(text: str) -> List[str]:
        if len(text) < 64 or count_tokens(text) <= max_tokens:
            return [text]
        half = max(len(text) // 2, 32)
        pieces = [p.strip() for p in _merge(_split(text, half, SEPARATORS), half, 0) if p.strip()]
        if len(pieces) < 2:
            pieces = [text[:half], text[half:]]
        return [part for piece in pieces for part in fit(piece)]

    out: List[TextChunk] = []
    for chunk in chunks:
        for text in fit(chunk.text):
            out.append(TextChunk(index=len(out), page=chunk.page, text=text))
    return out


def chunk_pages(pages: List[Page], size: int, overlap: int) -> List[TextChunk]:
    out: List[TextChunk] = []
    for page in pages:
        for text in _merge(_split(page.text, size, SEPARATORS), size, overlap):
            out.append(TextChunk(index=len(out), page=page.number, text=text))
    return out
