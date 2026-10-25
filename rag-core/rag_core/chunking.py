"""Chunking strategies (character based, page-aware, separator preferring).

- ``chunk_pages``            recursive: the frozen baseline (CHUNKING_STRATEGY=recursive, the default).
- ``chunk_pages_structured`` heading-aware: never crosses a heading, keeps fenced code blocks whole when
                             they fit, records the heading path with every chunk.
- ``chunk_pages_semantic``   groups sentences by embedding similarity (a breakpoint where adjacent
                             sentences stop being similar), inside heading sections, then enforces the size
                             limit. Opt-in because it costs embedding calls.

All strategies keep chunks inside one page, number chunks sequentially across the document, and are
followed by ``fit_to_token_limit`` in the pipeline (nothing is silently truncated by the model).
"""
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from rag_core.parsing import Page

SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")
_FENCE = re.compile(r"(^```.*?^```[ \t]*$)", re.S | re.M)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


@dataclass
class TextChunk:
    index: int
    page: Optional[int]
    text: str
    section: Optional[str] = None  # heading path, e.g. "Travel > Hotels" (only where the source has headings)
    method: str = "native"  # how the page text was obtained: native | ocr


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
            out.append(TextChunk(index=len(out), page=chunk.page, text=text, section=chunk.section, method=chunk.method))
    return out


def chunk_pages(pages: List[Page], size: int, overlap: int) -> List[TextChunk]:
    out: List[TextChunk] = []
    for page in pages:
        for text in _merge(_split(page.text, size, SEPARATORS), size, overlap):
            out.append(TextChunk(index=len(out), page=page.number, text=text, method=page.method))
    return out


# ---------------------------------------------------------------- heading-aware

def split_sections(page: Page, min_chars: int = 150) -> List[Tuple[List[str], str]]:
    """Cut a page at its headings. Returns [(heading path, section text)] in order; the text of a section
    starts with its own heading line. A section shorter than ``min_chars`` (a heading directly followed by
    a sub-heading) is merged into the next section so no tiny chunk is produced."""
    heads = sorted(page.headings)
    if not heads:
        return [([], page.text)]
    bounds = [h[0] for h in heads] + [len(page.text)]
    sections: List[Tuple[List[str], str]] = []
    if heads[0][0] > 0 and page.text[: heads[0][0]].strip():
        sections.append(([], page.text[: heads[0][0]]))
    stack: List[Tuple[int, str]] = []
    for (offset, level, title), end in zip(heads, bounds[1:]):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        sections.append(([t for _, t in stack], page.text[offset:end]))
    merged: List[Tuple[List[str], str]] = []
    carry = ""
    for i, (path, text) in enumerate(sections):
        text = carry + text
        carry = ""
        # Only a parent heading with (almost) no body of its own is folded into the sub-section that follows
        # it. A short section followed by a sibling or a higher-level heading stays a section of its own:
        # folding it into the next one would give its text the wrong heading path.
        nxt = sections[i + 1][0] if i + 1 < len(sections) else None
        if len(text.strip()) < min_chars and nxt is not None and path and len(nxt) > len(path) and nxt[: len(path)] == path:
            carry = text.rstrip("\n") + "\n"
            continue
        merged.append((path, text))
    return merged


def _split_protecting_code(text: str, size: int) -> List[str]:
    """Recursive split, but a fenced code block that fits in one chunk is never cut."""
    pieces: List[str] = []
    for i, part in enumerate(_FENCE.split(text)):
        if not part.strip():
            continue
        if i % 2 and len(part) <= size:
            pieces.append(part.rstrip() + "\n")
        else:
            pieces.extend(_split(part, size, SEPARATORS))
    return pieces


def chunk_pages_structured(pages: List[Page], size: int, overlap: int) -> List[TextChunk]:
    out: List[TextChunk] = []
    for page in pages:
        for path, text in split_sections(page):
            section = " > ".join(path) or None
            for piece in _merge(_split_protecting_code(text, size), size, overlap):
                out.append(TextChunk(index=len(out), page=page.number, text=piece, section=section, method=page.method))
    return out


# ---------------------------------------------------------------- semantic

EmbedFn = Callable[[List[str]], Awaitable[List[List[float]]]]


def split_sentences(text: str) -> List[str]:
    """Lossless sentence split (sentence end + whitespace, or a blank line); separators stay attached."""
    parts, last = [], 0
    for m in _SENTENCE.finditer(text):
        parts.append(text[last:m.end()])
        last = m.end()
    if last < len(text):
        parts.append(text[last:])
    return [p for p in parts if p.strip()]


def _percentile(values: List[float], pct: int) -> float:
    s = sorted(values)
    k = max(0, min(len(s) - 1, -(-pct * len(s) // 100) - 1))  # nearest rank
    return s[k]


def semantic_groups(sentences: List[str], vectors: List[List[float]], percentile: int, min_chars: int, max_chars: int) -> List[List[str]]:
    """Group consecutive sentences. A boundary is placed after sentence i when the cosine distance to
    sentence i+1 is above the ``percentile`` of all adjacent distances (a topic shift) and the group is
    already at least ``min_chars``; a group is also closed before it would exceed ``max_chars``.
    Deterministic for given vectors (unit vectors: cosine = dot product)."""
    if len(sentences) <= 1:
        return [sentences] if sentences else []
    dist = [1.0 - sum(a * b for a, b in zip(vectors[i], vectors[i + 1])) for i in range(len(sentences) - 1)]
    threshold = _percentile(dist, percentile)
    groups: List[List[str]] = []
    cur: List[str] = []
    cur_len = 0
    for i, s in enumerate(sentences):
        if cur and cur_len + len(s) > max_chars:
            groups.append(cur)
            cur, cur_len = [], 0
        cur.append(s)
        cur_len += len(s)
        if i < len(dist) and dist[i] > threshold and cur_len >= min_chars:
            groups.append(cur)
            cur, cur_len = [], 0
    if cur:
        groups.append(cur)
    return groups


async def chunk_pages_semantic(
    pages: List[Page], embed: EmbedFn, size: int, overlap: int, percentile: int = 90, min_chars: int = 300, max_sentences: int = 4000,
    buffer: int = 1,
) -> Tuple[List[TextChunk], Dict[str, object]]:
    """Semantic chunking inside heading sections. Returns (chunks, info). If the document has more than
    ``max_sentences`` sentences the structure-aware chunker is used instead (info['fallback'])."""
    plan: List[Tuple[Page, str, List[str]]] = []
    total = 0
    for page in pages:
        for path, text in split_sections(page):
            sents = split_sentences(text)
            total += len(sents)
            plan.append((page, " > ".join(path), sents))
    info: Dict[str, object] = {"sentences": total, "embedding_texts": 0, "fallback": None}
    if total > max_sentences:
        info["fallback"] = "too_many_sentences"
        return chunk_pages_structured(pages, size, overlap), info

    out: List[TextChunk] = []
    for page, section, sents in plan:
        if not sents:
            continue
        # embed each sentence together with its neighbours (smooths single odd sentences)
        windows = ["".join(sents[max(0, i - buffer): i + buffer + 1]) for i in range(len(sents))]
        vectors: List[List[float]] = []
        for i in range(0, len(windows), 128):
            vectors.extend(await embed(windows[i:i + 128]))
        info["embedding_texts"] = int(info["embedding_texts"]) + len(windows)  # type: ignore[arg-type]
        for group in semantic_groups(sents, vectors, percentile, min_chars, size):
            text = "".join(group).strip()
            if not text:
                continue
            for piece in _merge(_split(text, size, SEPARATORS), size, 0):  # a single huge sentence is still bounded
                out.append(TextChunk(index=len(out), page=page.number, text=piece, section=section or None, method=page.method))
    return out, info
