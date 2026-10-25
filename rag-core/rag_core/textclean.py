"""Conservative text hygiene for extracted text (PDF, OCR, plain text).

Every rule here is deliberately narrow, because a "clean-up" that rewrites text can corrupt exactly the
content users search for (identifiers, URLs, code, numbers, legal wording). Rules never touch:

- text inside Markdown code fences,
- tokens that look like URLs, e-mail addresses, paths or identifiers (``/ _ @ ` `` or digits),
- lines that look like list items, table rows or headings (when re-flowing paragraphs).

Each rule has regression tests in ``tests/test_textclean.py`` that show both what it fixes and what it
must leave alone.
"""
import math
import re
import unicodedata
from typing import List, Tuple

LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    "Ĳ": "IJ", "ĳ": "ij", "Œ": "OE", "œ": "oe",
}
# Characters that carry no meaning in extracted text: soft hyphen, zero-width space/joiners, BOM, word joiner.
_INVISIBLE = dict.fromkeys(map(ord, "­​‌‍⁠﻿"), None)
_SPACES = {ord(c): " " for c in "       "}

_FENCE = re.compile(r"(^```.*?^```[ \t]*$)", re.S | re.M)


def clean_chars(text: str) -> str:
    """Ligatures to letters, invisible characters removed, exotic spaces to spaces, NFC form.

    NFC (not NFKC) on purpose: NFKC would also rewrite superscripts, fractions and full-width forms.
    """
    for lig, plain in LIGATURES.items():
        if lig in text:
            text = text.replace(lig, plain)
    text = text.translate(_INVISIBLE).translate(_SPACES)
    text = text.replace(" ", "\n").replace(" ", "\n\n")
    return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------- line-end hyphenation

_HYPHEN_BREAK = re.compile(r"(?P<left>[A-Za-z]{2,})-[ \t]*\n[ \t]*(?P<right>[a-z][A-Za-z]*)")


def _looks_like_identifier(context: str) -> bool:
    return any(ch in context for ch in "/_@`\\:") or any(ch.isdigit() for ch in context)


def dehyphenate(text: str) -> Tuple[str, int]:
    """Join words split by a line-end hyphen ("inter-\nnational" -> "international").

    The hyphen is kept (and only the newline removed) when the hyphenated compound also occurs elsewhere
    in the same text, which is the best evidence available that it is a real compound ("well-known").
    Words that sit inside something identifier-like (URL, e-mail, path, snake_case, anything with digits)
    are never touched. Returns (text, number_of_joins).
    """
    if "-" not in text:
        return text, 0
    joins = 0

    def fixer(part: str):
        def fix(m: "re.Match[str]") -> str:
            nonlocal joins
            left, right = m.group("left"), m.group("right")
            before = part[max(0, m.start() - 12):m.start()].split()
            after = part[m.end():m.end() + 40].split()
            context = (before[-1] if before else "") + left + right + (after[0] if after else "")
            if _looks_like_identifier(context):
                return m.group(0)
            compound = f"{left}-{right}"
            joins += 1
            if re.search(rf"(?<![\w-]){re.escape(compound)}(?![\w-])", text, re.I):
                return compound  # a real compound that merely wrapped: keep the hyphen, drop the line break
            return left + right

        return fix

    parts = _FENCE.split(text)
    out = [part if i % 2 else _HYPHEN_BREAK.sub(fixer(part), part) for i, part in enumerate(parts)]
    return "".join(out), joins


# ---------------------------------------------------------------- repeated page headers / footers

def _signature(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.strip().lower()))


def remove_repeated_margins(pages: List[str], min_pages: int = 3, share: float = 0.6, max_len: int = 120, min_lines: int = 5) -> Tuple[List[str], int]:
    """Drop lines that repeat in the top/bottom two lines of most pages (running headers, footers, page numbers).

    A line is removed only if its digit-normalised form appears at a page margin on at least
    ``max(min_pages, ceil(share * n_pages))`` pages and is short. Body text is never touched.
    """
    n = len(pages)
    if n < min_pages:
        return pages, 0
    margin_sigs: List[set] = []
    split_pages = [p.split("\n") for p in pages]
    for lines in split_pages:
        idx = [i for i, ln in enumerate(lines) if ln.strip()]
        # Pages with fewer than `min_lines` lines are never stripped: there the "margins" would be the body.
        edge = set(idx[:2]) | set(idx[-2:]) if len(idx) >= min_lines else set()
        margin_sigs.append({_signature(lines[i]) for i in edge if len(lines[i].strip()) <= max_len})
    counts: dict = {}
    for sigs in margin_sigs:
        for s in sigs:
            counts[s] = counts.get(s, 0) + 1
    threshold = max(min_pages, math.ceil(share * n))
    repeated = {s for s, c in counts.items() if c >= threshold and s}
    if not repeated:
        return pages, 0
    removed = 0
    cleaned = []
    for lines in split_pages:
        idx = [i for i, ln in enumerate(lines) if ln.strip()]
        edge = set(idx[:2]) | set(idx[-2:]) if len(idx) >= min_lines else set()
        keep = []
        for i, ln in enumerate(lines):
            if i in edge and len(ln.strip()) <= max_len and _signature(ln) in repeated:
                removed += 1
                continue
            keep.append(ln)
        cleaned.append("\n".join(keep).strip("\n"))
    return cleaned, removed


# ---------------------------------------------------------------- re-flowing hard-wrapped paragraphs

_LIST_ITEM = re.compile(r"^\s*(?:[-*+•·▪◦‣]|\(?\d{1,3}[.)]|\(?[A-Za-z][.)])\s+")
_SENTENCE_END = (".", ":", ";", "!", "?", "”", '"', ")", "]")


def _structured(line: str) -> bool:
    return bool(_LIST_ITEM.match(line)) or "|" in line or line.rstrip().endswith(("{", "}", ";")) or line.startswith("#")


def unwrap_lines(text: str, min_line_len: int = 35) -> str:
    """Re-flow lines that were wrapped by the page layout into one line per paragraph.

    Joins a line to the next only when the line is long enough to be a wrapped line (not a heading or a
    short label), does not end a sentence, and the next line continues in lower case. List items, table
    rows (``|``) and code-like lines are never joined.
    """
    out_blocks = []
    for part_i, part in enumerate(_FENCE.split(text)):
        if part_i % 2:
            out_blocks.append(part)
            continue
        paragraphs = re.split(r"\n{2,}", part)
        rebuilt = []
        for para in paragraphs:
            lines = para.split("\n")
            merged = [lines[0]] if lines else []
            for nxt in lines[1:]:
                prev = merged[-1]
                joinable = (
                    len(prev.strip()) >= min_line_len
                    and nxt[:1].islower()
                    and not prev.rstrip().endswith(_SENTENCE_END)
                    and not _structured(prev)
                    and not _structured(nxt)
                )
                if joinable:
                    merged[-1] = prev.rstrip() + " " + nxt.lstrip()
                else:
                    merged.append(nxt)
            rebuilt.append("\n".join(merged))
        out_blocks.append("\n\n".join(rebuilt))
    return "".join(out_blocks)
