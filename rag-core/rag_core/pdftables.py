"""Table detection for text-layer PDFs (and a layout-based pass for OCR output).

Two kinds of table are detected, with very different reliability, and the report says which was used:

  lines  A table drawn with ruling lines (borders). pdfplumber follows the lines, so cell boundaries are exact.
         This is the reliable case.
  text   A borderless table: columns inferred from how text is aligned. This is heuristic. A candidate is accepted
         only if it is dense and regular (see `acceptable`); prose, lists and multi-column layouts usually fail that
         test, but nothing here can guarantee that a page of aligned text is, or is not, a table. Opt-in (`auto`).

Nothing is invented: a cell is exactly the text pdfplumber found in that position, empty cells stay empty, and a table
that fails the checks is left as the plain page text it came from. Rendering (header detection, one self-describing
line per row) is done by parsing.py with the same helpers CSV and DOCX tables use.
"""
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

_WS = re.compile(r"\s+")
_TEXT_SETTINGS = {"vertical_strategy": "text", "horizontal_strategy": "text", "min_words_vertical": 2, "min_words_horizontal": 2,
                  "snap_tolerance": 3, "join_tolerance": 3, "intersection_tolerance": 5}


@dataclass
class FoundTable:
    kind: str                     # lines | text
    bbox: Tuple[float, float, float, float]  # x0, top, x1, bottom in PDF points from the top-left
    rows: List[List[str]]
    page_height: float = 0.0
    continued: bool = False       # continues the table that ended the previous page
    header: Optional[List[str]] = None


@dataclass
class PageTables:
    tables: List[FoundTable] = field(default_factory=list)
    text_bands: List[Tuple[float, float]] = field(default_factory=list)


def clean_table(raw: List[List[Optional[str]]]) -> List[List[str]]:
    """Whitespace-normalised cells; fully empty rows and fully empty columns removed (borderless detection yields both)."""
    rows = [[_WS.sub(" ", (c or "")).strip() for c in r] for r in raw if r]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep = [j for j in range(width) if any(r[j] for r in rows)]
    return [[r[j] for j in keep] for r in rows]


def acceptable(rows: List[List[str]], kind: str) -> bool:
    """Is this candidate a table worth structuring? Bordered candidates need at least two rows and two columns.
    Borderless ones must also be dense and regular: at least 3 rows, most rows filling most columns, short cells."""
    if len(rows) < 2 or not rows[0] or len(rows[0]) < 2:
        return False
    if kind == "lines":
        return True
    ncols = len(rows[0])
    if len(rows) < 3 or ncols > 12:
        return False
    need = max(2, -(-ncols * 6 // 10))  # ceil(0.6 * ncols)
    dense = sum(1 for r in rows if sum(1 for c in r if c) >= need)
    cells = [c for r in rows for c in r if c]
    longest = max(len(c) for c in cells)
    mean = sum(len(c) for c in cells) / len(cells)
    return dense / len(rows) >= 0.8 and longest <= 80 and mean <= 30


def _find(page: Any, kind: str) -> List[FoundTable]:
    found = page.find_tables(_TEXT_SETTINGS) if kind == "text" else page.find_tables()
    out = []
    for t in found:
        rows = clean_table(t.extract())
        if acceptable(rows, kind):
            out.append(FoundTable(kind, tuple(t.bbox), rows, float(page.height)))
    out.sort(key=lambda f: f.bbox[1])
    # overlapping candidates (a text-strategy table inside a bordered one): keep the first by position
    kept: List[FoundTable] = []
    for f in out:
        if not kept or f.bbox[1] >= kept[-1].bbox[3] - 1:
            kept.append(f)
    return kept


def find_page_tables(page: Any, mode: str) -> List[FoundTable]:
    """mode: lines (bordered only) | auto (bordered, then borderless if none was found)."""
    tables = _find(page, "lines")
    if not tables and mode == "auto":
        tables = _find(page, "text")
    return tables


def split_page(page: Any, tables: List[FoundTable]) -> List[Tuple[str, Any]]:
    """The page as an ordered list of ("text", plumber_crop) and ("table", FoundTable), top to bottom."""
    parts: List[Tuple[str, Any]] = []
    top, width, height = 0.0, float(page.width), float(page.height)
    for t in tables:
        if t.bbox[1] - top > 2:
            parts.append(("text", page.crop((0, max(0.0, top), width, min(height, t.bbox[1])))))
        parts.append(("table", t))
        top = t.bbox[3]
    if height - top > 2:
        parts.append(("text", page.crop((0, max(0.0, top), width, height))))
    return parts


def link_continuations(per_page: List[List[FoundTable]], header_of: Callable[[List[str]], Optional[List[str]]]) -> None:
    """Mark tables that continue the previous page's last table, and resolve each table's column names.

    A table continues the previous one when it is the first table on its page, near the top, has the same number of
    columns as the previous page's last table (which ends near the bottom), and its first row is either a repeat of the
    header (dropped) or contains a numeric cell, i.e. is a record rather than a header (kept). Anything else starts a new table. If the
    previous table had no usable header, a continuation has none either: columns are never guessed."""
    prev: Optional[FoundTable] = None
    for tables in per_page:
        for idx, t in enumerate(tables):
            t.header = None
            if idx == 0 and prev is not None and len(t.rows[0]) == len(prev.rows[0]) and t.bbox[1] < 0.3 * t.page_height \
                    and prev.bbox[3] > 0.6 * prev.page_height:
                first = [c.strip().lower() for c in t.rows[0]]
                if prev.header and first == [c.strip().lower() for c in prev.header]:
                    t.continued, t.header = True, prev.header
                    t.rows = t.rows[1:]
                elif _looks_like_data(t.rows[0]):
                    t.continued, t.header = True, prev.header
            if not t.continued:
                guess = header_of(t.rows[0]) if len(t.rows) > 1 else None
                t.header = guess
                if guess:
                    t.rows = t.rows[1:]
        prev = tables[-1] if tables else None
        # a page without a table breaks the chain
        if not tables:
            prev = None


_NUMERIC_CELL = re.compile(r"^[-+]?[\d.,\s%€$£]*\d[\d.,\s%€$£]*$")


def _looks_like_data(row: List[str]) -> bool:
    """A header names columns; a row with any numeric cell is a record. Used only to decide that a table's first row on a
    new page is a continuation (not a new header)."""
    return any(_NUMERIC_CELL.match(c) for c in row if c)


class Budget:
    """Cooperative wall-clock limit across a document's pages."""

    def __init__(self, seconds: float):
        self.deadline = time.monotonic() + seconds

    def exhausted(self) -> bool:
        return time.monotonic() > self.deadline


# ---------------------------------------------------------------- OCR output (layout only)

_PIPE = re.compile(r"\s\|\s")


def ocr_table_blocks(text: str, min_rows: int = 3) -> List[Tuple[int, int, List[List[str]]]]:
    """Runs of >= min_rows consecutive OCR lines that split on ' | ' into the same number (>= 2) of cells.
    The OCR layer writes a ' | ' between boxes separated by a wide gap, so aligned columns survive as such runs. A row with a
    missing cell has a different cell count and ends the run: structure is never inferred across a gap."""
    lines = text.split("\n")
    blocks, start = [], None
    cells_prev: Optional[int] = None

    def close(end: int) -> None:
        nonlocal start
        if start is not None and end - start >= min_rows:
            blocks.append((start, end, [[c.strip() for c in _PIPE.split(l)] for l in lines[start:end]]))
        start = None

    for i, line in enumerate(lines):
        n = len(_PIPE.split(line)) if _PIPE.search(line) else 0
        if n >= 2 and (start is None or n == cells_prev):
            if start is None:
                start = i
            cells_prev = n
        else:
            close(i)
            if n >= 2:
                start, cells_prev = i, n
    close(len(lines))
    return blocks
