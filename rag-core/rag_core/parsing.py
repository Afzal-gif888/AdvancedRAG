"""Turn raw bytes into page-aware text. Pure functions, run in a worker thread."""
import csv
import io
import re
from dataclasses import dataclass
from typing import List, Optional

from rag_core import errors

TEXT_TYPES = {"text/plain", "text/markdown", "text/csv"}
PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
CSV = "text/csv"
IMAGE_TYPES = {"image/png", "image/jpeg"}


@dataclass
class Page:
    number: Optional[int]  # 1-based for PDFs; None for formats without pages
    text: str


_WS = re.compile(r"[ \t\f\v]+")
_BLANKS = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = "\n".join(_WS.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip()


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _row_text(header: Optional[List[str]], cells: List[str]) -> str:
    """One self-describing line per table/CSV row: `Battery life | 14 months` becomes
    `Property: Battery life | Value: 14 months` when a header exists, so a chunk taken from the
    middle of a long table still says what each value means."""
    if not header:
        return " | ".join(c for c in cells if c)
    parts = []
    for i, value in enumerate(cells):
        if value:
            name = header[i] if i < len(header) and header[i] else f"column {i + 1}"
            parts.append(f"{name}: {value}")
    return " | ".join(parts)


_NUMBER = re.compile(r"^[-+]?[\d.,\s%€$£]+$")


def _usable_header(cells: List[str]) -> Optional[List[str]]:
    """The first row is used as column names only if it looks like one: >= 2 distinct,
    non-empty names that are not all numbers (an all-numeric first row is data)."""
    names = [c.strip() for c in cells]
    if len(names) < 2 or any(not n for n in names) or len(set(names)) != len(names):
        return None
    if all(_NUMBER.match(n) for n in names):
        return None
    return names


def _docx_table(table) -> List[str]:
    rows = []
    for row in table.rows:
        cells, seen = [], set()
        for cell in row.cells:  # horizontally merged cells repeat: keep each underlying cell once
            if id(cell._tc) in seen:
                continue
            seen.add(id(cell._tc))
            cells.append(" ".join(cell.text.split()))
        rows.append(cells)
    if not rows:
        return []
    header = _usable_header(rows[0]) if len(rows) > 1 else None
    body = rows[1:] if header else rows
    lines = [_row_text(header, cells) for cells in body]
    return ([" | ".join(header)] if header else []) + [line for line in lines if line]


def _docx_text(data: bytes) -> str:
    """Paragraphs and tables in document order (a table stays under its heading)."""
    import docx
    from docx.table import Table

    document = docx.Document(io.BytesIO(data))
    parts: List[str] = []
    for block in document.iter_inner_content():
        if isinstance(block, Table):
            parts.extend(_docx_table(block))
        else:
            parts.append(block.text)
    return "\n".join(parts)


def _csv_rows(text: str) -> Optional[List[List[str]]]:
    """Non-empty rows of a delimited table, or None when the text does not look like one."""
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    try:
        rows = [[c.strip() for c in row] for row in csv.reader(io.StringIO(text), dialect)]
    except csv.Error:
        return None
    rows = [r for r in rows if any(r)]
    if len(rows) < 2 or max(len(r) for r in rows) < 2:
        return None
    return rows


def _csv_text(text: str) -> str:
    """Rows rendered as `column: value | ...` lines. Falls back to the raw text when the file
    does not look like a delimited table."""
    rows = _csv_rows(text)
    if rows is None:
        return text
    header = _usable_header(rows[0])
    body = rows[1:] if header else rows
    return "\n".join(line for line in (_row_text(header, r) for r in body) if line)


@dataclass
class Table:
    columns: List[str]
    rows: List[List[str]]


def parse_table(data: bytes, max_rows: int, max_columns: int) -> Optional[Table]:
    """The complete CSV as a table (for exact, whole-dataset analysis), with the same dialect and
    header detection as the text rendering. None when it is not a table or exceeds the limits."""
    rows = _csv_rows(_decode(data))
    if rows is None:
        return None
    header = _usable_header(rows[0])
    body = rows[1:] if header else rows
    width = max(len(header or []), max(len(r) for r in body))
    if not body or len(body) > max_rows or width > max_columns:
        return None
    columns = [(header[i] if header and i < len(header) else f"column {i + 1}") for i in range(width)]
    return Table(columns, [(r + [""] * (width - len(r)))[:width] for r in body])


def parse(data: bytes, mime_type: str) -> List[Page]:
    if mime_type in IMAGE_TYPES:
        raise errors.unsupported("Images cannot be searched yet: no OCR is configured.")
    try:
        if mime_type == PDF:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise errors.ServiceError(errors.PARSE_FAILED, "Password-protected PDFs are not supported.", 422)
            pages = [Page(i + 1, normalize(p.extract_text() or "")) for i, p in enumerate(reader.pages)]
        elif mime_type == DOCX:
            pages = [Page(None, normalize(_docx_text(data)))]
        elif mime_type == CSV:
            pages = [Page(None, normalize(_csv_text(_decode(data))))]
        elif mime_type in TEXT_TYPES:
            pages = [Page(None, normalize(_decode(data)))]
        else:
            raise errors.unsupported("This file type is not supported.")
    except errors.ServiceError:
        raise
    except Exception as exc:  # malformed files: never retry, never leak parser internals
        raise errors.parse_failed() from exc
    pages = [p for p in pages if p.text]
    if not pages:
        raise errors.no_text()
    return pages
