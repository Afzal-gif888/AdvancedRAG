"""Exact analysis of a CSV stored in the user's Knowledge Base (whole dataset, not retrieved chunks).

Security model: no SQL, no expressions, no code. A query is a small declarative spec — one of a
closed set of operations over named columns, with at most MAX_FILTERS simple filters — validated
against the table's own columns and executed in plain Python over the stored rows. Row/column
counts are bounded at ingestion (parsing.parse_table), and results are bounded by `limit`.
Ownership is enforced by the caller (main.py) before a table is loaded.
"""
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from advancerag_common.service import ServiceError

OPERATIONS = ("count", "distribution", "crosstab", "stats", "top_terms")
FILTER_OPS = ("eq", "ne", "contains", "not_contains", "gt", "gte", "lt", "lte", "in")
MAX_FILTERS = 5
MAX_LIMIT = 50
_NUM = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")
_WORDS = re.compile(r"[a-z][a-z'-]{2,}")
_STOP = set("""
the and for are but not you all any can had her was one our out has have from they this that with what when where
which who will would there their them then than too very just into about over under also its it's been were your
more most some such only own same other each few both how why because while being does did doing yet nor off again
""".split())


def invalid(message: str) -> ServiceError:
    return ServiceError("TABLE_QUERY_INVALID", message, 422)


def to_number(value: str) -> Optional[float]:
    v = value.strip().replace(" ", "")
    if v.endswith("%"):
        v = v[:-1]
    if not _NUM.match(v):
        return None
    try:
        n = float(v.replace(",", ""))
    except ValueError:
        return None
    return n if math.isfinite(n) else None


def column_types(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> Dict[str, str]:
    """numeric when >= 90 % of the non-empty values parse as numbers, else text."""
    out = {}
    for i, name in enumerate(columns):
        values = [r[i] for r in rows if r[i].strip()]
        numeric = sum(1 for v in values if to_number(v) is not None)
        out[name] = "numeric" if values and numeric / len(values) >= 0.9 else "text"
    return out


def describe(columns: Sequence[str], rows: Sequence[Sequence[str]], examples: int = 8) -> List[Dict[str, Any]]:
    types = column_types(columns, rows)
    out = []
    for i, name in enumerate(columns):
        counts = Counter(r[i] for r in rows if r[i].strip())
        col = {"name": name, "type": types[name], "distinct": len(counts), "missing": sum(1 for r in rows if not r[i].strip())}
        if types[name] == "text":
            col["examples"] = [v[:60] for v, _ in counts.most_common(examples)]
        else:
            nums = [n for n in (to_number(r[i]) for r in rows) if n is not None]
            if nums:
                col.update(min=min(nums), max=max(nums))
        out.append(col)
    return out


@dataclass
class Filter:
    column: str
    op: str
    value: Union[str, float, List[str]]


def _match(cell: str, f: Filter) -> bool:
    if f.op in ("eq", "ne"):
        hit = cell.strip().casefold() == str(f.value).strip().casefold()
        return hit if f.op == "eq" else not hit
    if f.op in ("contains", "not_contains"):
        hit = str(f.value).casefold() in cell.casefold()
        return hit if f.op == "contains" else not hit
    if f.op == "in":
        return cell.strip().casefold() in {str(v).strip().casefold() for v in f.value}  # type: ignore[union-attr]
    n = to_number(cell)
    if n is None:
        return False
    target = float(f.value)  # validated numeric
    return {"gt": n > target, "gte": n >= target, "lt": n < target, "lte": n <= target}[f.op]


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0


def run(columns: Sequence[str], rows: Sequence[Sequence[str]], *, operation: str, column: Optional[str] = None,
        by: Optional[str] = None, filters: Sequence[Filter] = (), limit: int = 20) -> Dict[str, Any]:
    if operation not in OPERATIONS:
        raise invalid(f"Unknown operation {operation!r}.")
    index = {c: i for i, c in enumerate(columns)}
    lower = {c.casefold(): c for c in columns}

    def col(name: Optional[str], role: str) -> int:
        if not name:
            raise invalid(f"'{role}' is required for {operation}.")
        real = name if name in index else lower.get(name.casefold())
        if real is None:
            raise invalid(f"Unknown column {name!r}. Columns: {', '.join(columns)}.")
        return index[real]

    if len(filters) > MAX_FILTERS:
        raise invalid(f"At most {MAX_FILTERS} filters.")
    checks = []
    for f in filters:
        if f.op not in FILTER_OPS:
            raise invalid(f"Unknown filter operator {f.op!r}.")
        if f.op in ("gt", "gte", "lt", "lte") and (isinstance(f.value, list) or to_number(str(f.value)) is None):
            raise invalid(f"Filter {f.op} needs a number.")
        if f.op == "in" and not isinstance(f.value, list):
            raise invalid("Filter 'in' needs a list.")
        if f.op in ("gt", "gte", "lt", "lte"):
            f = Filter(f.column, f.op, to_number(str(f.value)))
        checks.append((col(f.column, "filter column"), f))
    limit = max(1, min(int(limit), MAX_LIMIT))
    matched = [r for r in rows if all(_match(r[i], f) for i, f in checks)]
    base = {"operation": operation, "rows_total": len(rows), "rows_matched": len(matched)}

    if operation == "count":
        return {**base, "result": {"count": len(matched)}}

    if operation == "distribution":
        i = col(column, "column")
        counts = Counter(r[i].strip() or "(empty)" for r in matched)
        top = counts.most_common(limit)
        return {**base, "column": columns[i], "result": {
            "distinct": len(counts),
            "values": [{"value": v, "count": c, "percent": _pct(c, len(matched))} for v, c in top],
            "other_count": len(matched) - sum(c for _, c in top),
        }}

    if operation == "crosstab":
        i, g = col(column, "column"), col(by, "by")
        groups: Dict[str, Counter] = {}
        for r in matched:
            groups.setdefault(r[g].strip() or "(empty)", Counter())[r[i].strip() or "(empty)"] += 1
        ordered = sorted(groups.items(), key=lambda kv: -sum(kv[1].values()))[:limit]
        return {**base, "column": columns[i], "by": columns[g], "result": {"groups": [
            {"group": name, "total": sum(c.values()),
             "values": [{"value": v, "count": n, "percent": _pct(n, sum(c.values()))} for v, n in c.most_common(limit)]}
            for name, c in ordered]}}

    if operation == "stats":
        i = col(column, "column")
        nums = [n for n in (to_number(r[i]) for r in matched) if n is not None]
        if not nums:
            return {**base, "column": columns[i], "result": {"count_numeric": 0, "missing": len(matched)}}
        return {**base, "column": columns[i], "result": {
            "count_numeric": len(nums), "missing": len(matched) - len(nums),
            "min": min(nums), "max": max(nums), "sum": round(sum(nums), 6),
            "mean": round(statistics.fmean(nums), 6), "median": statistics.median(nums),
        }}

    # top_terms: word frequency in a text column (an approximation of "most frequent topics").
    i = col(column, "column")
    terms: Counter = Counter()
    for r in matched:
        terms.update(w for w in _WORDS.findall(r[i].casefold()) if w not in _STOP)
    exact = Counter(r[i].strip() for r in matched if r[i].strip())
    return {**base, "column": columns[i], "result": {
        "terms": [{"term": t, "count": c} for t, c in terms.most_common(limit)],
        "most_common_values": [{"value": v[:120], "count": c} for v, c in exact.most_common(min(limit, 10))],
    }}
