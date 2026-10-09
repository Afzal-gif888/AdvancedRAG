"""Exact statistics over a CSV in the user's Knowledge Base.

A quantitative question ("how many rows", "what percentage ... negative", "which language has the
highest share of ...") cannot be answered from a handful of retrieved chunks. Instead:

  plan     the fast model turns the question into a declarative table-query spec (closed set of
           operations, the table's real columns), validated here and again by RAG Core
  execute  RAG Core computes the results over every row of the user's own stored table
  answer   the model words the computed results; every number in the answer must occur in them
           (deterministic check), otherwise a deterministic rendering of the results is used

If no table can answer the question, the caller falls back to normal retrieval.
"""
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from agent_core.config import Settings
from agent_core.grading import _ask, parse_json

log = logging.getLogger("advancerag.agent.structured")

_QUANT = re.compile(
    r"\b(how many|number of|count|counts|percent|percentage|proportion|ratio|share|average|mean|median|"
    r"sum|total|distribution|breakdown|statistics?|most (?:common|frequent)|least (?:common|frequent)|"
    r"highest|lowest|top \d+|rows?|columns?|frequenc(?:y|ies)|compare|comparison|across|versus)\b|%",
    re.I,
)

PLANNER_SYSTEM = (
    "You are the TABLE QUERY PLANNER. You receive the user's QUESTION and the schemas of CSV TABLES from "
    "their Knowledge Base. Choose the one table that can answer it and write up to 3 queries.\n"
    "Operations (use only these, with column names exactly as in the schema):\n"
    "- count: number of rows (optionally filtered)\n"
    "- distribution: value counts and percentages of `column`\n"
    "- crosstab: for each value of `by`, the counts and percentages of `column`\n"
    "- stats: min, max, mean, median, sum of numeric `column`\n"
    "- top_terms: most frequent words and values in a text `column`\n"
    "Filters (optional, at most 5): {\"column\": ..., \"op\": eq|ne|contains|not_contains|gt|gte|lt|lte|in, \"value\": ...}.\n"
    "Reply with JSON only: {\"document_id\": \"<id of the table>\", \"queries\": [{\"operation\": \"...\", "
    "\"column\": \"...\", \"by\": null, \"filters\": [], \"limit\": 20}]}. If no table can answer the "
    "question, reply {\"document_id\": null, \"queries\": []}."
)


class PlannedFilter(BaseModel):
    column: str = Field(min_length=1, max_length=200)
    op: Literal["eq", "ne", "contains", "not_contains", "gt", "gte", "lt", "lte", "in"]
    value: Union[float, str, List[str]]


class PlannedQuery(BaseModel):
    operation: Literal["count", "distribution", "crosstab", "stats", "top_terms"]
    column: Optional[str] = Field(default=None, max_length=200)
    by: Optional[str] = Field(default=None, max_length=200)
    filters: List[PlannedFilter] = Field(default_factory=list, max_length=5)
    limit: int = Field(default=20, ge=1, le=50)


class Plan(BaseModel):
    document_id: uuid.UUID
    queries: List[PlannedQuery] = Field(min_length=1, max_length=3)


def is_quantitative(question: str) -> bool:
    return bool(_QUANT.search(question))


def _schema_text(tables: List[Dict[str, Any]]) -> str:
    out = []
    for t in tables:
        cols = []
        for c in t["columns"]:
            desc = f"{c['name']} ({c['type']}"
            if c.get("examples"):
                desc += ", e.g. " + ", ".join(json.dumps(e) for e in c["examples"][:6])
            if "min" in c:
                desc += f", range {c['min']}..{c['max']}"
            cols.append(desc + ")")
        out.append(f"TABLE document_id={t['document_id']} file={json.dumps(t['filename'])} rows={t['row_count']}\n  columns: " + "; ".join(cols))
    return "\n".join(out)


async def plan(llm, cfg: Settings, question: str, tables: List[Dict[str, Any]], error: Optional[str] = None) -> Optional[Plan]:
    user = f"TABLES:\n{_schema_text(tables)}\n\nQUESTION: {question}"
    if error:
        user += f"\n\nYour previous plan was rejected: {error}. Fix it."
    reply = await _ask(llm, cfg, PLANNER_SYSTEM, user, 500)
    data = parse_json(reply)
    if not data or not data.get("document_id"):
        log.info("table_plan_rejected", extra={"reason": "no_reply" if reply is None else "no_json" if not data else "no_table_chosen"})
        return None
    try:
        p = Plan.model_validate(data)
    except ValidationError as exc:
        fields = sorted({".".join(str(x) for x in e["loc"]) for e in exc.errors()})[:5]
        log.info("table_plan_rejected", extra={"reason": "invalid", "fields": fields})
        return None
    if str(p.document_id) not in {t["document_id"] for t in tables}:
        log.info("table_plan_rejected", extra={"reason": "unknown_table"})
        return None
    return p


_OPS = {"eq": "=", "ne": "!=", "contains": "contains", "not_contains": "does not contain", "gt": ">", "gte": ">=",
        "lt": "<", "lte": "<=", "in": "in"}


def _where(query: Optional[Dict[str, Any]]) -> str:
    """The filters a result was computed under, so the evidence says which rows were counted."""
    filters = (query or {}).get("filters") or []
    return " and ".join(f"{f['column']} {_OPS.get(f['op'], f['op'])} {json.dumps(f['value'])}" for f in filters)


def render(filename: str, results: List[Dict[str, Any]], queries: Optional[List[Dict[str, Any]]] = None) -> str:
    """Deterministic, human-readable rendering of computed results (evidence text and fallback).
    `queries` (same order as `results`) adds the filters each result was computed under."""
    lines = [f"Computed exactly over the full table {filename}:"]
    for i, r in enumerate(results):
        where = _where(queries[i] if queries and i < len(queries) else None)
        if where:
            scope = f"rows where {where}: {r['rows_matched']} of {r['rows_total']} rows"
        else:
            scope = f"{r['rows_matched']} of {r['rows_total']} rows" if r["rows_matched"] != r["rows_total"] else f"all {r['rows_total']} rows"
        res = r["result"]
        op = r["operation"]
        if op == "count":
            lines.append(f"- Row count ({scope}): {res['count']}")
        elif op == "distribution":
            vals = ", ".join(f"{v['value']}: {v['count']} ({v['percent']}%)" for v in res["values"])
            lines.append(f"- Distribution of {r['column']} ({scope}; {res['distinct']} distinct values): {vals}")
        elif op == "crosstab":
            lines.append(f"- {r['column']} by {r['by']} ({scope}):")
            for g in res["groups"]:
                vals = ", ".join(f"{v['value']}: {v['count']} ({v['percent']}%)" for v in g["values"])
                lines.append(f"  - {g['group']} (total {g['total']}): {vals}")
        elif op == "stats":
            if res.get("count_numeric"):
                lines.append(f"- {r['column']} ({scope}): count {res['count_numeric']}, min {res['min']}, max {res['max']}, "
                             f"mean {res['mean']}, median {res['median']}, sum {res['sum']}")
            else:
                lines.append(f"- {r['column']} ({scope}): no numeric values")
        elif op == "top_terms":
            terms = ", ".join(f"{t['term']} ({t['count']})" for t in res["terms"])
            values = ", ".join(f"\"{v['value']}\" ({v['count']})" for v in res["most_common_values"])
            lines.append(f"- Most frequent words in {r['column']} ({scope}): {terms}")
            lines.append(f"- Most frequent exact values in {r['column']}: {values}")
    return "\n".join(lines)


_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)*")


def numbers_supported(answer: str, evidence: str) -> bool:
    """Every number written in the answer (citation markers excluded) occurs in the computed
    results. Percentages may be rounded to fewer decimals."""
    text = re.sub(r"[\[【]\s*\d+(?:\s*[,;]\s*\d+)*\s*[\]】]", " ", answer)
    have = set()
    for n in _NUMBER.findall(evidence):
        try:
            v = float(n.replace(",", ""))
        except ValueError:
            continue
        have.update({v, round(v), round(v, 1)})
    for n in _NUMBER.findall(text):
        try:
            v = float(n.replace(",", ""))
        except ValueError:
            return False
        if v not in have:
            return False
    return True
