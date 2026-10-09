"""Exact CSV analysis over the whole stored table + ingestion verification.
Expected values are computed independently in the test from the same synthetic rows."""
import json
import uuid
from collections import Counter

import pytest

from rag_core import errors, tables
from rag_core.parsing import parse_table

U1, U2 = uuid.uuid4(), uuid.uuid4()
LANGS, SENT = ["en", "de", "fr"], ["positive", "negative", "neutral"]
ROWS = [[f"R{i:03d}", LANGS[i % 3], SENT[(i * 7) % 4 % 3], str(10 + i % 17),
         "Delivery was late again" if i % 5 == 0 else ("Great battery life" if i % 2 else "Screen scratches easily")] for i in range(1, 301)]
COLUMNS = ["review_id", "language", "sentiment", "rating", "text"]
CSV_BYTES = ("\n".join([",".join(COLUMNS)] + [",".join(r) for r in ROWS]) + "\n").encode()


def run(**kw):
    return tables.run(COLUMNS, ROWS, **kw)


# ---------------------------------------------------------------- engine

def test_count_and_distribution_are_exact_over_all_rows():
    assert run(operation="count")["result"]["count"] == 300
    dist = run(operation="distribution", column="sentiment")["result"]
    expected = Counter(r[2] for r in ROWS)
    assert {v["value"]: v["count"] for v in dist["values"]} == dict(expected)
    neg = next(v for v in dist["values"] if v["value"] == "negative")
    assert neg["percent"] == round(100 * expected["negative"] / 300, 2)


def test_crosstab_answers_which_group_has_the_highest_share():
    groups = run(operation="crosstab", column="sentiment", by="language")["result"]["groups"]
    for g in groups:
        rows = [r for r in ROWS if r[1] == g["group"]]
        neg = sum(1 for r in rows if r[2] == "negative")
        assert g["total"] == len(rows)
        assert next(v for v in g["values"] if v["value"] == "negative")["percent"] == round(100 * neg / len(rows), 2)


def test_stats_filters_and_terms():
    s = run(operation="stats", column="rating")["result"]
    nums = [float(r[3]) for r in ROWS]
    assert (s["count_numeric"], s["min"], s["max"], s["mean"]) == (300, min(nums), max(nums), round(sum(nums) / 300, 6))
    de_neg = run(operation="count", filters=[tables.Filter("language", "eq", "DE"), tables.Filter("sentiment", "eq", "negative")])
    assert de_neg["rows_matched"] == sum(1 for r in ROWS if r[1] == "de" and r[2] == "negative")
    high = run(operation="count", filters=[tables.Filter("rating", "gte", 25)])
    assert high["rows_matched"] == sum(1 for r in ROWS if float(r[3]) >= 25)
    terms = {t["term"]: t["count"] for t in run(operation="top_terms", column="text", limit=50)["result"]["terms"]}
    assert terms["late"] == sum(1 for r in ROWS if "late" in r[4]) and "was" not in terms


@pytest.mark.parametrize("kw,needle", [
    ({"operation": "distribution", "column": "nope"}, "Unknown column"),
    ({"operation": "crosstab", "column": "sentiment"}, "'by' is required"),
    ({"operation": "count", "filters": [tables.Filter("rating", "gt", "abc")]}, "needs a number"),
    ({"operation": "count", "filters": [tables.Filter("rating", "in", "x")]}, "needs a list"),
    ({"operation": "drop_table"}, "Unknown operation"),
    ({"operation": "count", "filters": [tables.Filter("language", "eq", "en")] * 6}, "At most"),
])
def test_invalid_specs_are_rejected_not_executed(kw, needle):
    with pytest.raises(errors.ServiceError) as exc:
        run(**kw)
    assert exc.value.code == "TABLE_QUERY_INVALID" and needle in exc.value.message


def test_parse_table_detects_header_pads_rows_and_enforces_limits():
    t = parse_table(CSV_BYTES, 1000, 50)
    assert t.columns == COLUMNS and len(t.rows) == 300 and t.rows[0] == ROWS[0]
    ragged = parse_table(b"a,b,c\n1,2\n3,4,5\n", 10, 10)
    assert ragged.rows == [["1", "2", ""], ["3", "4", "5"]]
    assert parse_table(CSV_BYTES, 299, 50) is None and parse_table(CSV_BYTES, 1000, 4) is None
    assert parse_table(b"just one line of prose", 10, 10) is None


# ---------------------------------------------------------------- service: stored at ingestion, owner-only

async def _ingest(client, doc, user, data, mime="text/csv", name="reviews.csv"):
    meta = {"document_id": str(doc), "user_id": str(user), "filename": name, "mime_type": mime}
    return await client.post("/api/v1/rag/ingest", data={"metadata": json.dumps(meta)}, files={"file": (name, data, mime)})


async def test_csv_table_is_stored_described_queried_and_deleted_with_the_document(client):
    doc = uuid.uuid4()
    assert (await _ingest(client, doc, U1, CSV_BYTES)).status_code == 200
    d = (await client.post("/api/v1/rag/tables/describe", json={"user_id": str(U1), "document_ids": [str(doc)]})).json()["tables"]
    assert len(d) == 1 and d[0]["row_count"] == 300 and d[0]["filename"] == "reviews.csv"
    cols = {c["name"]: c for c in d[0]["columns"]}
    assert cols["rating"]["type"] == "numeric" and cols["sentiment"]["type"] == "text" and set(cols["sentiment"]["examples"]) == set(SENT)

    q = {"user_id": str(U1), "document_id": str(doc), "queries": [{"operation": "count"}, {"operation": "distribution", "column": "sentiment"}]}
    r = (await client.post("/api/v1/rag/tables/query", json=q)).json()
    assert r["filename"] == "reviews.csv" and r["results"][0]["result"]["count"] == 300
    assert {v["value"]: v["count"] for v in r["results"][1]["result"]["values"]} == dict(Counter(row[2] for row in ROWS))

    # Another user: nothing described, nothing queried.
    assert (await client.post("/api/v1/rag/tables/describe", json={"user_id": str(U2), "document_ids": [str(doc)]})).json()["tables"] == []
    assert (await client.post("/api/v1/rag/tables/query", json={**q, "user_id": str(U2)})).status_code == 404
    bad = await client.post("/api/v1/rag/tables/query", json={**q, "queries": [{"operation": "distribution", "column": "nope"}]})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "TABLE_QUERY_INVALID"

    # Re-ingest replaces the table; delete removes it.
    assert (await _ingest(client, doc, U1, CSV_BYTES.split(b"\n", 101)[0] + b"\n" + b"\n".join(CSV_BYTES.split(b"\n")[1:101]))).status_code == 200
    r = (await client.post("/api/v1/rag/tables/query", json={**q, "queries": [{"operation": "count"}]})).json()
    assert r["results"][0]["result"]["count"] == 100
    assert (await client.delete(f"/api/v1/rag/documents/{doc}", params={"user_id": str(U1)})).status_code == 204
    assert (await client.post("/api/v1/rag/tables/query", json=q)).status_code == 404


async def test_non_csv_documents_have_no_table(client):
    doc = uuid.uuid4()
    assert (await _ingest(client, doc, U1, b"plain text about widgets and gadgets", "text/plain", "notes.txt")).status_code == 200
    d = (await client.post("/api/v1/rag/tables/describe", json={"user_id": str(U1), "document_ids": [str(doc)]})).json()
    assert d["tables"] == []


# ---------------------------------------------------------------- ingestion verification

async def test_ingestion_fails_cleanly_when_vectors_cannot_be_read_back(services, monkeypatch):
    from rag_core import db as rag_db
    from rag_core import pipeline
    from rag_core.config import settings
    from rag_core.db import Chunk, IndexedDocument

    monkeypatch.setattr(settings, "INGEST_VERIFY_SECONDS", 0)

    async def never_visible(user_id, ids, db=None):
        return 0

    monkeypatch.setattr(services.vectors, "count_existing", never_visible)
    doc = uuid.uuid4()
    async with rag_db.SessionLocal() as db:
        with pytest.raises(errors.ServiceError) as exc:
            await pipeline.ingest(db, services, document_id=doc, user_id=U1, filename="a.txt", mime_type="text/plain",
                                  data=b"Some text about warranties. " * 50)
    assert exc.value.code == "VECTOR_INDEX_UNVERIFIED" and exc.value.retryable
    async with rag_db.SessionLocal() as db:
        assert await db.get(IndexedDocument, doc) is None
        from sqlalchemy import func, select
        assert await db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == doc)) == 0
