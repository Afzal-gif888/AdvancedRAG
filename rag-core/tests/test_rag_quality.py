"""Regression tests for retrieval correctness found by the core RAG audit: document order and
table semantics in DOCX, self-describing CSV rows, row-safe chunk overlap, no silent embedding
truncation, and graceful degradation when the vector store is down."""
import io
import uuid

import pytest

from rag_core import errors, pipeline
from rag_core.chunking import TextChunk, chunk_pages, fit_to_token_limit
from rag_core.parsing import Page, parse

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _docx(build) -> bytes:
    import docx

    d = docx.Document()
    build(d)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------- DOCX

def test_docx_tables_stay_in_document_order_with_header_semantics():
    def build(d):
        d.add_heading("Specifications", 1)
        t = d.add_table(rows=1, cols=2)
        t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Property", "Value"
        for k, v in [("Battery life", "14 months"), ("Weight", "85 grams")]:
            row = t.add_row()
            row.cells[0].text, row.cells[1].text = k, v
        d.add_heading("Warranty", 1)
        d.add_paragraph("Covers 24 months.")

    text = parse(_docx(build), DOCX)[0].text
    # The table sits under its heading, before the next section (it used to be appended at the end).
    assert text.index("Specifications") < text.index("Battery life") < text.index("Warranty")
    assert "Property: Battery life | Value: 14 months" in text
    assert "Property: Weight | Value: 85 grams" in text


def test_docx_merged_cells_are_not_repeated():
    def build(d):
        t = d.add_table(rows=2, cols=3)
        for i, h in enumerate(["Region", "Q1", "Q2"]):
            t.rows[0].cells[i].text = h
        merged = t.rows[1].cells[1].merge(t.rows[1].cells[2])
        t.rows[1].cells[0].text, merged.text = "EMEA", "closed"

    text = parse(_docx(build), DOCX)[0].text
    assert text.count("closed") == 1


# ---------------------------------------------------------------- CSV

def test_csv_rows_are_self_describing():
    csv = "product,country,price_eur\nAurora X2,Germany,249\nHalo Display,Portugal,186\n"
    lines = parse(csv.encode(), "text/csv")[0].text.splitlines()
    assert lines == ["product: Aurora X2 | country: Germany | price_eur: 249",
                     "product: Halo Display | country: Portugal | price_eur: 186"]


def test_csv_semicolon_delimiter_quotes_and_empty_cells():
    csv = 'name;city;note\n"Doe, Jane";Berlin;\nLee;"Paris";"says ""hi"""\n'
    text = parse(csv.encode(), "text/csv")[0].text
    assert "name: Doe, Jane | city: Berlin" in text and "note:" not in text.splitlines()[0]
    assert 'note: says "hi"' in text


def test_csv_without_usable_header_or_columns_falls_back():
    no_header = "1,2\n5,6\n3,4\n"  # an all-numeric first row is data: no labels invented
    assert parse(no_header.encode(), "text/csv")[0].text.splitlines() == ["1 | 2", "5 | 6", "3 | 4"]
    duplicate_names = "a,a\nx,y\n"
    assert parse(duplicate_names.encode(), "text/csv")[0].text.splitlines() == ["a | a", "x | y"]
    with_year_columns = "region,2023,2024\nEMEA,10,12\n"
    assert parse(with_year_columns.encode(), "text/csv")[0].text == "region: EMEA | 2023: 10 | 2024: 12"
    single_column = "just\nsome\nlines\n"
    assert parse(single_column.encode(), "text/csv")[0].text == "just\nsome\nlines"


def test_large_csv_every_chunk_keeps_column_names():
    rows = ["product,country,price_eur"] + [f"Item {i},Country {i},{i}" for i in range(300)]
    chunks = chunk_pages(parse("\n".join(rows).encode(), "text/csv"), 300, 60)
    assert len(chunks) > 10
    for c in chunks:
        # Overlap starts on a row boundary, so no chunk begins with half a row.
        assert c.text.startswith("product: "), c.text[:40]


# ---------------------------------------------------------------- chunking

def test_overlap_starts_at_line_boundary_when_available():
    text = "\n".join(f"line {i:03d} " + "word " * 8 for i in range(40))
    chunks = chunk_pages([Page(None, text)], 200, 80)
    assert all(c.text.startswith("line ") for c in chunks)
    assert chunks[1].text.splitlines()[0].strip() in chunks[0].text  # still overlapping


def test_fit_to_token_limit_splits_oversized_chunks_and_renumbers():
    def count(text):  # 1 "token" per character: easy to reason about
        return len(text)

    big = " ".join(f"code-{i}" for i in range(200))  # ~1400 chars
    chunks = [TextChunk(0, 3, "short one"), TextChunk(1, 4, big)]
    out = fit_to_token_limit(chunks, count, max_tokens=300)
    assert all(count(c.text) <= 300 for c in out)
    assert [c.index for c in out] == list(range(len(out)))
    assert out[0].text == "short one" and {c.page for c in out[1:]} == {4}
    assert " ".join(c.text for c in out[1:]).split() == big.split()  # nothing lost


async def test_ingest_never_embeds_more_tokens_than_the_model_accepts(services, database):
    from rag_core import db as rag_db

    seen = []
    services.embeddings.max_tokens = 40
    services.embeddings.count_tokens = lambda text: len(text.split())
    original = services.embeddings.embed

    async def spy(texts, task):
        seen.extend(texts)
        return await original(texts, task)

    services.embeddings.embed = spy
    text = ("alpha beta gamma delta " * 60).encode()
    async with rag_db.SessionLocal() as db:
        out = await pipeline.ingest(db, services, document_id=uuid.uuid4(), user_id=uuid.uuid4(), filename="t.txt",
                                    mime_type="text/plain", data=text)
    assert out["chunks"] == len(seen) and all(len(t.split()) <= 40 for t in seen)


# ---------------------------------------------------------------- retrieval degradation

async def _ingest_guide(services, user, doc):
    from rag_core import db as rag_db

    async with rag_db.SessionLocal() as db:
        await pipeline.ingest(db, services, document_id=doc, user_id=user, filename="guide.txt", mime_type="text/plain",
                              data=b"Refunds are processed within five business days. Shipping is free above fifty dollars.")


async def test_vector_store_outage_degrades_to_keyword_search(services, database, monkeypatch):
    from rag_core import db as rag_db

    user, doc = uuid.uuid4(), uuid.uuid4()
    await _ingest_guide(services, user, doc)

    async def down(*a, **k):
        raise errors.vector_store_unavailable()

    monkeypatch.setattr(services.vectors, "query", down)
    async with rag_db.SessionLocal() as db:
        result = await pipeline.retrieve(db, services, user_id=user, query="refunds processed", document_ids=[doc])
    assert result["chunks"] and "Refunds" in result["chunks"][0]["text"]
    assert result["metadata"]["retrieval_strategy"] == "keyword_only_degraded"


async def test_misconfigured_vector_store_is_not_masked(services, database, monkeypatch):
    from rag_core import db as rag_db

    user, doc = uuid.uuid4(), uuid.uuid4()
    await _ingest_guide(services, user, doc)

    async def broken(*a, **k):
        raise errors.ServiceError("SOMETHING_PERMANENT", "bad", 400, retryable=False)

    monkeypatch.setattr(services.vectors, "query", broken)
    async with rag_db.SessionLocal() as db:
        with pytest.raises(errors.ServiceError):
            await pipeline.retrieve(db, services, user_id=user, query="refunds", document_ids=[doc])


# ---------------------------------------------------------------- real model

async def test_real_model_finds_a_row_deep_inside_a_large_csv(local_embeddings, database):
    """bge-small + hybrid retrieval must pull the exact row from a CSV spread over many chunks."""
    from rag_core import db as rag_db
    from rag_core.config import settings
    from rag_core.rerank import NoopReranker
    from rag_core.vectorstore import LocalVectorStore

    countries = ["Germany", "France", "Italy", "Spain", "Portugal", "Iceland", "Norway", "Greece", "Poland", "Austria"]
    products = ["Aurora X1", "Aurora X2", "Halo Display", "Stratus Cable", "Zephyr Relay"]
    rows = ["product,country,price_eur"] + [f"{p},{c},{100 + i * 10 + j}" for i, p in enumerate(products) for j, c in enumerate(countries)]
    svc = pipeline.Services(settings, local_embeddings, LocalVectorStore(), NoopReranker())
    user, doc = uuid.uuid4(), uuid.uuid4()
    async with rag_db.SessionLocal() as db:
        out = await pipeline.ingest(db, svc, document_id=doc, user_id=user, filename="prices.csv", mime_type="text/csv",
                                    data="\n".join(rows).encode())
        assert out["chunks"] > 3
        result = await pipeline.retrieve(db, svc, user_id=user, query="How much does the Zephyr Relay cost in Iceland?", document_ids=[doc])
    top = result["chunks"][0]["text"]
    assert "product: Zephyr Relay | country: Iceland | price_eur: 145" in top
