"""Chunking strategies: the recursive baseline is locked byte-for-byte; heading-aware and semantic chunking
are tested for what they must do (respect structure, size, determinism) and for what they must not do
(invent headings, cross pages, exceed limits)."""
import asyncio
import uuid

import pytest

from fixtures import docx_bytes, text_pdf
from rag_core import pipeline
from rag_core.chunking import (
    TextChunk,
    chunk_pages,
    chunk_pages_semantic,
    chunk_pages_structured,
    fit_to_token_limit,
    semantic_groups,
    split_sections,
    split_sentences,
)
from rag_core.config import settings
from rag_core.db import Chunk, IndexedDocument
from rag_core.parsing import DOCX, MARKDOWN, PDF, Page, markdown_headings, parse_document

SAMPLE = (
    "Section 2: Travel and expenses\n"
    "Hotel stays are reimbursed up to 180 euros per night in all European cities. "
    "Flights must be booked in economy class unless the flight time exceeds 9 hours, in which case premium economy is allowed.\n\n"
    "Section 3: Information security\n"
    "Laptops must use full-disk encryption. Security incidents must be reported within 24 hours of discovery. Lost devices are wiped remotely."
)

MD = """# Handbook

Welcome to the handbook. It explains policies in short sections that follow.

## Travel

Hotel stays are reimbursed up to 180 euros per night in all European cities. Flights must be booked in economy.

### Receipts

Receipts must be submitted within 30 days of the trip. Keep every receipt for the audit that follows each quarter.

## Security

Laptops must use full-disk encryption. Security incidents must be reported within 24 hours of discovery to the team.

```python
# not a heading, just a comment inside code
def report(incident):
    return "reported within 24 hours"
```

| Property | Value |
| --- | --- |
| Battery | 14 months |
"""


# ---------------------------------------------------------------- baseline is locked

def test_recursive_chunker_is_unchanged_from_the_frozen_baseline():
    """Expected values were produced by the pre-change implementation (see docs/rag-architecture.md)."""
    out = [(c.index, c.page, c.text) for c in chunk_pages([Page(1, SAMPLE)], 160, 30)]
    assert out == [
        (0, 1, "Section 2: Travel and expenses\nHotel stays are reimbursed up to 180 euros per night in all European cities."),
        (1, 1, "in all European cities. Flights must be booked in economy class unless the flight time exceeds 9 hours, in which case premium economy is allowed."),
        (2, 1, "premium economy is allowed.\n\nSection 3: Information security"),
        (3, 1, "3: Information security\nLaptops must use full-disk encryption. Security incidents must be reported within 24 hours of discovery. Lost devices are wiped remotely."),
    ]


def test_default_strategy_is_the_frozen_recursive_one():
    from rag_core.config import Settings

    assert Settings(_env_file=None, ENVIRONMENT="test", DATABASE_URL="sqlite+aiosqlite://").CHUNKING_STRATEGY == "recursive"


# ---------------------------------------------------------------- headings

def test_markdown_headings_are_detected_outside_code_fences_only():
    heads = markdown_headings(MD)
    assert [(lvl, title) for _, lvl, title in heads] == [(1, "Handbook"), (2, "Travel"), (3, "Receipts"), (2, "Security")]
    assert all(MD[off:].startswith("#" * lvl + " " + title) for off, lvl, title in heads)


def test_structured_chunks_never_cross_a_heading_and_carry_the_path():
    page = parse_document(MD.encode(), MARKDOWN).pages[0]
    chunks = chunk_pages_structured([page], size=400, overlap=60)
    by_section = {c.section: c for c in chunks}
    # "# Handbook" has almost no body of its own, so it is folded into its first sub-section (path of the deeper one)
    assert "Handbook" not in by_section and "Handbook > Travel" in by_section
    assert by_section["Handbook > Travel"].text.startswith("# Handbook")
    assert "Handbook > Travel > Receipts" in by_section and "Handbook > Security" in by_section
    titles = {"Handbook", "Travel", "Receipts", "Security"}
    for c in chunks:
        # A chunk may contain only headings of its own path (a folded-in parent heading is an ancestor);
        # it never contains the heading of a sibling or of a later section.
        own = set((c.section or "").split(" > "))
        head_lines = [ln.lstrip("#").strip() for ln in c.text.split(chr(10)) if ln.startswith("#")]
        present = {t for t in titles if t in head_lines}
        assert present <= own, (c.section, present)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_fenced_code_block_is_kept_whole_when_it_fits():
    page = parse_document(MD.encode(), MARKDOWN).pages[0]
    chunks = chunk_pages_structured([page], size=400, overlap=60)
    code = [c for c in chunks if "def report" in c.text]
    assert len(code) == 1 and code[0].text.count("```") >= 2 and "return \"reported within 24 hours\"" in code[0].text


def test_table_rows_stay_intact():
    page = parse_document(MD.encode(), MARKDOWN).pages[0]
    chunks = chunk_pages_structured([page], size=400, overlap=60)
    row = [c for c in chunks if "| Battery | 14 months |" in c.text]
    assert row, "a table row must not be split"


def test_tiny_heading_only_sections_are_merged_into_the_next_one():
    page = Page(None, "# Title\n## Sub\nBody text under the sub heading that is long enough to stand on its own as a section here.",
                headings=[(0, 1, "Title"), (8, 2, "Sub")])
    sections = split_sections(page)
    assert len(sections) == 1 and sections[0][0] == ["Title", "Sub"] and sections[0][1].startswith("# Title")


def test_text_before_the_first_heading_has_no_section():
    page = Page(None, "Preface text that comes before any heading and is long enough to count as a section on its own.\n\n# Intro\nIntro body text that is also long enough to be a real section by itself, no merging.",
                headings=[(103, 1, "Intro")])
    sections = split_sections(page, min_chars=20)
    assert sections[0][0] == [] and sections[1][0] == ["Intro"]


def test_pdf_pages_get_no_invented_headings():
    doc = parse_document(text_pdf(["INTRODUCTION\nThis is the first page of the document with a title line.", "Chapter two body text with many words on the second page."]), PDF)
    assert all(p.headings == [] for p in doc.pages)
    chunks = chunk_pages_structured(doc.pages, 400, 60)
    assert all(c.section is None for c in chunks)
    assert [c.page for c in chunks] == [1, 2]


def test_docx_heading_styles_become_sections():
    def build(d):
        d.add_heading("Warranty", 1)
        d.add_paragraph("The warranty covers manufacturing defects for twenty-four months from the date of delivery to the customer. " * 2)
        d.add_heading("Claims", 2)
        d.add_paragraph("Claims require the original receipt and the serial number of the product that is being returned to us. " * 2)
        d.add_paragraph("A normal paragraph that merely looks like a heading", style="Normal")

    doc = parse_document(docx_bytes(build), DOCX)
    page = doc.pages[0]
    assert [(lvl, t) for _, lvl, t in page.headings] == [(1, "Warranty"), (2, "Claims")]
    chunks = chunk_pages_structured([page], 300, 40)
    assert {c.section for c in chunks} == {"Warranty", "Warranty > Claims"}


def test_markdown_ignores_hash_lines_that_are_not_headings():
    text = "#hashtag without a space\nnormal line\n####### seven hashes\n#\n"
    assert markdown_headings(text) == []


# ---------------------------------------------------------------- semantic

TOPIC_A = ["Hotels are reimbursed per night. ", "Hotel receipts need a stamp. ", "Hotel bookings use the portal. "]
TOPIC_B = ["Laptops need disk encryption. ", "Lost laptops are wiped remotely. ", "Laptop incidents are reported fast. "]


def topic_vector(text: str):
    return [1.0, 0.0] if "hotel" in text.lower() else [0.0, 1.0]


async def fake_embed(texts):
    return [topic_vector(t) for t in texts]


def test_semantic_groups_break_where_the_topic_changes():
    sents = TOPIC_A + TOPIC_B
    groups = semantic_groups(sents, [topic_vector(s) for s in sents], percentile=80, min_chars=20, max_chars=10_000)
    assert [len(g) for g in groups] == [3, 3]
    assert "".join(groups[0]) == "".join(TOPIC_A)


async def test_semantic_chunking_end_to_end_with_scripted_embeddings():
    page = Page(1, "".join(TOPIC_A + TOPIC_B))
    chunks, info = await chunk_pages_semantic([page], fake_embed, size=1000, overlap=0, percentile=80, min_chars=20)
    assert len(chunks) == 2 and "Hotel receipts" in chunks[0].text and "Lost laptops" in chunks[1].text
    assert info["fallback"] is None and info["sentences"] == 6 and info["embedding_texts"] == 6
    assert all(c.page == 1 for c in chunks)


async def test_semantic_chunking_is_deterministic():
    page = Page(1, "".join(TOPIC_A + TOPIC_B) * 3)
    a, _ = await chunk_pages_semantic([page], fake_embed, 1000, 0, 80, 20)
    b, _ = await chunk_pages_semantic([page], fake_embed, 1000, 0, 80, 20)
    assert [c.text for c in a] == [c.text for c in b]


async def test_semantic_chunks_never_exceed_the_size_limit():
    long_topic = "".join(f"Hotel sentence number {i} says something about the hotel policy. " for i in range(60))
    chunks, _ = await chunk_pages_semantic([Page(1, long_topic)], fake_embed, size=300, overlap=0, percentile=90, min_chars=50)
    assert chunks and all(len(c.text) <= 300 for c in chunks)
    assert "".join(c.text.replace(" ", "") for c in chunks) == long_topic.replace(" ", "").strip()  # nothing lost


async def test_semantic_chunking_skips_itself_on_huge_documents_to_protect_the_embedding_budget():
    text = "".join(f"Sentence {i} about hotels. " for i in range(50))
    chunks, info = await chunk_pages_semantic([Page(None, text)], fake_embed, 500, 50, max_sentences=10)
    assert info["fallback"] == "too_many_sentences" and info["embedding_texts"] == 0 and chunks


async def test_semantic_edge_cases():
    for text in ["", "   ", "One short sentence."]:
        chunks, _ = await chunk_pages_semantic([Page(None, text)], fake_embed, 500, 50) if text.strip() else ([], {})
        assert len(chunks) == (1 if text.strip() else 0)
    # a sentence longer than the size is split, not dropped
    chunks, _ = await chunk_pages_semantic([Page(None, "word " * 400)], fake_embed, 300, 0)
    assert len(chunks) >= 6 and all(len(c.text) <= 300 for c in chunks)


def test_sentence_split_is_lossless():
    text = "First one. Second one!  Third?\n\nFourth without end"
    assert "".join(split_sentences(text)) == text


# ---------------------------------------------------------------- token guard keeps structure

def test_token_guard_preserves_section_and_method():
    long = TextChunk(0, 2, "word " * 400, section="A > B", method="ocr")
    out = fit_to_token_limit([long], lambda t: len(t.split()), 100)
    assert len(out) > 1 and all(c.section == "A > B" and c.method == "ocr" and c.page == 2 for c in out)
    assert [c.index for c in out] == list(range(len(out)))


# ---------------------------------------------------------------- through the real pipeline

async def _ingest(services, db_engine, strategy, monkeypatch, data=MD.encode(), mime=MARKDOWN):
    monkeypatch.setattr(settings, "CHUNKING_STRATEGY", strategy)
    from rag_core import db as rag_db

    doc, user = uuid.uuid4(), uuid.uuid4()
    async with rag_db.SessionLocal() as db:
        out = await pipeline.ingest(db, services, document_id=doc, user_id=user, filename="h.md", mime_type=mime, data=data)
    return doc, user, out


async def test_pipeline_stores_sections_and_the_strategy_used(services, database, monkeypatch):
    from sqlalchemy import select

    from rag_core import db as rag_db

    monkeypatch.setattr(settings, "CHUNK_SIZE", 400)
    doc, user, out = await _ingest(services, database, "structure", monkeypatch)
    assert out["extraction"]["chunking"]["strategy"] == "structure"
    async with rag_db.SessionLocal() as db:
        rows = (await db.scalars(select(Chunk).where(Chunk.document_id == doc).order_by(Chunk.chunk_index))).all()
        info = (await db.get(IndexedDocument, doc)).pipeline_info
    assert {r.section for r in rows} >= {"Handbook > Travel", "Handbook > Security"}
    assert all(r.extraction_method == "native" for r in rows)
    assert info["chunking"] == "structure" and info["chunk_size"] == 400


async def test_heading_context_is_embedded_but_not_stored_in_the_text(services, database, monkeypatch):
    """A chunk that never mentions 'Receipts' in its own text is still found by that word, because the
    embedding input carries the heading path. The returned text stays the original text."""
    monkeypatch.setattr(settings, "CHUNK_SIZE", 400)
    doc, user, _ = await _ingest(services, database, "structure", monkeypatch)
    from rag_core import db as rag_db

    async with rag_db.SessionLocal() as db:
        res = await pipeline.retrieve(db, services, user_id=user, query="Receipts", document_ids=[doc], top_k=3)
    assert res["chunks"], res
    top = res["chunks"][0]
    assert top["section"] == "Handbook > Travel > Receipts" and top["text"].startswith("### Receipts")
    assert "Handbook > Travel > Receipts" in res["context"]


async def test_semantic_strategy_runs_through_the_pipeline_and_counts_its_embeddings(services, database, monkeypatch):
    monkeypatch.setattr(settings, "CHUNK_SIZE", 600)
    monkeypatch.setattr(settings, "SEMANTIC_MIN_CHUNK_CHARS", 60)
    text = ("Hotels are reimbursed per night. Hotel receipts need a stamp. Hotel bookings use the portal. "
            "Laptops need disk encryption. Lost laptops are wiped remotely. Laptop incidents are reported fast.")
    doc, user, out = await _ingest(services, database, "semantic", monkeypatch, data=text.encode(), mime="text/plain")
    info = out["extraction"]["chunking"]
    assert info["strategy"] == "semantic" and info["sentences"] == 6 and info["embedding_texts"] == 6
    assert out["timings"]["chunking_embedding_texts"] == 6


async def test_recursive_default_ingests_identically_to_before(services, database, monkeypatch):
    monkeypatch.setattr(settings, "CHUNK_SIZE", 160)
    monkeypatch.setattr(settings, "CHUNK_OVERLAP", 30)
    doc, user, out = await _ingest(services, database, "recursive", monkeypatch, data=SAMPLE.encode(), mime="text/plain")
    assert out["chunks"] == 4
    from sqlalchemy import select

    from rag_core import db as rag_db

    async with rag_db.SessionLocal() as db:
        rows = (await db.scalars(select(Chunk).where(Chunk.document_id == doc).order_by(Chunk.chunk_index))).all()
    assert all(r.section is None for r in rows)
