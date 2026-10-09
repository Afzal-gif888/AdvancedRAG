"""Persistent Knowledge Base with strictly grounded, self-correcting RAG, through the whole chain:
real backend -> Agent Core -> RAG Core -> LLM service -> mock FreeLLMAPI (see conftest).

SUBSTITUTES USED HERE (these tests prove the wiring and the policies, not retrieval quality):
- embeddings: EMBEDDING_PROVIDER=fake (hash vectors, 64 dims), not BAAI/bge-small-en-v1.5
- vector store: VECTOR_STORE=local (vectors in RAG Core's database), not Pinecone
- reranker: none
- model: scripts/mock_freellmapi.py; the evidence grader is a keyword-overlap stand-in, the
  verifier always answers "supported", the table planner is rule-based
Real Pinecone / BGE / FreeLLMAPI behaviour is covered by the live run (docs, delivery report).

Numbers in test names refer to the acceptance scenarios of the Knowledge Base task.
"""
import csv
import io
import json
import re
import uuid
from collections import Counter

import httpx
import pytest

API = "/api/v1"
ABSTAIN = "I couldn't find sufficient evidence for this answer in your Knowledge Base."
WARRANTY = (
    "Warranty policy. Every AdvancedWidget ships with a 24 month warranty covering manufacturing defects.\n\n"
    "Claims are filed online with the original receipt. Water damage is not covered by the warranty."
)


def sse(text):
    out = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out


class Recorder(httpx.ASGITransport):
    def __init__(self, app):
        super().__init__(app=app)
        self.bodies = []

    async def handle_async_request(self, request):
        if request.content:
            self.bodies.append(json.loads(request.content))
        return await super().handle_async_request(request)


@pytest.fixture
def spies(stack):
    """Records backend -> Agent Core requests and Agent Core -> LLM service requests."""
    from agent_core import main as agent_main
    from agent_core.clients import LLMClient, RAGClient
    from agent_core.config import settings as agent_settings
    from agent_core.orchestrator import Orchestrator
    from app.integrations import http as backend_http

    to_agent, to_llm = Recorder(stack["agent_app"]), Recorder(stack["llm_app"])
    old = backend_http._clients["agent_core"]
    backend_http._clients["agent_core"] = httpx.AsyncClient(transport=to_agent, base_url="http://agent-core", timeout=30)
    agent_main.app.dependency_overrides[agent_main.get_orchestrator] = lambda: Orchestrator(
        agent_settings, RAGClient(agent_settings, transport=httpx.ASGITransport(app=stack["rag_app"])), LLMClient(agent_settings, transport=to_llm))
    yield to_agent, to_llm
    backend_http._clients["agent_core"] = old


async def login(client):
    r = await client.post(f"{API}/auth/register", json={"email": f"kb-{uuid.uuid4().hex[:8]}@example.com", "password": "Contract-pass1", "full_name": "KB"})
    client.cookies.clear()
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def drain():
    from app.integrations.rag_core import RAGCoreClient
    from app.integrations.storage import get_storage
    from app.workers.runner import process_one

    while await process_one(rag=RAGCoreClient("http://rag-core"), storage=get_storage()):
        pass


async def ready_doc(client, h, name, data, mime="text/plain"):
    data = data.encode() if isinstance(data, str) else data
    doc = (await client.post(f"{API}/documents", files={"file": (name, data, mime)}, headers=h)).json()
    await drain()
    assert (await client.get(f"{API}/documents/{doc['id']}", headers=h)).json()["status"] == "ready"
    return doc["id"]


async def new_chat(client, h):
    return (await client.post(f"{API}/conversations", json={}, headers=h)).json()["id"]


async def ask(client, h, conv, q, **extra):
    r = await client.post(f"{API}/conversations/{conv}/messages/stream", json={"content": q, **extra}, headers=h)
    assert r.status_code == 200, r.text
    events = sse(r.text)
    assert events[-1][0] == "message.completed", events[-1]
    return events, events[-1][1]["assistant_message"]


def names(events):
    return [n for n, _ in events]


def stream_bodies(bodies):
    """Agent Core -> LLM service calls that generate the answer (not the checks)."""
    return [b for b in bodies if not any(m in b["messages"][0]["content"] for m in
                                                 ("EVIDENCE GRADER", "SEARCH QUERY CORRECTION", "ANSWER VERIFIER",
                                                  "TABLE QUERY PLANNER", "standalone search query"))]


async def stored_meta(message_id):
    from app.db import session as backend_db
    from app.db.models import Message

    async with backend_db.session_scope() as db:
        return (await db.get(Message, uuid.UUID(message_id))).metadata_ or {}


async def rag_rows(document_id):
    """(indexed document rows, chunk rows, chunks with a stored vector) in RAG Core for a document."""
    from sqlalchemy import func, select

    from rag_core import db as rag_db

    doc = uuid.UUID(document_id)
    async with rag_db.SessionLocal() as s:
        docs = await s.scalar(select(func.count()).select_from(rag_db.IndexedDocument).where(rag_db.IndexedDocument.document_id == doc))
        chunks = await s.scalar(select(func.count()).select_from(rag_db.Chunk).where(rag_db.Chunk.document_id == doc))
        vectors = await s.scalar(select(func.count()).select_from(rag_db.Chunk)
                                 .where(rag_db.Chunk.document_id == doc, rag_db.Chunk.embedding.is_not(None)))
    return docs, chunks, vectors


def norm(text):
    return " ".join(text.split())


# ------------------------------------------------------------------------- 1. upload -> READY

async def test_1_pdf_upload_reaches_ready_only_when_chunks_and_vectors_exist(client, stack, spies):
    from test_agent_core_contracts import make_pdf

    h = await login(client)
    doc = await ready_doc(client, h, "handbook.pdf", make_pdf(["Cover page", "Remote work is allowed three days per week"]), "application/pdf")
    docs, chunks, vectors = await rag_rows(doc)
    assert docs == 1 and chunks >= 1 and vectors == chunks
    events, msg = await ask(client, h, await new_chat(client, h), "How many days per week is remote work allowed?")
    assert msg["citations"] and msg["citations"][0]["document_id"] == doc and msg["citations"][0]["page"] == 2


# ------------------------------------------------------------------------- 2, 3, 10. any chat, real passages

async def test_2_3_10_knowledge_base_answers_in_every_new_chat_with_citations_to_real_passages(client, stack, spies):
    to_agent, _ = spies
    h = await login(client)
    doc = await ready_doc(client, h, "warranty-policy.txt", WARRANTY)
    for question in ("How long is the warranty?", "Is water damage covered by the warranty?", "How are warranty claims filed?"):
        chat = await new_chat(client, h)  # brand-new conversation, nothing attached
        events, msg = await ask(client, h, chat, question)
        req = to_agent.bodies[-1]
        assert req["document_ids"] == [] and [d["document_id"] for d in req["library_documents"]] == [doc]
        assert next(d for n, d in events if n == "retrieval.completed")["retrieved_chunks"] >= 1
        assert msg["citations"] and {c["document_id"] for c in msg["citations"]} == {doc}
        for c in msg["citations"]:  # 10: every citation is a passage of the cited document
            assert c["filename"] == "warranty-policy.txt" and c["chunk_id"].startswith(f"{doc}:")
            assert norm(c["snippet"]) in norm(WARRANTY)
        markers = {int(n) for n in re.findall(r"\[(\d+)\]", msg["content"])}
        assert markers <= {c["source_id"] for c in msg["citations"]}  # no dangling [n] in the stored answer
        assert (await stored_meta(msg["id"])).get("retrieval_scope") == "knowledge_base"
        assert (await client.get(f"{API}/conversations/{chat}/documents", headers=h)).json() == []  # nothing attached

    # A follow-up in the last chat keeps working (history + Knowledge Base search).
    events, f = await ask(client, h, chat, "And how are claims filed?")
    assert {c["document_id"] for c in f["citations"]} == {doc}


# ------------------------------------------------------------------------- 4. chat attachments

async def test_4_chat_attachments_neither_ingest_nor_change_the_search_scope(client, stack, spies):
    to_agent, _ = spies
    h = await login(client)
    doc = await ready_doc(client, h, "warranty-policy.txt", WARRANTY)
    before = (await client.get(f"{API}/documents", headers=h)).json()["total"]
    chat = await new_chat(client, h)
    await ask(client, h, chat, "How long is the warranty?", document_ids=[doc])  # an older client attaching
    assert to_agent.bodies[-1]["document_ids"] == []  # retrieval scope is the Knowledge Base, not attachments
    assert (await client.get(f"{API}/documents", headers=h)).json()["total"] == before  # nothing new ingested


# ------------------------------------------------------------------------- 5, 6. absent answer, weak retrieval

async def test_5_6_absent_answer_gets_one_bounded_correction_then_the_exact_abstain_answer(client, stack, spies):
    _, to_llm = spies
    h = await login(client)
    await ready_doc(client, h, "warranty-policy.txt", WARRANTY)
    calls_before = len(to_llm.bodies)
    events, msg = await ask(client, h, await new_chat(client, h), "What does our dental plan cover for employees?")
    # Retrieval returned passages (no score cutoff), the grader judged none relevant: none are used.
    assert next(d for n, d in events if n == "retrieval.completed")["retrieved_chunks"] == 0
    assert msg["content"] == ABSTAIN and msg["citations"] == []
    meta = await stored_meta(msg["id"])
    assert meta["verification_status"] == "insufficient_evidence" and meta["evidence_grade"] == "insufficient"
    assert meta["correction_attempts"] == 1 and meta["answer_mode"] == "abstain"
    turn = to_llm.bodies[calls_before:]
    assert sum("SEARCH QUERY CORRECTION" in b["messages"][0]["content"] for b in turn) == 1
    assert stream_bodies(turn) == []  # no answer generated from weak evidence


async def test_general_question_is_still_answered_without_document_evidence(client, stack, spies):
    _, to_llm = spies
    h = await login(client)
    await ready_doc(client, h, "warranty-policy.txt", WARRANTY)
    events, msg = await ask(client, h, await new_chat(client, h), "Explain what a hash table is.")
    assert msg["content"].startswith("**Mock model reply**") and msg["citations"] == []
    sent = stream_bodies(to_llm.bodies)[-1]["messages"]
    assert "<evidence>" not in sent[-1]["content"] and "Water damage" not in json.dumps(sent)


# ------------------------------------------------------------------------- 7. isolation

async def test_7_other_users_never_see_the_document(client, stack, spies):
    to_agent, to_llm = spies
    owner = await login(client)
    doc = await ready_doc(client, owner, "warranty-policy.txt", WARRANTY)
    other = await login(client)
    events, msg = await ask(client, other, await new_chat(client, other), "According to the warranty document I uploaded, is water damage covered?")
    req = to_agent.bodies[-1]
    assert req["library_documents"] == [] and req["document_ids"] == []
    assert "retrieval.started" not in names(events) and msg["citations"] == []
    assert "Water damage" not in json.dumps(to_llm.bodies[-1])
    conv = await new_chat(client, other)  # explicitly passing the owner's id is rejected before anything runs
    r = await client.post(f"{API}/conversations/{conv}/messages/stream", json={"content": "summarize", "document_ids": [doc]}, headers=other)
    assert r.status_code == 404


# ------------------------------------------------------------------------- 8. delete

async def test_8_delete_removes_the_document_from_retrieval_chunks_and_vectors(client, stack, spies):
    to_agent, to_llm = spies
    h = await login(client)
    doc = await ready_doc(client, h, "warranty-policy.txt", WARRANTY)
    assert (await rag_rows(doc))[1] >= 1
    assert (await client.delete(f"{API}/documents/{doc}", headers=h)).status_code == 204
    await drain()  # the deletion job removes it from RAG Core
    assert await rag_rows(doc) == (0, 0, 0)
    events, msg = await ask(client, h, await new_chat(client, h), "Is water damage covered by the warranty?")
    assert to_agent.bodies[-1]["library_documents"] == [] and msg["citations"] == []
    assert "Water damage" not in json.dumps(to_llm.bodies[-1])


async def test_failed_and_unindexed_documents_are_never_used(client, stack, spies):
    from app.db import session as backend_db
    from app.db.models import Document

    to_agent, to_llm = spies
    h = await login(client)
    broken = (await client.post(f"{API}/documents", files={"file": ("empty.csv", b"  \n", "text/csv")}, headers=h)).json()["id"]
    await drain()
    assert (await client.get(f"{API}/documents/{broken}", headers=h)).json()["status"] == "failed"
    events, msg = await ask(client, h, await new_chat(client, h), "Analyze the spreadsheet I uploaded")
    assert to_agent.bodies[-1]["library_documents"] == [] and "retrieval.started" not in names(events)
    assert "no such ready document is in their Knowledge Base" in to_llm.bodies[-1]["messages"][0]["content"]

    # Ready in the backend but nothing indexed in RAG Core: searched, nothing found, abstained.
    ghost = (await client.post(f"{API}/documents", files={"file": ("ghost-manual.txt", b"never indexed", "text/plain")}, headers=h)).json()["id"]
    async with backend_db.session_scope() as db:
        (await db.get(Document, uuid.UUID(ghost))).status = "ready"
        await db.commit()
    events, msg = await ask(client, h, await new_chat(client, h), "What does the ghost manual document say?")
    assert next(d for n, d in events if n == "retrieval.completed")["retrieved_chunks"] == 0
    assert msg["content"] == ABSTAIN and msg["citations"] == []


# ------------------------------------------------------------------------- document selection
# Regression (2026-10-09): "What is the incident escalation code specified in my Knowledge Base?"
# with two ready PDFs answered "I found more than one matching document ... Which one should I
# use?". Several documents in scope must never produce a question about documents.

INCIDENT_Q = "What is the incident escalation code specified in my Knowledge Base?"
LARGE = "\n\n".join(f"Chapter {i}. Office logistics: desks, parking permits and catering schedules for building {i}." for i in range(1, 12))
SMOKE = "Quick smoke test. The incident escalation code is ESC-4417. Use it when paging the on-call engineer."
ATLAS = "Project Atlas runbook. The incident escalation code for Project Atlas is ATL-7731."
BOREALIS = "Project Borealis runbook. The incident escalation code for Project Borealis is BOR-2290."
NO_DOC_QUESTION = re.compile(r"more than one matching document|which (one|document|file)", re.I)


def model_input(to_llm):
    return stream_bodies(to_llm.bodies)[-1]["messages"]


async def test_two_documents_only_one_contains_the_answer(client, stack, spies):
    to_agent, to_llm = spies
    h = await login(client)
    large = await ready_doc(client, h, "AdvancedRAG_Large_Knowledge_Test.txt", LARGE)
    smoke = await ready_doc(client, h, "AdvancedRAG_Quick_Smoke_Test.txt", SMOKE)
    events, msg = await ask(client, h, await new_chat(client, h), INCIDENT_Q)
    assert not NO_DOC_QUESTION.search(msg["content"])
    assert {d["document_id"] for d in to_agent.bodies[-1]["library_documents"]} == {large, smoke}
    assert "ESC-4417" in msg["content"] and {c["document_id"] for c in msg["citations"]} == {smoke}
    assert all(norm(c["snippet"]) in norm(SMOKE) for c in msg["citations"])
    assert (await stored_meta(msg["id"]))["retrieval_scope"] == "knowledge_base"


async def test_different_codes_for_different_projects_reach_the_model_with_the_context_rule(client, stack, spies):
    """Wiring only (the mock model cannot judge context): both projects' passages are searched and
    given to the model with the rule to answer by context or ask about the context, never to pick
    a document. The real model's behaviour is checked in the live run."""
    _, to_llm = spies
    h = await login(client)
    await ready_doc(client, h, "atlas-runbook.txt", ATLAS)
    await ready_doc(client, h, "borealis-runbook.txt", BOREALIS)
    events, msg = await ask(client, h, await new_chat(client, h), "What is the incident escalation code for Project Borealis?")
    sent = model_input(to_llm)
    assert "BOR-2290" in sent[-1]["content"] and "Never ask the user to choose a document" in sent[0]["content"]
    assert not NO_DOC_QUESTION.search(msg["content"]) and msg["citations"]


async def test_several_documents_agreeing_are_answered_without_any_question(client, stack, spies):
    h = await login(client)
    await ready_doc(client, h, "handbook.txt", "Operations handbook. The incident escalation code is ESC-4417 for all teams.")
    await ready_doc(client, h, "AdvancedRAG_Quick_Smoke_Test.txt", SMOKE)
    events, msg = await ask(client, h, await new_chat(client, h), INCIDENT_Q)
    assert "ESC-4417" in msg["content"] and msg["citations"] and not NO_DOC_QUESTION.search(msg["content"])


async def test_a_question_naming_one_document_searches_only_it(client, stack, spies):
    _, to_llm = spies
    h = await login(client)
    atlas = await ready_doc(client, h, "atlas-runbook.txt", ATLAS)
    await ready_doc(client, h, "borealis-runbook.txt", BOREALIS)
    events, msg = await ask(client, h, await new_chat(client, h), "What incident escalation code does atlas-runbook.txt specify?")
    assert "BOR-2290" not in json.dumps(model_input(to_llm)) and {c["document_id"] for c in msg["citations"]} == {atlas}
    assert (await stored_meta(msg["id"]))["retrieval_scope"] == "named"


async def test_answer_absent_from_all_documents_abstains(client, stack, spies):
    h = await login(client)
    await ready_doc(client, h, "AdvancedRAG_Large_Knowledge_Test.txt", LARGE)
    await ready_doc(client, h, "AdvancedRAG_Quick_Smoke_Test.txt", SMOKE)
    events, msg = await ask(client, h, await new_chat(client, h), "What is the disaster recovery RTO specified in my Knowledge Base?")
    assert msg["content"] == ABSTAIN and msg["citations"] == []


async def test_other_users_and_deleted_documents_are_never_searched(client, stack, spies):
    to_agent, to_llm = spies
    other = await login(client)
    await ready_doc(client, other, "foreign-runbook.txt", "The incident escalation code is FOREIGN-999.")
    h = await login(client)
    deleted = await ready_doc(client, h, "old-runbook.txt", "The incident escalation code is OLD-111.")
    assert (await client.delete(f"{API}/documents/{deleted}", headers=h)).status_code == 204
    await drain()
    smoke = await ready_doc(client, h, "AdvancedRAG_Quick_Smoke_Test.txt", SMOKE)
    events, msg = await ask(client, h, await new_chat(client, h), INCIDENT_Q)
    assert [d["document_id"] for d in to_agent.bodies[-1]["library_documents"]] == [smoke]
    sent = json.dumps(model_input(to_llm))
    assert "FOREIGN-999" not in sent and "OLD-111" not in sent
    assert "ESC-4417" in msg["content"] and {c["document_id"] for c in msg["citations"]} == {smoke}


# ------------------------------------------------------------------------- 9. exact CSV statistics

async def test_9_csv_statistics_are_computed_exactly_over_the_full_dataset(client, stack, spies):
    _, to_llm = spies
    h = await login(client)
    sentiments = ["positive", "negative", "neutral", "positive", "negative"]
    rows = [[f"R{i:04d}", ["en", "de", "fr"][i % 3], sentiments[(i * 7) % 5], f"Review text number {i} about delivery"] for i in range(1, 1201)]
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows([["review_id", "language", "sentiment", "text"], *rows])
    doc = await ready_doc(client, h, "reviews.csv", buf.getvalue(), "text/csv")
    assert (await rag_rows(doc))[1] > 6  # far more chunks than retrieval's top-k: chunks alone could not answer

    events, msg = await ask(client, h, await new_chat(client, h), "What percentage of sentiment values are negative?")
    counts = Counter(r[2] for r in rows)
    expected = f"negative: {counts['negative']} ({round(100 * counts['negative'] / len(rows), 2)}%)"
    sent = stream_bodies(to_llm.bodies)[-1]["messages"][-1]["content"]
    assert f"all {len(rows)} rows" in sent and expected in sent  # evidence = results over every row
    assert expected in msg["content"]
    assert [c["chunk_id"] for c in msg["citations"]] == [f"{doc}:table"]
    meta = await stored_meta(msg["id"])
    assert meta["verification_status"] == "computed" and meta["table_analysis"] == "computed"


# ------------------------------------------------------------------------- follow-ups on earlier answers
# Regression (2026-10-09): after an exact CSV distribution, "give above in table format" answered
# "I couldn't find sufficient evidence ..." (it was re-searched as a new question over raw rows).

async def test_csv_result_then_table_follow_up_reuses_the_stored_result(client, stack, spies):
    to_agent, to_llm = spies
    h = await login(client)
    rows = [[f"R{i:04d}", "positive" if i % 10 < 5 else "negative" if i % 10 < 8 else "neutral"] for i in range(5000)]
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows([["review_id", "sentiment"], *rows])
    doc = await ready_doc(client, h, "AdvancedRAG_CSV_Exact_Stats_5000_Rows.csv", buf.getvalue(), "text/csv")
    chat = await new_chat(client, h)
    events, first = await ask(client, h, chat, "What is the sentiment distribution in the sentiment column?")
    assert (await stored_meta(first["id"]))["structured_result"]["results"][0]["rows_total"] == 5000
    calls = len(to_llm.bodies)
    for q in ("give above in table format", "give the above information that you provided recently in table format"):
        events, msg = await ask(client, h, chat, q)
        assert "| Positive | 2500 | 50.0% |" in msg["content"] and "| Total | 5000 | 100.0% |" in msg["content"]
        assert "retrieval.started" not in names(events) and [c["chunk_id"] for c in msg["citations"]] == [f"{doc}:table"]
        assert to_agent.bodies[-1]["history"][1]["structured"]["filename"] == "AdvancedRAG_CSV_Exact_Stats_5000_Rows.csv"
    assert len(to_llm.bodies) == calls  # deterministic: no model call at all
    # Another conversation of the same user has no access to these results.
    events, msg = await ask(client, h, await new_chat(client, h), "give above in table format")
    assert "2500" not in msg["content"]


async def test_document_answer_follow_up_keeps_the_citation(client, stack, spies):
    h = await login(client)
    doc = await ready_doc(client, h, "warranty-policy.txt", WARRANTY)
    chat = await new_chat(client, h)
    events, first = await ask(client, h, chat, "How long is the warranty?")
    events, msg = await ask(client, h, chat, "Summarize your previous answer")
    assert "retrieval.started" not in names(events) and msg["citations"]
    assert {c["chunk_id"] for c in msg["citations"]} <= {c["chunk_id"] for c in first["citations"]}
    assert all(c["document_id"] == doc for c in msg["citations"])
