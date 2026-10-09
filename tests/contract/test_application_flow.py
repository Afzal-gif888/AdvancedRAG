"""Authenticated application flows through the whole chain: real backend -> Agent Core -> RAG Core
-> LLM service -> mock FreeLLMAPI (see conftest). Covers what the single-turn contract tests do
not: multi-turn follow-ups from persisted history, unanswerable questions, user isolation,
persistence across sessions, cancellation, FreeLLMAPI rate limiting + retry, and cleanup."""
import asyncio
import json
import uuid

import httpx
import pytest

API = "/api/v1"
PASSWORD = "Contract-pass1"
WIDGET = (
    "Warranty policy. Every AdvancedWidget ships with a 24 month warranty covering manufacturing defects.\n\n"
    "Claims are filed online with the original receipt. Water damage is not covered by the warranty."
)
GADGET = "Gadget manual. The OmniGadget battery lasts 11 hours and recharges in 90 minutes over USB-C."


def sse(text):
    out = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out


async def login(client):
    email = f"f-{uuid.uuid4().hex[:8]}@example.com"
    r = await client.post(f"{API}/auth/register", json={"email": email, "password": PASSWORD, "full_name": "Flow User"})
    assert r.status_code == 201, r.text
    client.cookies.clear()  # each user authenticates by bearer token only
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


class RecordingASGI(httpx.ASGITransport):
    """ASGI transport that keeps the JSON body of every request (backend -> Agent Core)."""

    def __init__(self, app):
        super().__init__(app=app)
        self.bodies = []

    async def handle_async_request(self, request):
        if request.content:
            self.bodies.append(json.loads(request.content))
        return await super().handle_async_request(request)


@pytest.fixture
def agent_spy(stack):
    from app.integrations import http as backend_http

    transport = RecordingASGI(stack["agent_app"])
    old = backend_http._clients["agent_core"]
    backend_http._clients["agent_core"] = httpx.AsyncClient(transport=transport, base_url="http://agent-core", timeout=30)
    yield transport
    backend_http._clients["agent_core"] = old


def scripted_freellmapi(stack, handler):
    """Point the LLM service at a scripted FreeLLMAPI (same provider code, scripted upstream)."""
    from llm_service import main as llm_main
    from llm_service.providers import get_provider
    from llm_service.providers.freellmapi import FreeLLMAPIProvider

    provider = FreeLLMAPIProvider(base_url="http://freellmapi.test/v1", api_key="k", max_retries=0, transport=httpx.MockTransport(handler))
    llm_main.app.dependency_overrides[get_provider] = lambda: provider
    return provider


def restore_freellmapi(stack):
    from llm_service import main as llm_main
    from llm_service.providers import get_provider

    llm_main.app.dependency_overrides[get_provider] = lambda: stack["provider"]


def answer_with(text):
    def handler(request):
        if json.loads(request.content).get("stream"):
            chunks = [f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': w}}]})}\n\n" for w in text.split(" ") if w]
            chunks = [c.replace('"content": "', '"content": " ', 1) if i else c for i, c in enumerate(chunks)]
            body = "".join(chunks) + 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})

    return handler


async def ready_document(client, h, filename, text):
    from app.integrations.rag_core import RAGCoreClient
    from app.integrations.storage import get_storage
    from app.workers.runner import process_one

    doc = (await client.post(f"{API}/documents", files={"file": (filename, text.encode(), "text/plain")}, headers=h)).json()
    assert await process_one(rag=RAGCoreClient("http://rag-core"), storage=get_storage())
    assert (await client.get(f"{API}/documents/{doc['id']}", headers=h)).json()["status"] == "ready"
    return doc


async def ask(client, h, conv_id, content, **extra):
    r = await client.post(f"{API}/conversations/{conv_id}/messages/stream", json={"content": content, **extra}, headers=h)
    assert r.status_code == 200, r.text
    return sse(r.text)


async def stored_metadata(message_id):
    from app.db import session as backend_db
    from app.db.models import Message

    async with backend_db.session_scope() as db:
        return (await db.get(Message, uuid.UUID(message_id))).metadata_ or {}


async def messages(client, h, conv_id):
    return (await client.get(f"{API}/conversations/{conv_id}/messages", headers=h)).json()["items"]


# --------------------------------------------------------------------------------- flows

async def test_multi_turn_follow_up_uses_persisted_history_and_stays_grounded(client, stack, agent_spy):
    h = await login(client)
    doc = await ready_document(client, h, "warranty.txt", WIDGET)
    conv = (await client.post(f"{API}/conversations", json={}, headers=h)).json()

    first = await ask(client, h, conv["id"], "How long is the AdvancedWidget warranty?", document_ids=[doc["id"]])
    a1 = first[-1][1]["assistant_message"]
    assert first[-1][0] == "message.completed" and "24 month warranty" in a1["content"]
    second = await ask(client, h, conv["id"], "Is water damage covered by it?")
    a2 = second[-1][1]["assistant_message"]
    assert second[-1][0] == "message.completed", second[-1]

    # The backend sent the first turn (from the database) as history, and the user's Knowledge Base.
    req2 = agent_spy.bodies[-1]
    assert [m["role"] for m in req2["history"]] == ["user", "assistant"]
    assert req2["history"][0]["content"] == "How long is the AdvancedWidget warranty?"
    assert req2["history"][1]["content"] == a1["content"]
    assert req2["document_ids"] == [] and [d["document_id"] for d in req2["library_documents"]] == [doc["id"]]
    # The follow-up was retrieved against the same document and cited it.
    assert next(d for n, d in second if n == "retrieval.completed")["retrieved_chunks"] >= 1
    assert a2["citations"] and {c["document_id"] for c in a2["citations"]} == {doc["id"]}
    assert (await stored_metadata(a2["id"])).get("verification_status") == "verified"

    # Persisted in order, with citations, exactly as streamed.
    items = await messages(client, h, conv["id"])
    assert [(m["role"], m["status"]) for m in items] == [("user", "completed"), ("assistant", "completed")] * 2
    assert items[3]["content"] == a2["content"] and items[3]["citations"] == a2["citations"]


async def test_unanswerable_question_is_answered_without_invented_citations(client, stack):
    h = await login(client)
    doc = await ready_document(client, h, "warranty.txt", WIDGET)
    conv = (await client.post(f"{API}/conversations", json={}, headers=h)).json()
    # A model that (correctly) abstains: no source marker in the answer.
    scripted_freellmapi(stack, answer_with("I could not find the shipping cost in your documents."))
    try:
        events = await ask(client, h, conv["id"], "What does shipping to Norway cost?", document_ids=[doc["id"]])
    finally:
        restore_freellmapi(stack)
    final = events[-1][1]["assistant_message"]
    assert events[-1][0] == "message.completed" and "could not find" in final["content"]
    assert final["citations"] == []
    # The scripted model also answers the grader (not JSON): grading unavailable, no document named, no
    # citation -> recorded as an answer without document evidence (not as a cited document answer).
    assert (await stored_metadata(final["id"])).get("verification_status") == "no_relevant_documents"


async def test_question_with_no_indexed_evidence_gets_no_citations(client, stack):
    """A ready document with nothing in RAG Core (e.g. purged index) -> no sources, nothing cited,
    and the exact "not in your Knowledge Base" answer instead of anything invented."""
    from app.db import session as backend_db
    from app.db.models import Document

    h = await login(client)
    doc = (await client.post(f"{API}/documents", files={"file": ("ghost.txt", b"never indexed", "text/plain")}, headers=h)).json()
    async with backend_db.session_scope() as db:
        (await db.get(Document, uuid.UUID(doc["id"]))).status = "ready"
        await db.commit()
    conv = (await client.post(f"{API}/conversations", json={}, headers=h)).json()
    events = await ask(client, h, conv["id"], "What is in the ghost document?", document_ids=[doc["id"]])
    final = events[-1][1]["assistant_message"]
    assert events[-1][0] == "message.completed", events[-1]
    assert next(d for n, d in events if n == "retrieval.completed")["retrieved_chunks"] == 0
    assert final["citations"] == []
    assert final["content"] == "I couldn't find sufficient evidence for this answer in your Knowledge Base."
    assert (await stored_metadata(final["id"])).get("verification_status") == "insufficient_evidence"


async def test_user_isolation_across_the_whole_chain(client, stack, agent_spy):
    a, b = await login(client), await login(client)
    doc_a = await ready_document(client, a, "widget.txt", WIDGET)
    doc_b = await ready_document(client, b, "gadget.txt", GADGET)
    conv_a = (await client.post(f"{API}/conversations", json={}, headers=a)).json()
    await ask(client, a, conv_a["id"], "How long is the warranty?", document_ids=[doc_a["id"]])
    conv_b = (await client.post(f"{API}/conversations", json={}, headers=b)).json()

    # B cannot use, read, attach or delete A's resources (404: existence is not revealed).
    sent_before = len(agent_spy.bodies)
    r = await client.post(f"{API}/conversations/{conv_b['id']}/messages/stream",
                          json={"content": "warranty?", "document_ids": [doc_a["id"]]}, headers=b)
    assert r.status_code == 404
    assert len(agent_spy.bodies) == sent_before  # rejected before Agent Core was called
    assert await messages(client, b, conv_b["id"]) == []  # and nothing was persisted
    assert (await client.post(f"{API}/conversations/{conv_b['id']}/documents", json={"document_ids": [doc_a["id"]]}, headers=b)).status_code == 404
    assert (await client.get(f"{API}/conversations/{conv_a['id']}/messages", headers=b)).status_code == 404
    assert (await client.post(f"{API}/conversations/{conv_a['id']}/messages/stream", json={"content": "x"}, headers=b)).status_code == 404
    assert (await client.get(f"{API}/documents/{doc_a['id']}/content", headers=b)).status_code == 404
    assert (await client.delete(f"{API}/documents/{doc_a['id']}", headers=b)).status_code == 404
    assert (await client.get(f"{API}/documents/{doc_a['id']}", headers=a)).json()["status"] == "ready"

    # B's question that matches A's document text: only B's document is searched and cited.
    events = await ask(client, b, conv_b["id"], "How long is the warranty on the widget?", document_ids=[doc_b["id"]])
    sent = agent_spy.bodies[-1]
    assert [d["document_id"] for d in sent["library_documents"]] == [doc_b["id"]] and sent["user_id"] != agent_spy.bodies[0]["user_id"]
    final = events[-1][1]["assistant_message"]
    assert {c["document_id"] for c in final["citations"]} <= {doc_b["id"]}
    assert "24 month" not in final["content"]


async def test_history_persists_across_logout_and_login(client, stack):
    email = f"p-{uuid.uuid4().hex[:6]}@example.com"
    r = await client.post(f"{API}/auth/register", json={"email": email, "password": PASSWORD, "full_name": "Persist"})
    h = {"Authorization": f"Bearer {r.json()['access_token']}"}
    doc = await ready_document(client, h, "warranty.txt", WIDGET)
    conv = (await client.post(f"{API}/conversations", json={}, headers=h)).json()
    streamed = (await ask(client, h, conv["id"], "How long is the warranty?", document_ids=[doc["id"]]))[-1][1]["assistant_message"]

    assert (await client.post(f"{API}/auth/logout", headers=h)).status_code == 204
    client.cookies.clear()
    r = await client.post(f"{API}/auth/login", json={"email": email, "password": PASSWORD})
    h2 = {"Authorization": f"Bearer {r.json()['access_token']}"}
    convs = (await client.get(f"{API}/conversations", headers=h2)).json()["items"]
    assert [c["id"] for c in convs] == [conv["id"]]
    items = await messages(client, h2, conv["id"])
    assert items[1]["content"] == streamed["content"] and items[1]["citations"] == streamed["citations"]
    assert [d["id"] for d in (await client.get(f"{API}/conversations/{conv['id']}/documents", headers=h2)).json()] == [doc["id"]]


async def test_cancelled_generation_keeps_partial_text_and_next_turn_works(client, stack, agent_spy):
    """Disconnect mid-stream (Starlette cancels the SSE generator) with the real Agent Core client."""
    from sqlalchemy import select

    from app.db import session as backend_db
    from app.db.models import User
    from app.integrations.agent_core import AgentCoreClient
    from app.schemas.message import MessageCreate
    from app.services import agent_service, message_service

    h = await login(client)
    doc = await ready_document(client, h, "warranty.txt", WIDGET)
    conv = (await client.post(f"{API}/conversations", json={}, headers=h)).json()
    me = (await client.get(f"{API}/auth/me", headers=h)).json()
    async with backend_db.session_scope() as db:
        user = await db.scalar(select(User).where(User.id == uuid.UUID(me["id"])))
        turn = await message_service.prepare_turn(db, user, uuid.UUID(conv["id"]), MessageCreate(content="How long is the warranty?", document_ids=[doc["id"]]))

    gen = agent_service.stream_turn(AgentCoreClient(), turn)
    while "content.delta" not in await gen.__anext__():
        pass
    task = asyncio.ensure_future(gen.__anext__())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(100):  # the cancelled state is persisted in a shielded task
        items = await messages(client, h, conv["id"])
        if items[1]["status"] == "cancelled":
            break
        await asyncio.sleep(0.05)
    assert items[1]["status"] == "cancelled" and items[1]["error_code"] == "CANCELLED"
    assert items[1]["content"] and items[1]["citations"] == []

    # The conversation is still usable; the cancelled answer is not fed back as history.
    events = await ask(client, h, conv["id"], "And how are claims filed?")
    assert events[-1][0] == "message.completed"
    assert [m["role"] for m in agent_spy.bodies[-1]["history"]] == ["user"]


async def test_freellmapi_rate_limit_is_reported_and_retry_recovers(client, stack):
    h = await login(client)
    doc = await ready_document(client, h, "warranty.txt", WIDGET)
    conv = (await client.post(f"{API}/conversations", json={}, headers=h)).json()
    scripted_freellmapi(stack, lambda req: httpx.Response(429, json={"error": {"message": "provider quota for key sk-secret"}}, headers={"retry-after": "1"}))
    try:
        events = await ask(client, h, conv["id"], "How long is the warranty?", document_ids=[doc["id"]])
    finally:
        restore_freellmapi(stack)
    name, err = events[-1]
    assert name == "error" and err["code"] == "LLM_RATE_LIMITED"
    assert "sk-secret" not in json.dumps(events) and "quota" not in json.dumps(events)  # upstream detail stays server-side
    failed = err["assistant_message"]
    assert failed["status"] == "failed" and failed["content"] == ""

    r = await client.post(f"{API}/conversations/{conv['id']}/messages/{failed['id']}/retry", headers=h)
    retried = sse(r.text)
    assert retried[-1][0] == "message.completed"
    final = retried[-1][1]["assistant_message"]
    assert final["id"] == failed["id"] and "24 month warranty" in final["content"] and final["citations"]


async def test_conversation_delete_and_clear_all_data_clean_up_every_service(client, stack):
    from app.integrations.rag_core import RAGCoreClient
    from app.integrations.storage import get_storage
    from app.workers.runner import process_one
    from rag_core import db as rag_db
    from rag_core.db import IndexedDocument

    h = await login(client)
    doc = await ready_document(client, h, "warranty.txt", WIDGET)
    conv = (await client.post(f"{API}/conversations", json={}, headers=h)).json()
    await ask(client, h, conv["id"], "How long is the warranty?", document_ids=[doc["id"]])

    # Deleting a conversation removes its messages but not the user's documents.
    assert (await client.delete(f"{API}/conversations/{conv['id']}", headers=h)).status_code == 204
    assert (await client.get(f"{API}/conversations/{conv['id']}/messages", headers=h)).status_code == 404
    assert (await client.get(f"{API}/documents/{doc['id']}", headers=h)).json()["status"] == "ready"

    # Clearing all data removes documents from the backend, storage and RAG Core's index.
    conv2 = (await client.post(f"{API}/conversations", json={}, headers=h)).json()
    assert (await client.delete(f"{API}/profile/data", headers=h)).status_code == 204
    while await process_one(rag=RAGCoreClient("http://rag-core"), storage=get_storage()):
        pass
    async with rag_db.SessionLocal() as s:
        assert await s.get(IndexedDocument, uuid.UUID(doc["id"])) is None
    assert (await client.get(f"{API}/documents/{doc['id']}", headers=h)).status_code == 404
    assert (await client.get(f"{API}/conversations/{conv2['id']}", headers=h)).status_code == 404
    assert (await client.get(f"{API}/conversations", headers=h)).json()["items"] == []
