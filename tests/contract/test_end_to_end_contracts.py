import json
import uuid

import httpx

API = "/api/v1"
PASSWORD = "Contract-pass1"
DOC = (
    "Warranty policy. Every AdvancedWidget ships with a 24 month warranty covering manufacturing defects.\n\n"
    "Claims are filed online with the original receipt. Water damage is not covered by the warranty."
)


def sse(text):
    out = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out


async def _login(client):
    """Sign-up (name, email, password) signs the user in."""
    email = f"c-{uuid.uuid4().hex[:6]}@example.com"
    r = await client.post(f"{API}/auth/register", json={"email": email, "password": PASSWORD, "full_name": "Test User"})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def test_chat_without_documents_goes_through_agent_llm_and_freellmapi(client, stack):
    h = await _login(client)
    conv = (await client.post(f"{API}/conversations", json={}, headers=h)).json()
    r = await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "Say hello"}, headers=h)
    events = sse(r.text)
    names = [n for n, _ in events]
    assert names[0] == "message.created" and names[-1] == "message.completed", names
    assert "retrieval.started" not in names  # no documents -> Agent Core skips retrieval
    final = events[-1][1]["assistant_message"]
    assert final["content"].startswith("**Mock model reply** to: Say hello")
    # Persisted by the backend.
    msgs = (await client.get(f"{API}/conversations/{conv['id']}/messages", headers=h)).json()["items"]
    assert msgs[1]["status"] == "completed" and msgs[1]["content"] == final["content"]


async def test_document_rag_flow_with_citations(client, stack):
    from app.workers.runner import process_one
    from app.integrations.rag_core import RAGCoreClient
    from app.integrations.storage import get_storage

    h = await _login(client)
    doc = (await client.post(f"{API}/documents", files={"file": ("warranty.txt", DOC.encode(), "text/plain")}, headers=h)).json()
    # Backend worker -> real RAG Core app (parse, chunk, embed, index).
    assert await process_one(rag=RAGCoreClient("http://rag-core"), storage=get_storage())
    assert (await client.get(f"{API}/documents/{doc['id']}", headers=h)).json()["status"] == "ready"

    conv = (await client.post(f"{API}/conversations", json={}, headers=h)).json()
    await client.post(f"{API}/conversations/{conv['id']}/documents", json={"document_ids": [doc["id"]]}, headers=h)
    r = await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "How long is the warranty?"}, headers=h)
    events = sse(r.text)
    names = [n for n, _ in events]
    assert names[:2] == ["message.created", "agent.started"] and "retrieval.completed" in names, names
    assert next(d for n, d in events if n == "retrieval.completed")["retrieved_chunks"] >= 1
    final = events[-1][1]["assistant_message"]
    assert "24 month warranty" in final["content"] and "[1]" in final["content"]
    assert final["citations"] and final["citations"][0]["filename"] == "warranty.txt"
    assert final["citations"][0]["document_id"] == doc["id"]
    assert final["citations"][0]["chunk_id"].startswith(f"{doc['id']}:")  # chunk reference survives to the client


async def test_rag_core_down_gives_safe_error_and_no_fabrication(client, stack, monkeypatch):
    from agent_core.clients import LLMClient, RAGClient
    from agent_core.config import settings as agent_settings
    from agent_core.orchestrator import Orchestrator
    from agent_core import main as agent_main

    def down(request):
        raise httpx.ConnectError("rag down")

    llm_transport = httpx.ASGITransport(app=stack["llm_app"])
    agent_main.app.dependency_overrides[agent_main.get_orchestrator] = lambda: Orchestrator(
        agent_settings, RAGClient(agent_settings, transport=httpx.MockTransport(down)), LLMClient(agent_settings, transport=llm_transport)
    )
    h = await _login(client)
    conv = (await client.post(f"{API}/conversations", json={}, headers=h)).json()
    # A doc id is required for retrieval to be attempted; use an attached (unprocessed) doc.
    doc = (await client.post(f"{API}/documents", files={"file": ("x.txt", b"hello", "text/plain")}, headers=h)).json()
    from app.db import session as backend_db
    from app.db.models import Document

    async with backend_db.session_scope() as db:
        d = await db.get(Document, uuid.UUID(doc["id"]))
        d.status = "ready"
        await db.commit()
    r = await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "q", "document_ids": [doc["id"]]}, headers=h)
    events = sse(r.text)
    assert events[-1] == ("error", events[-1][1]) and events[-1][1]["code"] == "RAG_SERVICE_UNAVAILABLE"
    assert "content.delta" not in [n for n, _ in events]


async def test_freellmapi_bad_key_is_contained(client, stack, monkeypatch):
    provider = stack["provider"]
    monkeypatch.setattr(provider, "_api_key", "wrong-key")
    h = await _login(client)
    conv = (await client.post(f"{API}/conversations", json={}, headers=h)).json()
    r = await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "hi"}, headers=h)
    err = sse(r.text)[-1]
    assert err[0] == "error" and err[1]["code"] == "LLM_UNAVAILABLE"  # auth details stay server-side
    assert "wrong-key" not in r.text and "AUTH" not in r.text


async def test_document_delete_reaches_rag_core(client, stack):
    from app.workers.runner import process_one
    from app.integrations.rag_core import RAGCoreClient
    from app.integrations.storage import get_storage
    from rag_core import db as rag_db
    from rag_core.db import IndexedDocument

    h = await _login(client)
    doc = (await client.post(f"{API}/documents", files={"file": ("w.txt", DOC.encode(), "text/plain")}, headers=h)).json()
    rag = RAGCoreClient("http://rag-core")
    await process_one(rag=rag, storage=get_storage())
    async with rag_db.SessionLocal() as s:
        assert await s.get(IndexedDocument, uuid.UUID(doc["id"])) is not None
    await client.delete(f"{API}/documents/{doc['id']}", headers=h)
    await process_one(rag=rag, storage=get_storage())
    async with rag_db.SessionLocal() as s:
        assert await s.get(IndexedDocument, uuid.UUID(doc["id"])) is None


async def test_rag_core_rejection_code_reaches_the_user(client, stack):
    from app.workers.runner import process_one
    from app.integrations.rag_core import RAGCoreClient
    from app.integrations.storage import get_storage

    h = await _login(client)
    doc = (await client.post(f"{API}/documents", files={"file": ("empty.md", b"   \n\n  ", "text/markdown")}, headers=h)).json()
    await process_one(rag=RAGCoreClient("http://rag-core"), storage=get_storage())
    body = (await client.get(f"{API}/documents/{doc['id']}", headers=h)).json()
    assert body["status"] == "failed" and "No text could be extracted" in body["processing_error"]


async def test_internal_service_token_is_enforced_between_services(stack):
    for app in (stack["agent_app"], stack["rag_app"], stack["llm_app"]):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc") as c:
            assert (await c.get("/metrics")).status_code == 401
            assert (await c.get("/metrics", headers={"Authorization": "Bearer wrong"})).status_code == 401
