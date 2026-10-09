"""What the backend offers Agent Core as the user's Knowledge Base: every one of the caller's own
*ready* documents (never processing/failed/deleted/other users'), newest first, capped, the same
in every conversation and independent of conversation attachments."""
from app.core.config import settings
from app.integrations.rag_core import RAGCoreError
from app.workers.runner import process_one
from conftest import create_conversation, register_and_login, upload

API = "/api/v1"


async def _ready(client, auth, rag, storage, name):
    doc = (await upload(client, auth, name, f"content of {name}".encode())).json()
    while await process_one(rag=rag, storage=storage):
        pass
    return doc["id"]


async def _send(client, auth, conv_id, content="question", **extra):
    r = await client.post(f"{API}/conversations/{conv_id}/messages", json={"content": content, **extra}, headers=auth)
    assert r.status_code == 201, r.text


async def test_knowledge_base_is_every_own_ready_document_in_every_conversation(client, auth, agent, rag, storage, make_client):
    attached = await _ready(client, auth, rag, storage, "attached.txt")
    older = await _ready(client, auth, rag, storage, "older.txt")
    newer = await _ready(client, auth, rag, storage, "newer.txt")
    rag.fail = RAGCoreError("PARSE_FAILED", retryable=False)
    failed = (await upload(client, auth, "broken.txt", b"x")).json()["id"]
    while await process_one(rag=rag, storage=storage):
        pass
    rag.fail = None
    deleted = await _ready(client, auth, rag, storage, "deleted.txt")
    assert (await client.delete(f"{API}/documents/{deleted}", headers=auth)).status_code == 204
    other = make_client()
    other_auth = await register_and_login(other, "someone.else@example.com")
    foreign = await _ready(other, other_auth, rag, storage, "foreign.txt")
    processing = (await upload(client, auth, "processing.txt", b"still processing")).json()["id"]  # never processed

    conv = await create_conversation(client, auth)
    await _send(client, auth, conv["id"], document_ids=[attached])  # older clients may still attach
    req = agent.requests[-1]
    assert req.document_ids == []  # attachments never narrow or widen retrieval
    offered = [d["document_id"] for d in req.library_documents]
    assert offered == [newer, older, attached]  # own and ready, newest first
    for excluded in (processing, failed, deleted, foreign):
        assert excluded not in offered
    assert all(set(d) == {"document_id", "filename"} for d in req.library_documents)
    assert req.context == {"pending_document_count": 1}  # "processing.txt" is not searchable yet
    assert req.to_json()["library_documents"] == req.library_documents

    fresh = await create_conversation(client, auth)  # a new chat sees the same Knowledge Base
    await _send(client, auth, fresh["id"])
    assert [d["document_id"] for d in agent.requests[-1].library_documents] == offered


async def test_deleted_document_leaves_the_knowledge_base(client, auth, agent, rag, storage):
    a = await _ready(client, auth, rag, storage, "a.txt")
    b = await _ready(client, auth, rag, storage, "b.txt")
    assert (await client.delete(f"{API}/documents/{a}", headers=auth)).status_code == 204
    conv = await create_conversation(client, auth)
    await _send(client, auth, conv["id"])
    assert [d["document_id"] for d in agent.requests[-1].library_documents] == [b]


async def test_library_can_be_disabled_and_is_capped(client, auth, agent, rag, storage, monkeypatch):
    ids = [await _ready(client, auth, rag, storage, f"doc{i}.txt") for i in range(3)]
    conv = await create_conversation(client, auth)
    monkeypatch.setattr(settings, "LIBRARY_DISCOVERY_MAX_DOCUMENTS", 2)
    await _send(client, auth, conv["id"])
    assert [d["document_id"] for d in agent.requests[-1].library_documents] == [ids[2], ids[1]]
    monkeypatch.setattr(settings, "LIBRARY_DISCOVERY_ENABLED", False)
    await _send(client, auth, conv["id"])
    assert agent.requests[-1].library_documents == []
