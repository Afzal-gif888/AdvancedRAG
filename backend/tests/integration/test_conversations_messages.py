import json

from app.integrations.agent_core import _parse_sse
# pyrefly: ignore [missing-import]
from conftest import create_conversation, parse_sse, upload

API = "/api/v1"


async def test_conversation_crud(client, auth):
    conv = await create_conversation(client, auth)
    assert conv["title"] == "New Chat" and conv["archived_at"] is None

    r = await client.get(f"{API}/conversations/{conv['id']}", headers=auth)
    assert r.status_code == 200

    r = await client.patch(f"{API}/conversations/{conv['id']}", json={"title": "  Renamed   chat "}, headers=auth)
    assert r.json()["title"] == "Renamed chat"
    assert (await client.patch(f"{API}/conversations/{conv['id']}", json={"title": "   "}, headers=auth)).status_code == 422

    r = await client.delete(f"{API}/conversations/{conv['id']}", headers=auth)
    assert r.status_code == 204
    assert (await client.get(f"{API}/conversations/{conv['id']}", headers=auth)).status_code == 404


async def test_list_pagination_sorting_and_archive(client, auth):
    ids = [(await create_conversation(client, auth, f"Chat {i}"))["id"] for i in range(5)]
    r = await client.get(f"{API}/conversations?limit=2&offset=0", headers=auth)
    page = r.json()
    assert page["total"] == 5 and len(page["items"]) == 2 and page["limit"] == 2
    full = (await client.get(f"{API}/conversations", headers=auth)).json()["items"]
    stamps = [c["updated_at"] for c in full]
    assert stamps == sorted(stamps, reverse=True)  # newest first
    assert set(c["id"] for c in full) == set(ids)
    assert "messages" not in page["items"][0]

    r = await client.get(f"{API}/conversations?sort=title&order=asc", headers=auth)
    assert [c["title"] for c in r.json()["items"]] == [f"Chat {i}" for i in range(5)]

    await client.patch(f"{API}/conversations/{ids[0]}", json={"archived": True}, headers=auth)
    assert (await client.get(f"{API}/conversations", headers=auth)).json()["total"] == 4
    archived = (await client.get(f"{API}/conversations?archived=only", headers=auth)).json()
    assert [c["id"] for c in archived["items"]] == [ids[0]]
    assert (await client.get(f"{API}/conversations?q=chat 3", headers=auth)).json()["total"] == 1

    assert (await client.get(f"{API}/conversations?limit=1000", headers=auth)).status_code == 422


async def test_stream_persists_user_and_assistant_messages(client, auth, agent):
    conv = await create_conversation(client, auth)
    r = await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "What is RAG?"}, headers=auth)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    names = [e["event"] for e in events]
    assert names[0] == "message.created" and names[-1] == "message.completed"
    assert names.count("content.delta") == 3
    assert {"agent.started", "retrieval.started", "retrieval.completed", "generation.started"} >= set(names) - {
        "message.created", "content.delta", "generation.completed", "message.completed"
    }
    # Internal fields from Agent Core are never forwarded.
    assert "internal rewrite" not in r.text and "secret" not in r.text and '"internal"' not in r.text
    retrieval = next(e for e in events if e["event"] == "retrieval.completed")
    assert retrieval["data"] == {"retrieved_chunks": 3}

    final = events[-1]["data"]["assistant_message"]
    assert final["content"] == "Hello from the agent." and final["status"] == "completed"
    assert final["citations"] == [{"source_id": None, "document_id": None, "filename": "a.pdf", "page": 2, "chunk_id": None, "snippet": None}]

    msgs = (await client.get(f"{API}/conversations/{conv['id']}/messages", headers=auth)).json()
    assert [(m["role"], m["status"]) for m in msgs["items"]] == [("user", "completed"), ("assistant", "completed")]
    assert msgs["items"][1]["content"] == "Hello from the agent."

    # First message becomes the title.
    assert (await client.get(f"{API}/conversations/{conv['id']}", headers=auth)).json()["title"] == "What is RAG?"
    req = agent.requests[-1]
    assert str(req.conversation_id) == conv["id"] and req.message == "What is RAG?" and req.history == []


async def test_history_is_sent_to_agent(client, auth, agent):
    conv = await create_conversation(client, auth)
    await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "first"}, headers=auth)
    await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "second"}, headers=auth)
    assert agent.requests[-1].history == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "Hello from the agent.", "citations": [{"filename": "a.pdf", "page": 2}]},
    ]


async def test_agent_unavailable_preserves_user_message_and_marks_failure(client, auth, agent):
    agent.mode = "unavailable"
    conv = await create_conversation(client, auth)
    r = await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "hi"}, headers=auth)
    events = parse_sse(r.text)
    assert events[-1]["event"] == "error"
    assert events[-1]["data"]["code"] == "AGENT_SERVICE_UNAVAILABLE"
    assert events[-1]["data"]["message"] == "The AI service is temporarily unavailable."
    assert "localhost" not in r.text

    msgs = (await client.get(f"{API}/conversations/{conv['id']}/messages", headers=auth)).json()["items"]
    assert msgs[0]["role"] == "user" and msgs[0]["content"] == "hi"
    assert msgs[1]["status"] == "failed" and msgs[1]["content"] == "" and msgs[1]["error_code"] == "AGENT_SERVICE_UNAVAILABLE"

    # Safe retry path once the service is back.
    agent.mode = "ok"
    r = await client.post(f"{API}/conversations/{conv['id']}/messages/{msgs[1]['id']}/retry", headers=auth)
    assert parse_sse(r.text)[-1]["event"] == "message.completed"
    msgs = (await client.get(f"{API}/conversations/{conv['id']}/messages", headers=auth)).json()["items"]
    assert len(msgs) == 2 and msgs[1]["status"] == "completed"
    # Completed messages cannot be "retried".
    assert (await client.post(f"{API}/conversations/{conv['id']}/messages/{msgs[1]['id']}/retry", headers=auth)).status_code == 409


async def test_error_event_never_produces_fabricated_answer(client, auth, agent):
    agent.mode = "error_event"
    conv = await create_conversation(client, auth)
    r = await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "hi"}, headers=auth)
    assert parse_sse(r.text)[-1]["data"]["code"] == "GENERATION_FAILED"
    assert "internal:8003" not in r.text
    msgs = (await client.get(f"{API}/conversations/{conv['id']}/messages", headers=auth)).json()["items"]
    assert msgs[1]["status"] == "failed" and msgs[1]["content"] == ""


async def test_empty_agent_output_is_a_failure(client, auth, agent):
    agent.mode = "empty"
    conv = await create_conversation(client, auth)
    r = await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "hi"}, headers=auth)
    assert parse_sse(r.text)[-1]["data"]["code"] == "EMPTY_RESPONSE"


async def test_non_streaming_send(client, auth, agent):
    conv = await create_conversation(client, auth)
    r = await client.post(f"{API}/conversations/{conv['id']}/messages", json={"content": "hello"}, headers=auth)
    assert r.status_code == 201
    body = r.json()
    assert body["user_message"]["content"] == "hello"
    assert body["assistant_message"]["content"] == "Hello from the agent."
    assert body["assistant_message"]["citations"] == [{"source_id": None, "document_id": None, "filename": "a.pdf", "page": 1, "chunk_id": None, "snippet": None}]

    agent.mode = "unavailable"
    r = await client.post(f"{API}/conversations/{conv['id']}/messages", json={"content": "again"}, headers=auth)
    assert r.status_code == 503
    err = r.json()["error"]
    assert err["code"] == "AGENT_SERVICE_UNAVAILABLE" and {"user_message_id", "assistant_message_id"} <= set(err["details"])


async def test_client_cannot_insert_assistant_messages(client, auth, agent):
    conv = await create_conversation(client, auth)
    r = await client.post(f"{API}/conversations/{conv['id']}/messages", json={"role": "assistant", "content": "fake"}, headers=auth)
    assert r.status_code == 422
    assert (await client.post(f"{API}/conversations/{conv['id']}/messages", json={"content": "   "}, headers=auth)).status_code == 422


async def test_message_attachments_are_linked_and_only_ready_docs_sent(client, auth, agent):
    conv = await create_conversation(client, auth)
    doc = (await upload(client, auth)).json()
    r = await client.post(
        f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "summarise", "document_ids": [doc["id"]]}, headers=auth
    )
    created = parse_sse(r.text)[0]["data"]["user_message"]
    assert created["attachments"] == [{"id": doc["id"], "filename": "notes.txt"}]
    linked = (await client.get(f"{API}/conversations/{conv['id']}/documents", headers=auth)).json()
    assert [d["id"] for d in linked] == [doc["id"]]
    # Not processed yet, so not handed to the agent for retrieval.
    assert agent.requests[-1].document_ids == []
    assert agent.requests[-1].context == {"pending_document_count": 1}


async def test_deleting_conversation_cascades_messages(client, auth, agent):
    conv = await create_conversation(client, auth)
    await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "x"}, headers=auth)
    await client.delete(f"{API}/conversations/{conv['id']}", headers=auth)
    assert (await client.get(f"{API}/conversations/{conv['id']}/messages", headers=auth)).status_code == 404


async def test_sse_parser_handles_multiline_and_comments():
    class FakeResponse:
        async def aiter_lines(self):
            for line in [": keepalive", "event: content.delta", 'data: {"delta": "a"}', "", "event: content.delta", "data: raw text", "", "event: x", "data: {}"]:
                yield line

    events = [e async for e in _parse_sse(FakeResponse())]
    assert [(e.event, e.data) for e in events] == [("content.delta", {"delta": "a"}), ("content.delta", {"delta": "raw text"}), ("x", {})]
    assert json.dumps(events[0].data)


async def test_verified_answer_replaces_the_streamed_draft_and_is_what_is_stored(client, auth, agent):
    agent.replace = "I couldn't find sufficient evidence for this answer in your Knowledge Base."
    conv = await create_conversation(client, auth)
    r = await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "q"}, headers=auth)
    events = parse_sse(r.text)
    names = [e["event"] for e in events]
    assert names.index("content.replace") > names.index("content.delta")
    assert next(e for e in events if e["event"] == "content.replace")["data"] == {"content": agent.replace}
    assert events[-1]["data"]["assistant_message"]["content"] == agent.replace
    msgs = (await client.get(f"{API}/conversations/{conv['id']}/messages", headers=auth)).json()["items"]
    assert msgs[1]["content"] == agent.replace  # the unverified draft is never persisted


async def test_history_carries_citations_and_computed_results_of_this_conversation_only(client, auth, agent, make_client):
    """Earlier answers' citations and stored table results go back to Agent Core with the history
    (follow-ups reuse them); results are server-side metadata, never in the browser response."""
    from app.integrations.agent_core import StreamEvent
    from conftest import register_and_login

    result = {"document_id": "11111111-1111-1111-1111-111111111111", "filename": "s.csv", "results": [{"operation": "count"}]}

    async def stream(req):
        agent.requests.append(req)
        yield StreamEvent("content.delta", {"delta": "5000 rows [1]."})
        yield StreamEvent("generation.completed", {"citations": [{"source_id": 1, "filename": "s.csv", "chunk_id": "x:table"}],
                                                   "metadata": {"structured_result": result, "prompt": "never stored"}})

    agent.stream = stream
    conv = await create_conversation(client, auth)
    r = await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "how many rows?"}, headers=auth)
    assert "structured_result" not in r.text  # not exposed to the browser
    await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "give above in table format"}, headers=auth)
    hist = agent.requests[-1].history
    assert hist[1]["role"] == "assistant" and hist[1]["structured"] == result and hist[1]["citations"][0]["chunk_id"] == "x:table"
    assert "structured" not in hist[0]

    other = await create_conversation(client, auth)  # another conversation starts with no history
    await client.post(f"{API}/conversations/{other['id']}/messages/stream", json={"content": "give above in table format"}, headers=auth)
    assert agent.requests[-1].history == []
    intruder = make_client()
    h2 = await register_and_login(intruder, "intruder.history@example.com")
    r = await intruder.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "give above in table format"}, headers=h2)
    assert r.status_code == 404  # another user cannot use (or read) this conversation's results


def test_oversized_structured_results_are_not_stored():
    from app.integrations.agent_core import safe_metadata

    assert "structured_result" not in safe_metadata({"structured_result": {"results": ["x" * 70_000]}})
    assert safe_metadata({"structured_result": {"results": []}, "chain_of_thought": "x"}) == {"structured_result": {"results": []}}
