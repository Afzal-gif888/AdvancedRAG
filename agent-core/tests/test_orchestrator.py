"""Orchestration through the HTTP API with programmable fakes for RAG Core and the LLM service
(unit level). Real-service behaviour: tests/contract/test_agent_core_contracts.py."""
import asyncio
import json

import httpx
import pytest

from conftest import DOC_A, DOC_B, USER, body, chunk, parse

GRADE_OK = json.dumps({"verdict": "sufficient", "relevant": [1], "conflict": False, "question_type": "document"})
VERIFIED = json.dumps({"supported": True, "unsupported": [], "revised": ""})


def names(events):
    return [n for n, _ in events]


async def stream(client, **kw):
    r = await client.post("/api/v1/agent/stream", json=body(**kw), headers=kw.pop("_headers", None) or {})
    assert r.status_code == 200
    return parse(r.text)


# TEST 1 + 5 ----------------------------------------------------------------
async def test_answerable_document_question_streams_grounded_answer_with_citations(client, fakes):
    fakes.checks = {"EVIDENCE GRADER": GRADE_OK, "ANSWER VERIFIER": VERIFIED}
    events = await stream(client)
    assert names(events)[:4] == ["agent.started", "retrieval.started", "retrieval.completed", "generation.started"]
    assert names(events)[-1] == "generation.completed"
    assert "".join(d["delta"] for n, d in events if n == "content.delta") == fakes.answer
    # [7] was never supplied: the marker is removed and the corrected text replaces the draft.
    assert ("content.replace", {"content": "Refunds take five business days [1]. See also."}) in events
    done = events[-1][1]
    assert done["citations"] == [{"source_id": 1, "document_id": DOC_A, "filename": "policy.pdf", "page": 3, "chunk_id": f"{DOC_A}:0",
                                  "snippet": "Refunds are processed within five business days."}]
    meta = done["metadata"]
    assert meta["retrieved_chunks"] == 1 and meta["citation_count"] == 1 and meta["verification_status"] == "verified"
    assert meta["retrieval_strategy"] == "hybrid_rrf_rerank" and meta["model"] == "provider/model-x"
    # RAG got the user scope, the backend-verified documents and the frozen top-k.
    assert fakes.retrieve_calls == [{"user_id": USER, "query": "How long do refunds take?", "document_ids": [DOC_A], "top_k": 6}]
    user_msg = fakes.stream_calls[0]["messages"][-1]["content"]
    assert '<source id="1" document="policy.pdf" page="3">' in user_msg and user_msg.endswith("Question: How long do refunds take?")


# TEST 18 -------------------------------------------------------------------
async def test_multiple_documents_produce_multiple_correct_citations(client, fakes):
    fakes.rag_chunks = [chunk(1, text="Refunds: 5 days."), chunk(2, doc=DOC_B, filename="faq.docx", page=None, text="Shipping is free.")]
    fakes.answer = "Refunds take 5 days [1] and shipping is free [2]."
    done = (await stream(client, document_ids=[DOC_A, DOC_B]))[-1][1]
    assert [(c["source_id"], c["filename"], c["page"], c["chunk_id"]) for c in done["citations"]] == [
        (1, "policy.pdf", 3, f"{DOC_A}:0"), (2, "faq.docx", None, f"{DOC_B}:1")]


# TEST 3 --------------------------------------------------------------------
async def test_general_question_does_not_call_rag(client, fakes):
    fakes.answer = "Python is a programming language."
    events = await stream(client, message="What is Python?", document_ids=[])
    assert "retrieval.started" not in names(events) and fakes.retrieve_calls == [] and fakes.generate_calls == []
    done = events[-1][1]
    assert done["citations"] == [] and done["metadata"]["retrieval_strategy"] == "none"
    assert done["metadata"]["verification_status"] == "not_applicable"
    assert "<evidence>" not in fakes.stream_calls[0]["messages"][-1]["content"]


# TEST 2 + 17 ---------------------------------------------------------------
async def test_empty_retrieval_instructs_the_model_to_decline(client, fakes):
    fakes.rag_mode = "empty"
    fakes.answer = "The provided documents do not contain information about dental insurance."
    events = await stream(client, message="What is the dental insurance policy?")
    assert events[2] == ("retrieval.completed", {"retrieved_chunks": 0})
    system = fakes.stream_calls[0]["messages"][0]["content"]
    assert "contains no relevant passages" in system and "reply with exactly this sentence" in system
    done = events[-1][1]
    assert done["citations"] == [] and done["metadata"]["verification_status"] == "no_relevant_documents"


async def test_answer_without_citations_is_flagged_uncited(client, fakes):
    fakes.checks = {"EVIDENCE GRADER": GRADE_OK, "ANSWER VERIFIER": VERIFIED}
    fakes.answer = "The documents do not mention the Bluetooth range."
    done = (await stream(client, message="What is the Bluetooth range?"))[-1][1]
    assert done["citations"] == [] and done["metadata"]["verification_status"] == "uncited"


# TEST 4 --------------------------------------------------------------------
async def test_follow_up_retrieves_with_the_resolved_query(client, fakes):
    history = [{"role": "user", "content": "What is the remote work policy?"},
               {"role": "assistant", "content": "Up to three days per week [1]."}]
    fakes.rewrite = "remote work policy working from an apartment"
    events = await stream(client, message="Can I do that from my apartment?", history=history)
    assert fakes.retrieve_calls[0]["query"] == "remote work policy working from an apartment"
    assert "remote work policy" in fakes.generate_calls[0]["messages"][-1]["content"]
    msgs = fakes.stream_calls[0]["messages"]
    assert msgs[1:3] == [history[0], {"role": "assistant", "content": "Up to three days per week."}]
    assert msgs[-1]["content"].endswith("Question: Can I do that from my apartment?")  # the user's own words are answered
    assert events[-1][1]["metadata"]["query_rewritten"] is True


async def test_follow_up_survives_a_failed_rewrite(client, fakes):
    fakes.rewrite_mode = "junk"
    history = [{"role": "user", "content": "What is the remote work policy?"}, {"role": "assistant", "content": "Three days."}]
    await stream(client, message="Can I do that from my apartment?", history=history)
    assert fakes.retrieve_calls[0]["query"] == "What is the remote work policy?\nCan I do that from my apartment?"


# TEST 19 -------------------------------------------------------------------
async def test_long_irrelevant_history_is_bounded(client, fakes):
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"unrelated chat {i} " + "z" * 3000} for i in range(200)]
    await stream(client, message="How long do refunds take?", history=history)
    msgs = fakes.stream_calls[0]["messages"]
    assert len(msgs) <= 12 + 2 and sum(len(m["content"]) for m in msgs[1:-1]) <= 24_000
    assert msgs[-1]["content"].endswith("Question: How long do refunds take?")


# TEST 6 / 7 / 8 ------------------------------------------------------------
@pytest.mark.parametrize("mode,reason", [("malformed", "rag_invalid_response"), ("timeout", "rag_timeout"),
                                         ("down", "rag_unreachable"), ("http_500", "rag_http_error_503:VECTOR_STORE_UNAVAILABLE")])
async def test_rag_failures_are_safe_and_never_answered(client, fakes, caplog, mode, reason):
    fakes.rag_mode = mode
    caplog.set_level("INFO", logger="advancerag.agent.orchestrator")
    events = await stream(client)
    assert events[-1] == ("error", {"code": "RAG_SERVICE_UNAVAILABLE", "message": "Document search is unavailable."})
    assert fakes.stream_calls == [] and "content.delta" not in names(events)
    assert any(getattr(r, "reason", None) == reason for r in caplog.records)  # precise cause kept server-side


async def test_foreign_document_chunk_is_rejected(client, fakes):
    """TEST 16 (unit level): even if RAG Core returned a chunk from a document the backend did not
    authorise for this turn, Agent Core refuses to use it."""
    fakes.rag_chunks = [chunk(1, doc=DOC_B)]
    events = await stream(client, document_ids=[DOC_A])
    assert events[-1][1]["code"] == "RAG_SERVICE_UNAVAILABLE" and fakes.stream_calls == []


# TEST 9 / 10 / 11 / 20 -----------------------------------------------------
@pytest.mark.parametrize("mode,code", [
    ("down", "LLM_UNAVAILABLE"), ("timeout", "LLM_TIMEOUT"), ("error:LLM_RATE_LIMITED", "LLM_RATE_LIMITED"),
    ("error:LLM_AUTH_FAILED", "LLM_UNAVAILABLE"), ("error:LLM_INVALID_RESPONSE", "LLM_UNAVAILABLE"),
    ("bad_token", "LLM_UNAVAILABLE"), ("no_done", "LLM_UNAVAILABLE"), ("empty", "GENERATION_FAILED"),
])
async def test_llm_failures_map_to_stable_codes_with_one_call(client, fakes, mode, code):
    fakes.llm_mode = mode
    events = await stream(client, document_ids=[])
    assert events[-1][0] == "error" and events[-1][1]["code"] == code
    assert len(fakes.stream_calls) == 1  # Agent Core never retries (no retry storm)
    assert "AUTH" not in events[-1][1]["message"]


async def test_rate_limited_follow_up_rewrite_fails_fast_without_generation(client, fakes):
    fakes.rewrite_mode = "error:LLM_RATE_LIMITED"
    events = await stream(client, message="and that?", history=[{"role": "user", "content": "refunds?"}])
    assert events[-1] == ("error", {"code": "LLM_RATE_LIMITED", "message": "The AI service is temporarily rate-limited. Please try again shortly."})
    assert len(fakes.generate_calls) == 1 and fakes.stream_calls == [] and fakes.retrieve_calls == []


async def test_turn_deadline_is_enforced(client, fakes, monkeypatch):
    from agent_core.config import settings

    monkeypatch.setattr(settings, "AGENT_TURN_TIMEOUT_SECONDS", 0.000001)
    events = await stream(client, document_ids=[])
    assert events[-1][1]["code"] == "LLM_TIMEOUT"


async def test_unexpected_bug_becomes_a_safe_error_event(client, fakes, monkeypatch):
    from agent_core import orchestrator

    def boom(*a, **k):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(orchestrator, "build_evidence", boom)
    events = await stream(client)
    assert events[-1] == ("error", {"code": "GENERATION_FAILED", "message": "The response could not be generated."})


# TEST 12 / 13 / 14 ---------------------------------------------------------
async def test_streaming_event_order_and_completion(client, fakes):
    events = await stream(client)
    order = names(events)
    assert order.index("generation.started") < order.index("content.delta") < order.index("generation.completed") == len(order) - 1


async def test_streaming_failure_ends_with_error_event(client, fakes):
    fakes.llm_mode = "error:LLM_STREAM_INTERRUPTED"
    events = await stream(client, document_ids=[])
    assert names(events)[-2:] == ["content.delta", "error"] and events[-1][1]["code"] == "LLM_UNAVAILABLE"


async def test_client_disconnect_closes_upstream_llm_stream(fakes):
    """Closing the agent stream (client went away) closes the LLM service response."""
    from agent_core.clients import LLMClient, RAGClient
    from agent_core.config import settings
    from agent_core.orchestrator import Orchestrator
    from agent_core.schemas import AgentRequest

    closed = asyncio.Event()

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'event: token\ndata: {"delta": "first"}\n\n'
            await asyncio.sleep(30)  # the model is still generating
            yield b'event: done\ndata: {}\n\n'

        async def aclose(self):
            closed.set()

    def llm(request):
        return httpx.Response(200, stream=SlowStream(), headers={"content-type": "text/event-stream"})

    orch = Orchestrator(settings, RAGClient(settings, transport=httpx.MockTransport(fakes.rag)), LLMClient(settings, transport=httpx.MockTransport(llm)))
    gen = orch.stream(AgentRequest.model_validate(body(document_ids=[])))
    seen = []
    async for name, data in gen:
        seen.append(name)
        if name == "content.delta":
            break
    await gen.aclose()
    assert seen[-1] == "content.delta"
    await asyncio.wait_for(closed.wait(), timeout=2)  # upstream stream released


async def test_cancellation_during_retrieval_cancels_the_rag_call(fakes):
    from agent_core.clients import LLMClient, RAGClient
    from agent_core.config import settings
    from agent_core.orchestrator import Orchestrator
    from agent_core.schemas import AgentRequest

    started, cancelled = asyncio.Event(), asyncio.Event()

    class SlowRag(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise

    orch = Orchestrator(settings, RAGClient(settings, transport=SlowRag()), LLMClient(settings, transport=httpx.MockTransport(fakes.llm)))

    async def consume():
        async for _ in orch.stream(AgentRequest.model_validate(body())):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set() and fakes.stream_calls == []  # no LLM call after cancellation


# Correlation, auth, health ------------------------------------------------
async def test_request_id_propagates_to_rag_and_llm(client, fakes):
    history = [{"role": "user", "content": "refunds?"}]
    r = await client.post("/api/v1/agent/stream", json=body(history=history), headers={"X-Request-ID": "trace-12345678"})
    assert r.headers["x-request-id"] == "trace-12345678"
    assert {h["x-request-id"] for _, h in fakes.headers} == {"trace-12345678"}
    assert {svc for svc, _ in fakes.headers} == {"rag", "llm"}


async def test_invoke_contract(client, fakes):
    r = await client.post("/api/v1/agent/invoke", json=body())
    assert r.status_code == 200
    out = r.json()
    assert out["content"] == "Refunds take five business days [1]. See also." and len(out["citations"]) == 1 and out["metadata"]["citation_count"] == 1
    fakes.llm_mode = "error:LLM_RATE_LIMITED"
    r = await client.post("/api/v1/agent/invoke", json=body())
    assert r.status_code == 503 and r.json()["error"]["code"] == "LLM_RATE_LIMITED" and r.json()["error"]["retryable"] is True


async def test_pending_documents_note(client, fakes):
    await stream(client, context={"pending_document_count": 2})
    assert "2 document(s) in the Knowledge Base are still being processed" in fakes.stream_calls[0]["messages"][0]["content"]


async def test_request_validation(client):
    for bad in (body(message=""), body(user_id="u1"), body(document_ids=["doc-1"]), body(history=[{"role": "system", "content": "x"}])):
        assert (await client.post("/api/v1/agent/stream", json=bad)).status_code == 422
    assert (await client.get("/health")).json() == {"status": "ok"}


async def test_internal_token_required_when_configured(monkeypatch, fakes):
    """The real app wiring: with a token configured, calls without it are rejected."""
    from fastapi import Depends, FastAPI

    from advancerag_common.service import internal_auth

    app = FastAPI()
    app.add_api_route("/x", lambda: {"ok": True}, dependencies=[Depends(internal_auth("t" * 40))])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://a") as c:
        assert (await c.get("/x")).status_code == 401
        assert (await c.get("/x", headers={"Authorization": "Bearer " + "t" * 40})).status_code == 200


async def test_ready_reports_dependencies_without_generating(client, fakes):
    r = await client.get("/ready")
    assert r.status_code == 200 and r.json() == {"status": "ready", "checks": {"rag_core": True, "llm": True}}
    fakes.rag_mode = "down"
    r = await client.get("/ready")
    assert r.status_code == 503 and r.json()["checks"] == {"rag_core": False, "llm": True}
    assert fakes.stream_calls == [] and fakes.generate_calls == [] and fakes.retrieve_calls == []


def test_no_provider_keys_or_secrets_in_agent_config():
    from agent_core.config import Settings

    fields = set(Settings.model_fields)
    assert not {f for f in fields if any(p in f for p in ("OPENROUTER", "GROQ", "GOOGLE", "DEEPSEEK", "FREELLMAPI", "PINECONE"))}


def test_production_requires_service_token():
    from agent_core.config import ConfigError, Settings

    with pytest.raises(ConfigError):
        Settings(ENVIRONMENT="production", RAG_CORE_URL="https://rag.internal", LLM_SERVICE_URL="https://llm.internal", _env_file=None).check_safety()
    with pytest.raises(ConfigError):
        Settings(ENVIRONMENT="production", INTERNAL_SERVICE_TOKEN="t" * 40, _env_file=None).check_safety()  # localhost URLs
    Settings(ENVIRONMENT="production", INTERNAL_SERVICE_TOKEN="t" * 40, RAG_CORE_URL="https://rag.internal", LLM_SERVICE_URL="https://llm.internal", _env_file=None).check_safety()


def test_frozen_retrieval_parameters(monkeypatch):
    """Frozen RAG baseline (docs/rag-architecture.md): change only with a new benchmark run."""
    from agent_core.config import Settings

    for name in ("RETRIEVAL_TOP_K", "QUERY_REWRITE_ENABLED", "MAX_HISTORY_CHARS", "MAX_CONTEXT_CHARS"):
        monkeypatch.delenv(name, raising=False)
    s = Settings(_env_file=None, ENVIRONMENT="test")
    assert (s.RETRIEVAL_TOP_K, s.QUERY_REWRITE_ENABLED, s.MAX_HISTORY_CHARS, s.MAX_CONTEXT_CHARS) == (6, True, 24_000, 12_000)
    with pytest.raises(ValueError):
        Settings(_env_file=None, ENVIRONMENT="test", MAX_CONTEXT_CHARS=20_000)  # can never exceed RAG Core's cap
