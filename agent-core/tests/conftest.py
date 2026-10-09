"""Unit-test fakes: a programmable RAG Core + LLM service behind httpx.MockTransport.
Real-service behaviour is covered by tests/contract/test_agent_core_contracts.py."""
import os

os.environ.update({"ADVANCERAG_ENV_FILE": "none", "ENVIRONMENT": "test", "INTERNAL_SERVICE_TOKEN": "", "LOG_LEVEL": "WARNING"})

import json  # noqa: E402
import uuid  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402

USER = str(uuid.UUID(int=1))
DOC_A = str(uuid.UUID(int=0xA))
DOC_B = str(uuid.UUID(int=0xB))


def chunk(n, doc=DOC_A, filename="policy.pdf", page=3, text="Refunds are processed within five business days.", index=None):
    index = n - 1 if index is None else index
    return {"source_id": n, "chunk_id": f"{doc}:{index}", "document_id": doc, "filename": filename, "page": page,
            "chunk_index": index, "text": text, "score": 0.9 / n}


class FakeBackends:
    def __init__(self):
        self.rag_mode = "ok"        # ok | empty | down | timeout | http_500 | malformed | raw
        self.rag_chunks = [chunk(1)]
        self.rag_raw = None
        self.llm_mode = "ok"        # ok | error:<CODE> | down | timeout | empty | no_done | bad_token
        self.rewrite_mode = "ok"    # ok | error:<CODE> | down | junk
        self.answer = "Refunds take five business days [1]. See also [7]."
        self.rewrite = "standalone refund timing question"
        # Replies to the self-correction checks, by a marker of their system prompt:
        # "EVIDENCE GRADER", "SEARCH QUERY CORRECTION", "ANSWER VERIFIER", "TABLE QUERY PLANNER".
        # A list is consumed one reply per call. Unset checks get `rewrite` (not JSON -> fail-safe).
        self.checks = {}
        self.tables = []            # /tables/describe reply
        self.table_results = None   # /tables/query reply (None -> 422)
        self.retrieve_calls, self.generate_calls, self.stream_calls, self.table_calls = [], [], [], []
        self.headers = []

    def rag(self, request: httpx.Request) -> httpx.Response:
        if self.rag_mode == "down":
            raise httpx.ConnectError("refused")
        if request.method == "GET":
            return httpx.Response(200, json={"status": "ready"})
        self.headers.append(("rag", dict(request.headers)))
        if self.rag_mode == "timeout":
            raise httpx.ReadTimeout("slow")
        body = json.loads(request.content)
        if request.url.path.startswith("/api/v1/rag/tables/"):
            self.table_calls.append((request.url.path.rsplit("/", 1)[-1], body))
            if request.url.path.endswith("/describe"):
                return httpx.Response(200, json={"tables": self.tables})
            if self.table_results is None:
                return httpx.Response(422, json={"error": {"code": "TABLE_QUERY_INVALID"}})
            return httpx.Response(200, json=self.table_results)
        self.retrieve_calls.append(body)
        if self.rag_mode == "http_500":
            return httpx.Response(503, json={"error": {"code": "VECTOR_STORE_UNAVAILABLE"}})
        if self.rag_mode == "malformed":
            return httpx.Response(200, text="<html>proxy error</html>")
        if self.rag_mode == "raw":
            return httpx.Response(200, json=self.rag_raw)
        chunks = [] if self.rag_mode == "empty" else self.rag_chunks
        return httpx.Response(200, json={"chunks": chunks, "context": "(ignored by Agent Core)",
                                         "metadata": {"retrieval_strategy": "hybrid_rrf_rerank", "retrieval_latency_ms": 12.5, "rerank_latency_ms": 3.1}})

    def llm(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(503 if self.llm_mode == "down" else 200, json={"status": "ready"})
        self.headers.append(("llm", dict(request.headers)))
        body = json.loads(request.content)
        if request.url.path == "/v1/generate":
            self.generate_calls.append(body)
            if self.rewrite_mode == "down":
                raise httpx.ConnectError("refused")
            if self.rewrite_mode.startswith("error:"):
                return httpx.Response(503, json={"error": {"code": self.rewrite_mode[6:], "message": "x", "retryable": True}})
            if self.rewrite_mode == "junk":
                return httpx.Response(200, json={"unexpected": True})
            system = body["messages"][0]["content"]
            reply = self.rewrite
            for marker, value in self.checks.items():
                if marker in system:
                    reply = value.pop(0) if isinstance(value, list) else value
                    break
            return httpx.Response(200, json={"content": reply, "model": "m-fast", "finish_reason": "stop", "usage": {}, "latency_ms": 1})
        self.stream_calls.append(body)
        if self.llm_mode == "down":
            raise httpx.ConnectError("refused")
        if self.llm_mode == "timeout":
            raise httpx.ReadTimeout("slow")
        if self.llm_mode.startswith("error:"):
            text = 'event: token\ndata: {"delta": "Partial"}\n\nevent: error\ndata: {"code": "%s", "message": "x", "retryable": true}\n\n' % self.llm_mode[6:]
            return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
        if self.llm_mode == "bad_token":
            return httpx.Response(200, text='event: token\ndata: {"delta": 42}\n\n', headers={"content-type": "text/event-stream"})
        tokens = [] if self.llm_mode == "empty" else [self.answer[i : i + 10] for i in range(0, len(self.answer), 10)]
        text = "".join(f"event: token\ndata: {json.dumps({'delta': t})}\n\n" for t in tokens)
        if self.llm_mode != "no_done":
            text += 'event: done\ndata: {"model": "provider/model-x", "finish_reason": "stop", "usage": {"total_tokens": 42}, "latency_ms": 5}\n\n'
        return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})


@pytest.fixture
def fakes():
    return FakeBackends()


@pytest.fixture
def orchestrator(fakes):
    from agent_core.clients import LLMClient, RAGClient
    from agent_core.config import settings
    from agent_core.orchestrator import Orchestrator

    return Orchestrator(settings, RAGClient(settings, transport=httpx.MockTransport(fakes.rag)), LLMClient(settings, transport=httpx.MockTransport(fakes.llm)))


@pytest.fixture
async def client(orchestrator):
    from agent_core import main

    main.app.dependency_overrides[main.get_orchestrator] = lambda: orchestrator
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://agent") as c:
        yield c
    main.app.dependency_overrides.clear()


def body(**kw):
    base = {"user_id": USER, "conversation_id": str(uuid.UUID(int=2)), "message_id": str(uuid.UUID(int=3)),
            "message": "How long do refunds take?", "document_ids": [DOC_A]}
    return {**base, **kw}


def parse(text):
    out = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out
