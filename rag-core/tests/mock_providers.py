"""A real HTTP server emulating Pinecone (control plane, data plane, rerank). It exercises RAG
Core's wire handling without a key or network. Embeddings are produced locally by fastembed."""
import math
import socket
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

PINECONE_KEY = "test-pinecone-key"
DIM = 384  # BAAI/bge-small-en-v1.5


class State:
    def __init__(self):
        self.reset()

    def reset(self):
        self.namespaces = {}
        self.rerank_calls = 0
        self.rerank_fail = False
        self.dimension = DIM
        self.metric = "cosine"
        self.last_query = None


state = State()
pinecone = FastAPI()


def _pc_auth(request: Request):
    return request.headers.get("api-key") == PINECONE_KEY


@pinecone.get("/indexes/{name}")
async def describe(name: str, request: Request):
    if not _pc_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if name != "advancerag-test":
        return JSONResponse({"error": "not found"}, status_code=404)
    host = f"{request.url.scheme}://{request.url.netloc}/data"
    return {"name": name, "dimension": state.dimension, "metric": state.metric, "host": host, "status": {"ready": True}}


@pinecone.post("/data/vectors/upsert")
async def upsert(request: Request):
    body = await request.json()
    ns = state.namespaces.setdefault(body["namespace"], {})
    for v in body["vectors"]:
        assert "text" not in v["metadata"]
        ns[v["id"]] = v
    return {"upsertedCount": len(body["vectors"])}


@pinecone.post("/data/query")
async def query(request: Request):
    body = await request.json()
    state.last_query = body
    allowed = set(body["filter"]["document_id"]["$in"])
    ns = state.namespaces.get(body["namespace"], {})
    q = body["vector"]
    scored = []
    for vid, v in ns.items():
        if v["metadata"]["document_id"] in allowed:
            n = math.sqrt(sum(x * x for x in v["values"])) or 1.0
            scored.append({"id": vid, "score": sum(a * b for a, b in zip(q, v["values"])) / n})
    scored.sort(key=lambda m: m["score"], reverse=True)
    return {"matches": scored[: body["topK"]], "namespace": body["namespace"]}


@pinecone.get("/data/vectors/fetch")
async def fetch(request: Request):
    ns = state.namespaces.get(request.query_params.get("namespace", ""), {})
    ids = request.query_params.getlist("ids")
    return {"vectors": {i: ns[i] for i in ids if i in ns}, "namespace": request.query_params.get("namespace")}


@pinecone.post("/data/vectors/delete")
async def delete(request: Request):
    body = await request.json()
    ns = state.namespaces.get(body["namespace"], {})
    for vid in body["ids"]:
        ns.pop(vid, None)
    return {}


@pinecone.post("/rerank")
async def rerank(request: Request):
    state.rerank_calls += 1
    if state.rerank_fail:
        return JSONResponse({"error": "boom"}, status_code=500)
    body = await request.json()
    terms = set(body["query"].lower().split())
    scores = [(i, sum(d["text"].lower().count(t) for t in terms) + 0.001 * (100 - i)) for i, d in enumerate(body["documents"])]
    scores.sort(key=lambda x: x[1], reverse=True)
    return {"model": body["model"], "data": [{"index": i, "score": s} for i, s in scores[: body["top_n"]]]}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerThread:
    def __init__(self, app, port):
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("server did not start")
            time.sleep(0.02)
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=5)
