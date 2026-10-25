"""Bounded, reproducible load tests against an ISOLATED local stack (never a deployment).

    python scripts/load_test.py --scenario ingest --users 3 --docs-per-user 2 --doc-kb 60
    python scripts/load_test.py --scenario chat   --users 8 --turns 3 --llm-latency-ms 800
    python scripts/load_test.py --scenario mixed  --users 4 --docs-per-user 2 --doc-kb 40 --turns 3
    python scripts/load_test.py --scenario ocr    --users 2 --docs-per-user 1 --pages 3

What it starts: the real four services as local processes on ports 8300-8303 (backend, agent-core, rag-core, llm) against two
throw-away PostgreSQL databases (names must start with `loadtest_`, host must be local; they are dropped and recreated), the
mock gateway (a fixed per-call delay stands in for model latency), the real local embedding model, the in-database vector store
and no Pinecone. Rate limiting is OFF (a handful of users would otherwise trip per-user limits that are not what is measured);
the per-user concurrent-stream cap stays ON.

What it measures: per-request latency percentiles, throughput, error counts by kind, time-to-first-token for chat, upload-to-ready
time for documents, peak memory and mean CPU of every service process, and the job queue's depth over time (sampled from the
database every second). Every result file records the exact configuration and the machine.

What it does NOT show: behaviour on the deployed free-tier hosts (0.1 CPU / 512 MB), real model latency, Pinecone latency,
multi-instance behaviour, or any capacity beyond the sizes actually run. Numbers belong to this machine on this day.
"""
import argparse
import asyncio
import json
import os
import platform
import random
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rag-core" / "tests"))
PORTS = {"backend": 8300, "agent-core": 8301, "rag-core": 8302, "llm": 8303, "mock": 3902}
TOKEN = "loadtest-internal-token-0123456789abcdefghijklmnop"
WORDS = ("refund shipping invoice warranty customer policy delivery account balance contract terms support returns payment order product "
         "service quality standard region office report annual budget review inventory warehouse carrier shipment schedule audit").split()


# ---------------------------------------------------------------- stack

class Stack:
    def __init__(self, args):
        self.args = args
        self.procs = {}
        self.workdir = Path(tempfile.mkdtemp(prefix="advancerag-load-"))

    def pg(self, db):
        return f"{self.args.pg_url.rsplit('/', 1)[0]}/{db}"

    async def recreate_databases(self):
        import asyncpg

        host = self.args.pg_url.split("@")[-1].split("/")[0].split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            sys.exit("refusing: the PostgreSQL host must be local")
        dsn = self.args.pg_url.replace("postgresql+asyncpg://", "postgresql://").rsplit("/", 1)[0] + "/postgres"
        conn = await asyncpg.connect(dsn)
        try:
            for db in ("loadtest_backend", "loadtest_rag"):
                await conn.execute(f'DROP DATABASE IF EXISTS "{db}" WITH (FORCE)')
                await conn.execute(f'CREATE DATABASE "{db}"')
        finally:
            await conn.close()

    def env(self, name):
        a = self.args
        base = {**os.environ, "PYTHONUNBUFFERED": "1", "ENVIRONMENT": "development", "INTERNAL_SERVICE_TOKEN": TOKEN, "LOG_LEVEL": "WARNING",
                "ADVANCERAG_ENV_FILE": "none"}
        if name == "llm":
            return {**base, "FREELLMAPI_BASE_URL": f"http://127.0.0.1:{PORTS['mock']}/v1", "FREELLMAPI_API_KEY": "load-mock-key"}
        if name == "rag-core":
            return {**base, "DATABASE_URL": self.pg("loadtest_rag"), "EMBEDDING_PROVIDER": a.embeddings, "EMBEDDING_DIMENSIONS": "384" if a.embeddings == "fake" else "",
                    "EMBEDDING_CACHE_DIR": str(ROOT / "rag-core" / "var" / "models"), "VECTOR_STORE": "local", "RERANKER": "none",
                    "OCR_ENABLED": "true" if a.scenario == "ocr" else "false", "INGEST_CONCURRENCY": str(a.rag_concurrency),
                    "EMBEDDING_THREADS": str(a.embedding_threads) if a.embedding_threads else ""}
        if name == "agent-core":
            return {**base, "RAG_CORE_URL": f"http://127.0.0.1:{PORTS['rag-core']}", "LLM_SERVICE_URL": f"http://127.0.0.1:{PORTS['llm']}"}
        return {**base, "DATABASE_URL": self.pg("loadtest_backend"), "JWT_SECRET": "load-access-secret-0123456789abcdefghijklmnopqr",
                "JWT_REFRESH_SECRET": "load-refresh-secret-0123456789abcdefghijklmnopq", "AGENT_CORE_URL": f"http://127.0.0.1:{PORTS['agent-core']}",
                "RAG_CORE_URL": f"http://127.0.0.1:{PORTS['rag-core']}", "FRONTEND_ORIGIN": "http://localhost:5199", "LOCAL_STORAGE_PATH": str(self.workdir / "storage"),
                "STORAGE_BACKEND": a.storage, "EMAIL_PROVIDER": "file", "EMAIL_OUTBOX_PATH": str(self.workdir / "outbox"), "RATE_LIMIT_ENABLED": "false",
                "WORKER_POLL_INTERVAL_SECONDS": "0.5", "WORKER_CONCURRENCY": str(a.worker_concurrency), "LOG_JSON": "false",
                "MAX_CONCURRENT_STREAMS_PER_USER": str(a.stream_cap)}

    def wait(self, url, timeout=120):
        import urllib.request

        end = time.time() + timeout
        while time.time() < end:
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status < 500:
                        return
            except Exception:
                time.sleep(0.4)
        raise RuntimeError(f"timed out waiting for {url}")

    def start(self):
        py = sys.executable
        self.procs["mock"] = subprocess.Popen([py, str(ROOT / "scripts" / "mock_freellmapi.py"), "--port", str(PORTS["mock"]), "--key", "load-mock-key",
                                               "--latency-ms", str(self.args.llm_latency_ms)])
        self.wait(f"http://127.0.0.1:{PORTS['mock']}/docs")
        for d in ("rag-core", "backend"):
            subprocess.run([py, "-m", "alembic", "upgrade", "head"], cwd=ROOT / d, env=self.env(d), check=True, capture_output=True)
        for name, app in (("llm", "llm_service.main:app"), ("rag-core", "rag_core.main:app"), ("agent-core", "agent_core.main:app"), ("backend", "app.main:app")):
            d = "llm" if name == "llm" else name
            self.procs[name] = subprocess.Popen([py, "-m", "uvicorn", app, "--host", "127.0.0.1", "--port", str(PORTS[name])], cwd=ROOT / d, env=self.env(name))
            self.wait(f"http://127.0.0.1:{PORTS[name]}/health")
        # RAG Core's /ready needs the embedding model loaded
        self.wait(f"http://127.0.0.1:{PORTS['rag-core']}/ready", timeout=300)

    def stop(self):
        for p in reversed(list(self.procs.values())):
            if p.poll() is None:
                p.terminate()
        for p in self.procs.values():
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
        shutil.rmtree(self.workdir, ignore_errors=True)


# ---------------------------------------------------------------- sampling

class Sampler:
    """Once a second: RSS and CPU of every service process (and its children), and the backend job queue from the database."""

    def __init__(self, stack, pg_url):
        self.stack, self.pg_url = stack, pg_url
        self.rss, self.cpu, self.queue, self.stop_flag = {}, {}, [], False

    async def run(self):
        import asyncpg
        import psutil

        procs = {n: psutil.Process(p.pid) for n, p in self.stack.procs.items() if n != "mock"}
        cache = {}
        for p in procs.values():
            p.cpu_percent(None)
        conn = await asyncpg.connect(self.pg_url.replace("postgresql+asyncpg://", "postgresql://"))
        t0 = time.time()
        try:
            while not self.stop_flag:
                for name, p in procs.items():
                    try:
                        # persistent Process objects: cpu_percent needs a previous call on the SAME object (a fresh object reads 0);
                        # on Windows the venv launcher's real interpreter is a child process.
                        group = [cache.setdefault(c.pid, c) for c in [p] + p.children(recursive=True)]
                        self.rss.setdefault(name, []).append(sum(x.memory_info().rss for x in group) / 1e6)
                        self.cpu.setdefault(name, []).append(sum(x.cpu_percent(None) for x in group))
                    except psutil.Error:
                        pass
                rows = await conn.fetch("select status, count(*) c from background_jobs where status in ('pending','running') group by status")
                counts = {r["status"]: r["c"] for r in rows}
                self.queue.append((round(time.time() - t0, 1), counts.get("pending", 0), counts.get("running", 0)))
                await asyncio.sleep(1)
        finally:
            await conn.close()

    def summary(self):
        return {
            "peak_rss_mb": {n: round(max(v), 1) for n, v in self.rss.items()},
            "mean_cpu_percent_of_one_core": {n: round(statistics.mean(v), 1) for n, v in self.cpu.items()},
            "queue": {"max_pending": max((q[1] for q in self.queue), default=0), "max_running": max((q[2] for q in self.queue), default=0),
                      "samples": len(self.queue), "seconds_with_pending_jobs": sum(1 for q in self.queue if q[1] > 0),
                      "timeline_every_5s": [q for i, q in enumerate(self.queue) if i % 5 == 0][:400]},
        }


# ---------------------------------------------------------------- workloads

def pct(values, p):
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))], 3)


def stats(values):
    return {"n": len(values), "p50": pct(values, 50), "p95": pct(values, 95), "max": round(max(values), 3) if values else None,
            "mean": round(statistics.mean(values), 3) if values else None}


def text_doc(kb, seed):
    r = random.Random(seed)
    out, n = [], 0
    while n < kb * 1024:
        para = " ".join(r.choice(WORDS) + (str(r.randrange(100000)) if r.random() < 0.08 else "") for _ in range(r.randrange(50, 110))) + f". Ref {uuid.uuid4().hex[:10]}.\n\n"
        out.append(para)
        n += len(para)
    return "".join(out).encode()[: kb * 1024]


def scanned_pdf(pages, seed):
    from fixtures import scanned_pdf as make
    from PIL import Image, ImageDraw, ImageFont

    imgs = []
    for p in range(pages):
        img = Image.new("RGB", (1400, 500), "white")
        d = ImageDraw.Draw(img)
        for k in range(4):
            d.text((50, 40 + k * 100), f"Load test page {p + 1} line {k + 1}: the invoice {seed}{p}{k} is due in {10 + k} days.", fill="black", font=ImageFont.load_default(size=34))
        imgs.append(img)
    return make(imgs)


async def register(client):
    r = await client.post("/api/v1/auth/register", json={"email": f"load-{uuid.uuid4().hex[:10]}@example.com", "password": "Load-test-Passw0rd!", "full_name": "Load User"})
    r.raise_for_status()
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def upload_and_wait(client, headers, name, data, mime, timeout, record):
    t0 = time.monotonic()
    try:
        r = await client.post("/api/v1/documents", files={"file": (name, data, mime)}, headers=headers)
        if r.status_code not in (200, 201):
            record["errors"].append(f"upload_http_{r.status_code}")
            return None
        doc = r.json()
        record["upload_seconds"].append(time.monotonic() - t0)
        while time.monotonic() - t0 < timeout:
            s = (await client.get(f"/api/v1/documents/{doc['id']}", headers=headers)).json()
            if s["status"] == "ready":
                record["ready_seconds"].append(time.monotonic() - t0)
                record["chunks_hint"] += len(data) // 900
                return doc["id"]
            if s["status"] == "failed":
                record["errors"].append(f"ingest_failed:{s.get('processing_error')}")
                return None
            await asyncio.sleep(0.5)
        record["errors"].append("ingest_timeout")
    except Exception as exc:
        record["errors"].append(f"upload_exc:{type(exc).__name__}")
    return None


async def ingest_workload(base, args, kind="text"):
    rec = {"upload_seconds": [], "ready_seconds": [], "errors": [], "chunks_hint": 0}
    limits = httpx.Limits(max_connections=200)
    async with httpx.AsyncClient(base_url=base, timeout=httpx.Timeout(600, connect=10), limits=limits) as client:
        users = [await register(client) for _ in range(args.users)]
        sem = asyncio.Semaphore(args.client_concurrency)
        t0 = time.monotonic()

        async def one(u, i, j):
            async with sem:
                if kind == "ocr":
                    return await upload_and_wait(client, u, f"scan{i}_{j}.pdf", scanned_pdf(args.pages, f"{i}{j}"), "application/pdf", args.doc_timeout, rec)
                return await upload_and_wait(client, u, f"doc{i}_{j}.txt", text_doc(args.doc_kb, i * 100 + j), "text/plain", args.doc_timeout, rec)

        ids = await asyncio.gather(*[one(u, i, j) for i, u in enumerate(users) for j in range(args.docs_per_user)])
        wall = time.monotonic() - t0
    ok = sum(1 for x in ids if x)
    return {"documents_submitted": len(ids), "documents_ready": ok, "wall_seconds": round(wall, 2), "docs_per_minute": round(ok / wall * 60, 2) if wall else None,
            "approx_chunks_per_second": round(rec["chunks_hint"] / wall, 2) if wall else None, "upload_seconds": stats(rec["upload_seconds"]),
            "upload_to_ready_seconds": stats(rec["ready_seconds"]), "errors": dict(sorted({e: rec["errors"].count(e) for e in set(rec["errors"])}.items()))}


async def chat_workload(base, args, users_ready=None):
    rec = {"ttft": [], "total": [], "errors": [], "turns": 0, "delta_events": 0}
    async with httpx.AsyncClient(base_url=base, timeout=httpx.Timeout(300, connect=10), limits=httpx.Limits(max_connections=400)) as client:
        users = users_ready or [await register(client) for _ in range(args.users)]
        if not users_ready:  # give every user one small document so retrieval has something to find
            await asyncio.gather(*[upload_and_wait(client, u, "kb.txt", text_doc(8, 900 + i), "text/plain", args.doc_timeout,
                                                    {"upload_seconds": [], "ready_seconds": [], "errors": [], "chunks_hint": 0}) for i, u in enumerate(users)])
        questions = ["What is the refund policy for the carrier?", "When is the warranty review scheduled?", "Summarise the shipment schedule.", "Which region has the annual budget?"]

        async def user_session(i, headers):
            conv = (await client.post("/api/v1/conversations", json={"title": f"load {i}"}, headers=headers)).json()
            for t in range(args.turns):
                t0 = time.monotonic()
                first = None
                try:
                    async with client.stream("POST", f"/api/v1/conversations/{conv['id']}/messages/stream", json={"content": questions[(i + t) % len(questions)]}, headers=headers) as r:
                        if r.status_code != 200:
                            body = (await r.aread()).decode()[:120]
                            rec["errors"].append(f"http_{r.status_code}:{'TOO_MANY_STREAMS' if 'TOO_MANY_STREAMS' in body else 'other'}")
                            continue
                        done = False
                        async for line in r.aiter_lines():
                            if line == "event: content.delta" and first is None:
                                first = time.monotonic() - t0
                            elif line == "event: content.delta":
                                rec["delta_events"] += 1
                            elif line == "event: generation.completed":
                                done = True
                            elif line == "event: error":
                                rec["errors"].append("stream_error_event")
                        if done:
                            rec["turns"] += 1
                            rec["total"].append(time.monotonic() - t0)
                            if first is not None:
                                rec["ttft"].append(first)
                        else:
                            rec["errors"].append("stream_ended_without_completion")
                except Exception as exc:
                    rec["errors"].append(f"exc:{type(exc).__name__}")

        t0 = time.monotonic()
        await asyncio.gather(*[user_session(i, h) for i, h in enumerate(users)])
        wall = time.monotonic() - t0
    return {"users": len(users), "turns_per_user": args.turns, "turns_completed": rec["turns"], "wall_seconds": round(wall, 2),
            "turns_per_minute": round(rec["turns"] / wall * 60, 2) if wall else None, "time_to_first_token_seconds": stats(rec["ttft"]),
            "turn_total_seconds": stats(rec["total"]), "errors": dict(sorted({e: rec["errors"].count(e) for e in set(rec["errors"])}.items()))}


# ---------------------------------------------------------------- main

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["ingest", "chat", "mixed", "ocr"], required=True)
    ap.add_argument("--pg-url", default="postgresql+asyncpg://test@127.0.0.1:55432/postgres", help="admin URL of a LOCAL PostgreSQL server")
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--docs-per-user", type=int, default=2)
    ap.add_argument("--doc-kb", type=int, default=60)
    ap.add_argument("--pages", type=int, default=3, help="pages per scanned PDF (ocr scenario)")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--llm-latency-ms", type=float, default=800.0, help="mock model delay per completion (agent-core makes about 4 per turn)")
    ap.add_argument("--embeddings", choices=["local", "fake"], default="local")
    ap.add_argument("--embedding-threads", type=int, default=0)
    ap.add_argument("--rag-concurrency", type=int, default=1, help="INGEST_CONCURRENCY of RAG Core")
    ap.add_argument("--worker-concurrency", type=int, default=1, help="WORKER_CONCURRENCY of the backend")
    ap.add_argument("--stream-cap", type=int, default=3)
    ap.add_argument("--storage", choices=["local", "db"], default="local")
    ap.add_argument("--client-concurrency", type=int, default=16)
    ap.add_argument("--doc-timeout", type=float, default=1800)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "results"))
    args = ap.parse_args()

    stack = Stack(args)
    await stack.recreate_databases()
    print("starting isolated stack ...", flush=True)
    stack.start()
    signal.signal(signal.SIGINT, lambda *_: (stack.stop(), sys.exit(130)))
    base = f"http://127.0.0.1:{PORTS['backend']}"
    sampler = Sampler(stack, stack.pg("loadtest_backend"))
    sampling = asyncio.create_task(sampler.run())
    result = {"stamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "scenario": args.scenario, "config": {k: v for k, v in vars(args).items() if k not in ("pg_url", "out")},
              "machine": {"platform": platform.platform(), "logical_cpus": os.cpu_count(), "python": platform.python_version()},
              "method": "isolated local stack, throw-away PostgreSQL, mock LLM with a fixed delay, rate limiting off, see the script docstring for limits"}
    try:
        t0 = time.monotonic()
        if args.scenario in ("ingest", "ocr"):
            result["ingestion"] = await ingest_workload(base, args, "ocr" if args.scenario == "ocr" else "text")
        elif args.scenario == "chat":
            result["chat"] = await chat_workload(base, args)
        else:
            # mixed: users with a ready document chat while another set of users uploads
            async with httpx.AsyncClient(base_url=base, timeout=300) as c:
                chatters = [await register(c) for _ in range(args.users)]
                await asyncio.gather(*[upload_and_wait(c, u, "kb.txt", text_doc(8, 700 + i), "text/plain", args.doc_timeout,
                                                        {"upload_seconds": [], "ready_seconds": [], "errors": [], "chunks_hint": 0}) for i, u in enumerate(chatters)])
            t_mixed = time.monotonic()
            ing, chat = await asyncio.gather(ingest_workload(base, args), chat_workload(base, args, users_ready=chatters))
            result["ingestion"], result["chat_while_ingesting"] = ing, chat
            result["mixed_wall_seconds"] = round(time.monotonic() - t_mixed, 2)
        result["total_seconds"] = round(time.monotonic() - t0, 2)
    finally:
        sampler.stop_flag = True
        await asyncio.sleep(1.5)
        sampling.cancel()
        result["resources"] = sampler.summary()
        stack.stop()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{time.strftime('%Y%m%d-%H%M%S')}-load-{args.scenario}{('-' + args.tag) if args.tag else ''}.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("resources",)}, indent=1))
    print(json.dumps({k: v for k, v in result["resources"].items() if k != "queue"}, indent=1), json.dumps({k: v for k, v in result["resources"]["queue"].items() if k != "timeline_every_5s"}))
    print("wrote", path)


if __name__ == "__main__":
    asyncio.run(main())
