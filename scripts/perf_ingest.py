"""Ingestion size ladder: wall time, chunk count, outcome and peak memory for documents of 100 KB .. 25 MB.

What this measures (and what it does not):
  * The real pipeline (spool file -> parse -> chunk -> embed -> store rows + vectors -> verify) against a
    DISPOSABLE PostgreSQL database and the in-database vector store.
  * Embeddings default to the deterministic FAKE provider so the ladder isolates parsing, chunking, memory and
    database cost. Real-model embedding throughput is measured separately (--real-embed-sample) on a small
    sample; the two are NOT combined into a "total time for 25 MB" claim.
  * Peak memory is the process's peak working set (psutil), sampled after each document.
  * It never talks to Pinecone or any production service. The target database is dropped and recreated.

  python scripts/perf_ingest.py --db postgresql+asyncpg://user@127.0.0.1:55432/ragcore_perf [--real-embed-sample 200]
"""
import argparse
import asyncio
import json
import os
import platform
import random
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rag-core"))
sys.path.insert(0, str(ROOT / "rag-core" / "tests"))
sys.path.insert(0, str(ROOT / "shared"))

os.environ.setdefault("ADVANCERAG_ENV_FILE", "none")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("EMBEDDING_PROVIDER", "fake")
os.environ.setdefault("VECTOR_STORE", "local")
os.environ.setdefault("RERANKER", "none")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "384")
os.environ.setdefault("LOG_LEVEL", "WARNING")

WORDS = ("refund shipping invoice warranty customer policy delivery account balance contract terms support "
         "returns payment order product service quality standard region office report annual budget review").split()


def text_of(size: int, rng: random.Random) -> bytes:
    out, n = [], 0
    while n < size:
        para = " ".join(rng.choice(WORDS) + (str(rng.randrange(1000)) if rng.random() < 0.05 else "") for _ in range(rng.randrange(40, 120))) + ".\n\n"
        out.append(para)
        n += len(para)
    return "".join(out).encode("utf8")[:size]


def peak_mb() -> float:
    import psutil

    info = psutil.Process().memory_info()
    return round(getattr(info, "peak_wset", info.rss) / 1e6, 1)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="a DISPOSABLE postgres database; its tables are dropped and recreated")
    ap.add_argument("--sizes-kb", default="100,1024,5120,10240,25600")
    ap.add_argument("--real-embed-sample", type=int, default=0, help="also time N real bge-small embeddings (chunks/second)")
    args = ap.parse_args()
    if "prod" in args.db.lower():
        sys.exit("refusing: the database name looks like production")
    os.environ["DATABASE_URL"] = args.db  # settings are read at import time

    from rag_core import db as rag_db
    from rag_core import errors, pipeline
    from rag_core.config import settings
    from rag_core.embeddings import DOCUMENT, build_embeddings
    from rag_core.rerank import NoopReranker
    from rag_core.vectorstore import LocalVectorStore

    engine = rag_db.configure_engine(args.db)
    async with engine.begin() as conn:
        await conn.run_sync(rag_db.Base.metadata.drop_all)
        await conn.run_sync(rag_db.Base.metadata.create_all)
    svc = pipeline.Services(settings, build_embeddings(settings), LocalVectorStore(), NoopReranker())
    rng, user, rows = random.Random(7), uuid.uuid4(), []

    for kb in [int(s) for s in args.sizes_kb.split(",")]:
        data = text_of(kb * 1024, rng)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(data)
            path = Path(f.name)
        before = peak_mb()
        t = time.perf_counter()
        row = {"size_kb": kb, "type": "text"}
        try:
            async with rag_db.SessionLocal() as db:
                out = await pipeline.ingest(db, svc, document_id=uuid.uuid4(), user_id=user, filename=f"perf_{kb}.txt", mime_type="text/plain", path=path)
            row.update(outcome="indexed", chunks=out["chunks"])
        except errors.ServiceError as exc:
            row.update(outcome="refused", code=exc.code, retryable=exc.retryable, message=exc.message)
        row["seconds"] = round(time.perf_counter() - t, 2)
        row["peak_rss_mb"] = peak_mb()
        row["peak_before_mb"] = before
        rows.append(row)
        print(json.dumps(row), flush=True)
        path.unlink(missing_ok=True)

    real = None
    if args.real_embed_sample:
        os.environ["EMBEDDING_PROVIDER"] = "local"
        settings.EMBEDDING_PROVIDER = "local"
        emb = build_embeddings(settings)
        await emb.warm_up()
        sample = [text_of(1000, rng).decode() for _ in range(args.real_embed_sample)]
        t = time.perf_counter()
        for i in range(0, len(sample), 32):
            await emb.embed(sample[i:i + 32], DOCUMENT)
        dt = time.perf_counter() - t
        real = {"model": emb.model, "chunks": len(sample), "chunk_chars": 1000, "seconds": round(dt, 2), "chunks_per_second": round(len(sample) / dt, 1)}
        print(json.dumps({"real_embedding_sample": real}), flush=True)

    result = {
        "stamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "machine": {"platform": platform.platform(), "cpus": os.cpu_count(), "python": platform.python_version()},
        "limits": {"MAX_FILE_BYTES": settings.MAX_FILE_BYTES, "MAX_CHUNKS_PER_DOCUMENT": settings.MAX_CHUNKS_PER_DOCUMENT,
                   "CHUNK_SIZE": settings.CHUNK_SIZE, "CHUNK_OVERLAP": settings.CHUNK_OVERLAP},
        "embedding_for_ladder": "fake (deterministic, near-zero cost)", "ladder": rows, "real_embedding_sample": real,
    }
    out_path = ROOT / "evaluation" / "results" / f"{time.strftime('%Y%m%d-%H%M%S')}-perf-ingest.json"
    out_path.write_text(json.dumps(result, indent=1), encoding="utf8")
    print("wrote", out_path)


if __name__ == "__main__":
    asyncio.run(main())
