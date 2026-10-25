"""Reproducible retrieval evaluation against the real RAG Core pipeline.

    python scripts/run_retrieval_eval.py --db postgresql+asyncpg://USER@HOST:PORT/advancerag_eval \
        --strategies recursive,structure,semantic --ocr --rerank-local

What it does
  1. Ingests the synthetic corpus (evaluation/dataset/corpus) through rag_core.pipeline.ingest with the REAL
     embedding model (BAAI/bge-small-en-v1.5, 384-d) and the in-database vector store, once per chunking
     strategy, each under its own user (so the runs cannot see each other).
  2. For every labelled question computes four rankings: dense (cosine), keyword (PostgreSQL full-text
     ts_rank_cd), RRF of the two (k=60, the production fusion), and a reference BM25. Optionally a local
     cross-encoder rerank of the RRF candidates (NOT the production reranker; see the note in the output).
  3. Scores them with the metrics of rag_core.evaluation and writes evaluation/results/<stamp>.json.

Safety: it needs a database you pass explicitly, drops and recreates the RAG Core tables in it, and refuses any
database whose name does not contain "eval". It never touches Pinecone, the LLM gateway or any .env file.
What it cannot measure: the production Pinecone index / hosted reranker (those need real credentials), and
answer quality (that needs an LLM). Results describe THIS corpus on THIS machine, not production.
"""
import argparse
import asyncio
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "rag-core"), str(ROOT / "shared")]

MIME = {".md": "text/markdown", ".txt": "text/plain", ".csv": "text/csv", ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="SQLAlchemy URL of a DISPOSABLE PostgreSQL database whose name contains 'eval'")
    ap.add_argument("--strategies", default="recursive", help="comma separated: recursive,structure,semantic")
    ap.add_argument("--ocr", action="store_true", help="enable OCR so the scanned-page questions can be answered")
    ap.add_argument("--rerank-local", action="store_true", help="also score a local cross-encoder rerank (indicative only)")
    ap.add_argument("--pdf-tables", choices=["off", "lines", "auto"], default="off", help="PDF_TABLES setting used when ingesting the corpus")
    ap.add_argument("--experiments", action="store_true", help="also score keyword-ranking variants (idf, tf, and_then_or) and weighted fusion")
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "results"))
    ap.add_argument("--tag", default="")
    return ap.parse_args()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


async def main() -> int:
    args = parse_args()
    if "eval" not in args.db.rsplit("/", 1)[-1]:
        print("refusing: the database name must contain 'eval' (this script drops tables)", file=sys.stderr)
        return 2
    os.environ.update({"ADVANCERAG_ENV_FILE": "none", "ENVIRONMENT": "test", "DATABASE_URL": args.db, "EMBEDDING_PROVIDER": "local",
                       "VECTOR_STORE": "local", "RERANKER": "none", "LOG_LEVEL": "WARNING", "INTERNAL_SERVICE_TOKEN": ""})
    os.environ.setdefault("EMBEDDING_CACHE_DIR", str(ROOT / "rag-core" / "var" / "models"))

    from sqlalchemy import select

    from rag_core import db as rag_db
    from rag_core import evaluation as ev
    from rag_core import ocr as ocr_mod
    from rag_core import pipeline
    from rag_core.config import settings
    from rag_core.db import Chunk, IndexedDocument
    from rag_core.embeddings import QUERY, build_embeddings
    from rag_core.rerank import NoopReranker
    from rag_core.vectorstore import LocalVectorStore

    settings.OCR_ENABLED = bool(args.ocr)
    settings.PDF_TABLES = args.pdf_tables
    engine = rag_db.configure_engine(args.db)
    async with engine.begin() as conn:
        await conn.run_sync(rag_db.Base.metadata.drop_all)
        await conn.run_sync(rag_db.Base.metadata.create_all)
        await conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_rag_chunks_text_fts ON rag_chunks USING GIN (to_tsvector('english', text))")

    embeddings = build_embeddings(settings)
    await embeddings.warm_up()
    svc = pipeline.Services(settings, embeddings, LocalVectorStore(), NoopReranker())

    questions = ev.load_questions(ROOT / "evaluation" / "dataset" / "questions.jsonl")
    manifest = json.loads((ROOT / "evaluation" / "dataset" / "manifest.json").read_text(encoding="utf8"))
    corpus = ROOT / "evaluation" / "dataset" / "corpus"

    cross = None
    if args.rerank_local:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        cross = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2", cache_dir=str(ROOT / "rag-core" / "var" / "models"))

    results = {"meta": {
        "pdf_tables": args.pdf_tables, "experiments": bool(args.experiments),
        "stamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "commit": git_commit(), "dataset_version": manifest["version"],
        "dataset_files_sha256": {k: v["sha256"][:12] for k, v in manifest["files"].items()}, "embedding_model": embeddings.model,
        "dimensions": embeddings.dimensions, "chunk_size": settings.CHUNK_SIZE, "chunk_overlap": settings.CHUNK_OVERLAP,
        "ocr_enabled": settings.OCR_ENABLED, "vector_store": "in-database brute-force cosine (same metric as the Pinecone index)",
        "keyword": "PostgreSQL full-text ts_rank_cd (production code path pipeline._keyword)",
        "reranker": "none in the pipeline rows; the optional 'rerank_local' row uses Xenova/ms-marco-MiniLM-L-6-v2 (NOT bge-reranker-v2-m3)" if cross else "not evaluated",
        "machine": {"platform": platform.platform(), "cpus": os.cpu_count(), "python": platform.python_version()},
        "note": "Synthetic corpus, one run per strategy, no tuning on dev or test. Not a statement about production quality.",
    }, "strategies": {}}

    for strategy in [s.strip() for s in args.strategies.split(",") if s.strip()]:
        settings.CHUNKING_STRATEGY = strategy
        user = uuid.uuid4()
        doc_ids, ingest_stats = {}, {"chunks": 0, "embedding_texts": 0, "ingest_seconds": 0.0}
        for path in sorted(p for p in corpus.rglob("*") if p.is_file()):
            name = path.relative_to(corpus).as_posix()
            if name == "scan.pdf" and not args.ocr:
                continue
            did = uuid.uuid4()
            doc_ids[name] = did
            t = time.perf_counter()
            async with rag_db.SessionLocal() as db:
                out = await pipeline.ingest(db, svc, document_id=did, user_id=user, filename=name, mime_type=MIME[path.suffix], path=path)
            ingest_stats["ingest_seconds"] += time.perf_counter() - t
            ingest_stats["chunks"] += out["chunks"]
            ingest_stats["embedding_texts"] += out["chunks"] + out["timings"].get("chunking_embedding_texts", 0)
        ingest_stats["ingest_seconds"] = round(ingest_stats["ingest_seconds"], 1)

        async with rag_db.SessionLocal() as db:
            rows = (await db.execute(select(Chunk.id, Chunk.text, Chunk.section, IndexedDocument.filename)
                                     .join(IndexedDocument, IndexedDocument.document_id == Chunk.document_id).where(Chunk.user_id == user))).all()
        info = {r[0]: (r[3], r[1]) for r in rows}
        bm25 = ev.BM25Index({cid: text for cid, (_, text) in info.items()})
        all_docs = list(doc_ids.values())

        variants = ("kw_idf", "rrf_idf", "kw_tf", "rrf_tf", "kw_andor", "rrf_andor", "rrf_w05") if args.experiments else ()
        methods = ("dense", "keyword", "rrf", "bm25", "bm25_rrf") + variants + (("rerank_local",) + (("rerank_local_idf",) if args.experiments else ()) if cross else ())
        per_q = {m: [] for m in methods}
        lat = {m: lat_list for m, lat_list in ((m, []) for m in methods)}
        top_dense_score = {"answerable": [], "unanswerable": []}
        scored_questions = []
        for q in questions:
            if q.get("requires") == "ocr" and not args.ocr:
                continue
            t0 = time.perf_counter()
            qvec = (await svc.embeddings.embed([q["question"]], QUERY))[0]
            embed_ms = (time.perf_counter() - t0) * 1000
            async with rag_db.SessionLocal() as db:
                t = time.perf_counter()
                dense = await svc.vectors.query(user, qvec, 20, all_docs, db=db)
                dense_ms = (time.perf_counter() - t) * 1000 + embed_ms
                t = time.perf_counter()
                kw = await pipeline._keyword(db, user, q["question"], all_docs, 20)
                kw_ms = (time.perf_counter() - t) * 1000
                alt = {}
                if args.experiments:
                    for tag, mode in (("idf", "idf"), ("tf", "tf"), ("andor", "and_then_or")):
                        t = time.perf_counter()
                        alt[tag] = await pipeline._keyword(db, user, q["question"], all_docs, 20, mode=mode)
                        lat.setdefault(f"kw_{tag}", []).append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            bm = bm25.search(q["question"], 20)
            bm_ms = (time.perf_counter() - t) * 1000
            fused = pipeline.rrf([[c for c, _ in dense], [c for c, _ in kw]], settings.RRF_K)[:20]
            bm_fused = pipeline.rrf([[c for c, _ in dense], [c for c, _ in bm]], settings.RRF_K)[:20]
            rankings = {"dense": [c for c, _ in dense], "keyword": [c for c, _ in kw], "rrf": [c for c, _ in fused],
                        "bm25": [c for c, _ in bm], "bm25_rrf": [c for c, _ in bm_fused]}
            lat["dense"].append(dense_ms); lat["keyword"].append(kw_ms); lat["bm25"].append(bm_ms)
            lat["rrf"].append(dense_ms + kw_ms); lat["bm25_rrf"].append(dense_ms + bm_ms)
            if args.experiments:
                dense_ids = [c for c, _ in dense]
                for tag in ("idf", "tf", "andor"):
                    rankings[f"kw_{tag}"] = [c for c, _ in alt[tag]]
                    rankings[f"rrf_{tag}"] = [c for c, _ in pipeline.rrf([dense_ids, rankings[f"kw_{tag}"]], settings.RRF_K)[:20]]
                    lat[f"rrf_{tag}"].append(dense_ms + lat[f"kw_{tag}"][-1])
                rankings["rrf_w05"] = [c for c, _ in pipeline.rrf([dense_ids, [c for c, _ in kw]], settings.RRF_K, [1.0, 0.5])[:20]]
                lat["rrf_w05"].append(dense_ms + kw_ms)
            if cross:
                for name, base in (("rerank_local", "rrf"),) + ((("rerank_local_idf", "rrf_idf"),) if args.experiments else ()):
                    t = time.perf_counter()
                    cand = rankings[base]
                    scores = list(cross.rerank(q["question"], [info[c][1] for c in cand]))
                    order = sorted(range(len(cand)), key=lambda i: -scores[i])
                    rankings[name] = [cand[i] for i in order]
                    lat[name].append(lat[base][-1] + (time.perf_counter() - t) * 1000)
            if q["answerable"]:
                top_dense_score["answerable"].append(dense[0][1] if dense else 0.0)
                for m in methods:
                    per_q[m].append({"qid": q["id"], "type": q["type"], "split": q["split"], **ev.score_question([info[c] for c in rankings[m]], q["relevant"])})
                scored_questions.append(q["id"])
            else:
                top_dense_score["unanswerable"].append(dense[0][1] if dense else 0.0)

        def summarise(rows):
            if not rows:
                return {}
            a = ev.aggregate([{k: v for k, v in r.items() if isinstance(v, float)} for r in rows])
            n = len(rows)
            lo, hi = ev.wilson(sum(r["hit@6"] for r in rows), n)
            mlo, mhi = ev.bootstrap_ci([r["rr"] for r in rows])
            return {"n": n, **{k: round(v, 4) for k, v in a.items() if k.split("@")[-1] in ("1", "3", "6", "20") or k == "rr"},
                    "hit@6_ci95": [round(lo, 3), round(hi, 3)], "mrr_ci95": [round(mlo, 3), round(mhi, 3)]}

        by_method = {}
        for m in methods:
            rows = per_q[m]
            by_type = {t: summarise([r for r in rows if r["type"] == t]) for t in sorted({r["type"] for r in rows})}
            by_split = {s: summarise([r for r in rows if r["split"] == s]) for s in ("dev", "test")}
            by_method[m] = {"overall": summarise(rows), "by_type": by_type, "by_split": by_split,
                            "per_question": {r["qid"]: [round(r["rr"], 4), r["hit@1"], r["hit@6"], r["recall@20"], r["type"], r["split"]] for r in rows},
                            "latency_ms": {"p50": round(ev.percentile(lat[m], 50), 1), "p95": round(ev.percentile(lat[m], 95), 1), "n": len(lat[m])}}
        auc = ev.auc(top_dense_score["answerable"], top_dense_score["unanswerable"])
        results["strategies"][strategy] = {
            "ingest": ingest_stats, "questions_scored": len(scored_questions), "by_method": by_method,
            "unanswerable": {"n": len(top_dense_score["unanswerable"]), "top_dense_score_auc_answerable_vs_unanswerable": round(auc, 3),
                             "median_top_score_answerable": round(sorted(top_dense_score["answerable"])[len(top_dense_score["answerable"]) // 2], 3),
                             "median_top_score_unanswerable": round(sorted(top_dense_score["unanswerable"])[len(top_dense_score["unanswerable"]) // 2], 3)},
        }
        print(f"[{strategy}] chunks={ingest_stats['chunks']} embedding_texts={ingest_stats['embedding_texts']} ingest={ingest_stats['ingest_seconds']}s "
              f"| " + " | ".join(f"{m}: hit@1={by_method[m]['overall']['hit@1']:.3f} hit@6={by_method[m]['overall']['hit@6']:.3f} mrr={by_method[m]['overall']['rr']:.3f}" for m in methods))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{results['meta']['commit']}{('-' + args.tag) if args.tag else ''}.json"
    (out_dir / name).write_text(json.dumps(results, indent=2), encoding="utf8")
    print("wrote", out_dir / name)
    await engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
