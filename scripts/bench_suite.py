"""Performance benchmark suite against the real RAG Core pipeline (no Pinecone, no LLM, no production service).

    python scripts/bench_suite.py stages    --db postgresql+asyncpg://test@127.0.0.1:55432/ragcore_bench [--repeats 3]
    python scripts/bench_suite.py formats   --db ... [--quick]
    python scripts/bench_suite.py embedding --db ...
    python scripts/bench_suite.py lifecycle --db ...

Sub-commands
  stages     Real `pipeline.retrieve()` over the evaluation corpus: per-stage latency (query embedding, dense search, keyword
             search, RRF, rerank, context construction) with mean/min/p50/p90/p95/p99/max, cold vs warm, concurrency scaling,
             empty-result and failure rates, and answer-evidence-in-context (a retrieval-side proxy, NOT answer correctness).
  formats    Ingestion of every supported type (txt, pdf text/scanned/mixed, docx, csv, png, jpeg) and size, with real
             embeddings: parse / chunk / embed / upsert / total seconds, peak process memory, chunks, characters, and a
             retrieval check that a planted sentence is found after ingestion.
  embedding  Embedding throughput and latency by batch size with the real local model.
  lifecycle  Storage footprint, index/row consistency (audit), reindex, replacement and deletion timings.

Safety: needs a DISPOSABLE PostgreSQL database whose name contains "bench" (tables are dropped/recreated). Vector store is the
in-database brute-force store, embeddings are the real local ONNX model, the reranker is either none or a LOCAL cross-encoder
stand-in (NOT the production bge-reranker-v2-m3). Everything labelled "mock" or "stand-in" in the output is exactly that.
Results: evaluation/results/<stamp>-bench-<command>.json
"""
import argparse
import asyncio
import io
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "rag-core"), str(ROOT / "rag-core" / "tests"), str(ROOT / "shared")]

MIME = {".md": "text/markdown", ".txt": "text/plain", ".csv": "text/csv", ".pdf": "application/pdf", ".png": "image/png",
        ".jpg": "image/jpeg", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
SEED = 20261026


def commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def dirty() -> bool:
    try:
        return bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    except Exception:
        return True


def machine() -> dict:
    import onnxruntime
    import psutil

    info = {"platform": platform.platform(), "logical_cpus": os.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "python": platform.python_version(), "onnxruntime": onnxruntime.__version__, "cpu": platform.processor()}
    try:
        import fastembed

        info["fastembed"] = getattr(fastembed, "__version__", "?")
    except Exception:
        pass
    return info


def stats(values) -> dict:
    """mean/min/p50/p90/p95/p99/max (nearest-rank percentiles). p99 is only meaningful with >= 100 samples; the sample count is returned."""
    from rag_core.evaluation import percentile

    v = list(values)
    if not v:
        return {"n": 0}
    return {"n": len(v), "mean": round(sum(v) / len(v), 2), "min": round(min(v), 2), "p50": round(percentile(v, 50), 2),
            "p90": round(percentile(v, 90), 2), "p95": round(percentile(v, 95), 2), "p99": round(percentile(v, 99), 2),
            "max": round(max(v), 2), "p99_reliable": len(v) >= 100}


class MemorySampler:
    """Peak RSS of this process (MB) sampled every 20 ms during a block; also mean CPU utilisation of the process."""

    def __init__(self):
        import psutil

        self.p = psutil.Process()
        self.peak = 0.0
        self.cpu0 = None

    def __enter__(self):
        self.stop = False
        self.base = self.p.memory_info().rss / 1e6
        self.t0 = time.perf_counter()
        t = self.p.cpu_times()
        self.cpu0 = t.user + t.system
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        while not self.stop:
            self.peak = max(self.peak, self.p.memory_info().rss / 1e6)
            time.sleep(0.02)

    def __exit__(self, *a):
        self.stop = True
        self.thread.join()
        self.peak = max(self.peak, self.p.memory_info().rss / 1e6)
        t = self.p.cpu_times()
        wall = max(time.perf_counter() - self.t0, 1e-9)
        self.cpu_cores_busy = round((t.user + t.system - self.cpu0) / wall, 2)  # 1.0 = one core fully busy

    def report(self) -> dict:
        return {"rss_before_mb": round(self.base, 1), "peak_rss_mb": round(self.peak, 1), "cpu_cores_busy_avg": self.cpu_cores_busy}


def setup_env(db_url: str):
    os.environ.update({"ADVANCERAG_ENV_FILE": "none", "ENVIRONMENT": "test", "DATABASE_URL": db_url, "EMBEDDING_PROVIDER": "local",
                       "VECTOR_STORE": "local", "RERANKER": "none", "LOG_LEVEL": "WARNING", "INTERNAL_SERVICE_TOKEN": ""})
    os.environ.setdefault("EMBEDDING_CACHE_DIR", str(ROOT / "rag-core" / "var" / "models"))


async def boot(args, ocr: bool):
    from rag_core import db as rag_db
    from rag_core import pipeline
    from rag_core.config import settings
    from rag_core.embeddings import build_embeddings
    from rag_core.rerank import NoopReranker
    from rag_core.vectorstore import LocalVectorStore

    if "bench" not in args.db.rsplit("/", 1)[-1]:
        sys.exit("refusing: database name must contain 'bench' (this script drops tables)")
    settings.OCR_ENABLED = ocr
    engine = rag_db.configure_engine(args.db)
    async with engine.begin() as conn:
        await conn.run_sync(rag_db.Base.metadata.drop_all)
        await conn.run_sync(rag_db.Base.metadata.create_all)
        await conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_rag_chunks_text_fts ON rag_chunks USING GIN (to_tsvector('english', text))")
    t = time.perf_counter()
    emb = build_embeddings(settings)
    await emb.warm_up()
    warm = time.perf_counter() - t
    svc = pipeline.Services(settings, emb, LocalVectorStore(), NoopReranker())
    return rag_db, pipeline, settings, svc, engine, warm


def write_result(name: str, payload: dict):
    out = ROOT / "evaluation" / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{time.strftime('%Y%m%d-%H%M%S')}-bench-{name}.json"
    payload["meta"] = {**payload.get("meta", {}), "stamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "commit": commit(), "working_tree_dirty": dirty(),
                       "machine": machine(), "command": "python " + " ".join(sys.argv), "seed": SEED}
    path.write_text(json.dumps(payload, indent=2), encoding="utf8")
    print("wrote", path)


# ------------------------------------------------------------------ stages

class LocalCrossEncoderReranker:
    """STAND-IN reranker: a local cross-encoder (Xenova/ms-marco-MiniLM-L-6-v2). Not the production reranker."""

    name = "local-cross-encoder-standin"

    def __init__(self):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2", cache_dir=str(ROOT / "rag-core" / "var" / "models"))

    async def rerank(self, query, docs, top_n):
        import anyio

        scores = await anyio.to_thread.run_sync(lambda: list(self.model.rerank(query, [t for _, t in docs])))
        order = sorted(range(len(docs)), key=lambda i: -scores[i])[:top_n]
        return [(docs[i][0], float(scores[i])) for i in order]

    async def aclose(self):
        return None


async def ingest_corpus(rag_db, pipeline, svc, settings, user, corpus: Path):
    docs = {}
    for path in sorted(p for p in corpus.rglob("*") if p.is_file()):
        name = path.relative_to(corpus).as_posix()
        did = uuid.uuid4()
        async with rag_db.SessionLocal() as db:
            await pipeline.ingest(db, svc, document_id=did, user_id=user, filename=name, mime_type=MIME[path.suffix], path=path)
        docs[name] = did
    return docs


async def cmd_stages(args):
    setup_env(args.db)
    rag_db, pipeline, settings, svc, engine, warm = await boot(args, ocr=True)
    from rag_core import evaluation as ev
    from rag_core.db import Chunk, IndexedDocument
    from sqlalchemy import select

    user = uuid.uuid4()
    corpus = ROOT / "evaluation" / "dataset" / "corpus"
    t = time.perf_counter()
    docs = await ingest_corpus(rag_db, pipeline, svc, settings, user, corpus)
    ingest_s = time.perf_counter() - t
    doc_ids = list(docs.values())
    questions = ev.load_questions(ROOT / "evaluation" / "dataset" / "questions.jsonl")
    async with rag_db.SessionLocal() as db:
        n_chunks = len((await db.execute(select(Chunk.id).where(Chunk.user_id == user))).all())
        info = {r[0]: (r[1], r[2]) for r in (await db.execute(select(Chunk.id, IndexedDocument.filename, Chunk.text).join(
            IndexedDocument, IndexedDocument.document_id == Chunk.document_id).where(Chunk.user_id == user))).all()}

    # Timing wrappers around the real stages used by pipeline.retrieve() (no behavioural change).
    acc: dict = {}

    def wrap_async(obj, attr, key):
        orig = getattr(obj, attr)

        async def inner(*a, **k):
            t0 = time.perf_counter()
            try:
                return await orig(*a, **k)
            finally:
                acc.setdefault(key, []).append((time.perf_counter() - t0) * 1000)
        setattr(obj, attr, inner)

    wrap_async(svc.embeddings, "embed", "query_embedding_ms")
    wrap_async(svc.vectors, "query", "dense_search_ms")
    wrap_async(pipeline, "_keyword", "keyword_search_ms")
    orig_rrf = pipeline.rrf

    def timed_rrf(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_rrf(*a, **k)
        finally:
            acc.setdefault("rrf_ms", []).append((time.perf_counter() - t0) * 1000)
    pipeline.rrf = timed_rrf

    cross = LocalCrossEncoderReranker()
    results = {"meta": {"corpus": "evaluation/dataset/corpus (v1.2, synthetic)", "documents": len(docs), "chunks": n_chunks,
                        "questions": len(questions), "answerable": sum(1 for q in questions if q["answerable"]),
                        "embedding_model": svc.embeddings.model, "dimensions": svc.embeddings.dimensions,
                        "keyword_ranking": settings.KEYWORD_RANKING, "retrieval_top_k": settings.RETRIEVAL_TOP_K,
                        "retrieval_candidates": settings.RETRIEVAL_CANDIDATES, "rrf_k": settings.RRF_K,
                        "vector_store": "in-database brute-force cosine in Python (NOT Pinecone; search time scales with chunk count)",
                        "reranker_note": "'rerank_standin' rows use a local MiniLM cross-encoder, NOT bge-reranker-v2-m3 via Pinecone",
                        "embedding_model_load_s": round(warm, 2), "corpus_ingest_s": round(ingest_s, 1), "repeats": args.repeats}}

    for label, reranker in (("no_rerank", svc.reranker), ("rerank_standin", cross)):
        svc.reranker = reranker
        orig_rerank = reranker.rerank

        async def timed_rerank(q, d, n, _o=orig_rerank):
            t0 = time.perf_counter()
            try:
                return await _o(q, d, n)
            finally:
                acc.setdefault("rerank_ms", []).append((time.perf_counter() - t0) * 1000)
        reranker.rerank = timed_rerank
        # cold: first retrieval after the reranker is loaded for the first time in this process is measured separately
        acc.clear()
        total, per = [], {}
        fails, empties, rows_scored, failures_detail = 0, 0, [], []
        cold = None
        rng = random.Random(SEED)
        order = []
        for _ in range(args.repeats):
            qs = list(questions)
            rng.shuffle(qs)
            order.extend(qs)
        sampler = MemorySampler()
        with sampler:
            for i, q in enumerate(order):
                acc_before = {k: len(v) for k, v in acc.items()}
                t0 = time.perf_counter()
                try:
                    async with rag_db.SessionLocal() as db:
                        res = await pipeline.retrieve(db, svc, user_id=user, query=q["question"], document_ids=doc_ids, top_k=10)
                except Exception as exc:  # counted, never hidden
                    fails += 1
                    failures_detail.append(f"{q['id']}: {type(exc).__name__}")
                    continue
                ms = (time.perf_counter() - t0) * 1000
                if i == 0:
                    cold = round(ms, 1)
                total.append(ms)
                comp = {k: v[acc_before.get(k, 0):][-1] if len(v) > acc_before.get(k, 0) else 0.0 for k, v in acc.items()}
                comp["context_and_db_other_ms"] = max(ms - sum(comp.values()), 0.0)
                for k, v in comp.items():
                    per.setdefault(k, []).append(v)
                if not res["chunks"]:
                    empties += 1
                if i < len(questions):  # quality is computed once per question (first repeat); latency uses every repeat
                    pass
                ranked = [(c["filename"], c["text"]) for c in res["chunks"]]
                if q["answerable"] and i < len(order) // args.repeats:
                    rows_scored.append({"qid": q["id"], "type": q["type"], **ev.score_question(ranked, q["relevant"], ks=(1, 3, 5, 6, 10)),
                                        "context_has_evidence": float(any(ev.is_relevant(f, t, q["relevant"]) for f, t in ranked))})
        agg = ev.aggregate([{k: v for k, v in r.items() if isinstance(v, float)} for r in rows_scored]) if rows_scored else {}
        n = len(rows_scored)
        results[label] = {
            "queries_issued": len(order), "completed": len(total), "failures": fails, "failure_rate": round(fails / len(order), 4),
            "failure_detail_sample": failures_detail[:5], "empty_result_rate": round(empties / max(len(total), 1), 4),
            "cold_first_query_ms": cold, "latency_ms_total_retrieve": stats(total),
            "stage_latency_ms": {k: stats(v) for k, v in sorted(per.items())},
            "quality_answerable_n": n,
            "quality": {k: round(v, 4) for k, v in agg.items() if k.split("@")[0] in ("hit", "precision", "recall", "ndcg") or k in ("rr", "context_has_evidence")},
            "hit@5_wilson95": [round(x, 3) for x in ev.wilson(sum(r["hit@5"] for r in rows_scored), n)] if n else None,
            "mrr_bootstrap95": [round(x, 3) for x in ev.bootstrap_ci([r["rr"] for r in rows_scored])] if n else None,
            "resources": sampler.report(),
        }
        print(label, json.dumps({k: results[label][k] for k in ("failures", "empty_result_rate", "cold_first_query_ms", "latency_ms_total_retrieve", "quality")}, indent=1))

    # Concurrency scaling of the full retrieve() path (no rerank vs stand-in rerank)
    results["concurrency"] = {}
    for label, reranker in (("no_rerank", None), ("rerank_standin", cross)):
        if reranker is None:
            from rag_core.rerank import NoopReranker

            svc.reranker = NoopReranker()
        else:
            svc.reranker = reranker
        for conc in (1, 4, 8, 16):
            qs = [q["question"] for q in random.Random(SEED + conc).sample(questions, min(len(questions), 160))]
            lat, errs = [], 0
            sem = asyncio.Semaphore(conc)

            async def one(text):
                nonlocal errs
                async with sem:
                    t0 = time.perf_counter()
                    try:
                        async with rag_db.SessionLocal() as db:
                            await pipeline.retrieve(db, svc, user_id=user, query=text, document_ids=doc_ids, top_k=6)
                        lat.append((time.perf_counter() - t0) * 1000)
                    except Exception:
                        errs += 1
            sampler = MemorySampler()
            with sampler:
                w0 = time.perf_counter()
                await asyncio.gather(*(one(t) for t in qs))
                wall = time.perf_counter() - w0
            results["concurrency"][f"{label}_c{conc}"] = {"queries": len(qs), "errors": errs, "wall_s": round(wall, 2),
                                                          "throughput_qps": round(len(lat) / wall, 2), "latency_ms": stats(lat), "resources": sampler.report()}
            print(label, conc, results["concurrency"][f"{label}_c{conc}"]["throughput_qps"], "qps; p50", results["concurrency"][f"{label}_c{conc}"]["latency_ms"].get("p50"))
    write_result("stages", results)
    await engine.dispose()


# ------------------------------------------------------------------ formats

def make_cases(tmp: Path, quick: bool):
    """Synthetic but realistic-shaped documents. Each carries a planted sentence used for the post-ingestion retrieval check."""
    from fixtures import docx_bytes, mixed_pdf, scanned_pdf, text_image, text_pdf, image_bytes

    rng = random.Random(SEED)
    words = ("refund shipping invoice warranty customer policy delivery account balance contract terms support returns payment order "
             "product service quality standard region office report annual budget review supplier schedule contract renewal audit").split()

    def prose(size: int) -> str:
        out, n = [], 0
        while n < size:
            para = " ".join(rng.choice(words) for _ in range(rng.randrange(40, 110))).capitalize() + ".\n\n"
            out.append(para)
            n += len(para)
        return "".join(out)[:size]

    def planted(tag: str) -> str:
        return f"The verification code for document {tag} is QZV-{tag}-7731."

    def with_marker(text: str, tag: str) -> str:
        mid = len(text) // 2
        return text[:mid] + "\n\n" + planted(tag) + "\n\n" + text[mid:]

    cases = []

    def add(name, kind, filename, data: bytes, tag, ocr=False, repeats=1):
        p = tmp / filename
        p.write_bytes(data)
        cases.append({"name": name, "kind": kind, "path": p, "bytes": len(data), "tag": tag, "ocr": ocr, "repeats": repeats})

    add("txt_100KB", "text", "t100k.txt", with_marker(prose(100_000), "T100K").encode(), "T100K", repeats=3)
    add("txt_1MB", "text", "t1m.txt", with_marker(prose(1_000_000), "T1M").encode(), "T1M")
    if not quick:
        add("txt_3.2MB_near_chunk_limit", "text", "t32m.txt", with_marker(prose(3_200_000), "T32M").encode(), "T32M")
    for mb in (5, 10, 25):
        add(f"txt_{mb}MB", "text", f"t{mb}m.txt", with_marker(prose(mb * 1_000_000), f"T{mb}M").encode(), f"T{mb}M")

    pages10 = [with_marker(prose(2200), "P10") if i == 4 else prose(2200) for i in range(10)]

    def wrap_lines(page: str):
        import textwrap

        return "\n".join(textwrap.wrap(page.replace("\n\n", " "), 95)[:48])
    add("pdf_text_10p", "pdf_text", "p10.pdf", text_pdf([wrap_lines(p) for p in pages10]), "P10", repeats=3)
    pages100 = [wrap_lines(with_marker(prose(2400), "P100") if i == 50 else prose(2400)) for i in range(100)]
    add("pdf_text_100p", "pdf_text", "p100.pdf", text_pdf(pages100), "P100")

    def image_page(tag: str, extra: int):
        lines = [f"Scanned page about {rng.choice(words)} and {rng.choice(words)}", planted(tag)] + [
            " ".join(rng.choice(words) for _ in range(9)) for _ in range(extra)]
        return text_image(lines, size=30, width=1500)
    add("pdf_scanned_3p_ocr", "pdf_scanned", "s3.pdf", scanned_pdf([image_page("SCAN3", 8) if i == 1 else image_page("SCANX", 8) for i in range(3)]), "SCAN3", ocr=True)
    if not quick:
        add("pdf_scanned_10p_ocr", "pdf_scanned", "s10.pdf", scanned_pdf([image_page("SCAN10", 8) if i == 5 else image_page("SCANY", 8) for i in range(10)]), "SCAN10", ocr=True)
    mixed = []
    for i in range(6):
        mixed.append(wrap_lines(prose(1800)) if i % 2 == 0 else image_page("MIX6" if i == 3 else "MIXX", 8))
    add("pdf_mixed_6p_ocr", "pdf_mixed", "m6.pdf", mixed_pdf(mixed), "MIX6", ocr=True)

    def docx_of(size: int, tag: str):
        text = with_marker(prose(size), tag)

        def build(d):
            d.add_heading("Quarterly operations report", 1)
            for para in [x for x in text.split("\n\n") if x.strip()]:
                d.add_paragraph(para)
        return docx_bytes(build)
    add("docx_100KB", "docx", "d100k.docx", docx_of(100_000, "D100K"), "D100K", repeats=3)
    add("docx_1MB", "docx", "d1m.docx", docx_of(1_000_000, "D1M"), "D1M")

    def csv_of(size: int, tag: str):
        rows, n, i = ["sku,product,region,units,price_eur,note"], 0, 0
        while n < size:
            line = f"SKU-{i:06d},{rng.choice(words)} {rng.choice(words)},{rng.choice(['EU','US','APAC'])},{rng.randrange(1000)},{rng.randrange(1, 9999)/100:.2f},{rng.choice(words)} {rng.choice(words)}"
            if i == size // 120:
                line = f"SKU-{tag},verification {tag},EU,1,1.00,The verification code for document {tag} is QZV-{tag}-7731."
            rows.append(line)
            n += len(line) + 1
            i += 1
        return "\n".join(rows).encode()
    add("csv_100KB", "csv", "c100k.csv", csv_of(100_000, "C100K"), "C100K", repeats=3)
    add("csv_1MB", "csv", "c1m.csv", csv_of(1_000_000, "C1M"), "C1M")

    img = text_image([f"Photographed notice {rng.choice(words)}", planted("IMG1"), "Opening hours are nine to five on weekdays.", "Contact the office for the annual report."], size=34, width=1400)
    add("png_ocr", "image", "i1.png", image_bytes(img, "PNG"), "IMG1", ocr=True, repeats=3)
    add("jpeg_ocr", "image", "i1.jpg", image_bytes(img.convert("RGB"), "JPEG"), "IMG1", ocr=True, repeats=3)
    return cases


async def cmd_formats(args):
    setup_env(args.db)
    rag_db, pipeline, settings, svc, engine, warm = await boot(args, ocr=True)
    from rag_core import errors, ocr as ocr_mod
    from rag_core.chunking import chunk_pages
    from rag_core.parsing import parse_document
    import anyio

    tmp = Path(tempfile.mkdtemp(prefix="bench_formats_"))
    cases = make_cases(tmp, args.quick)
    rows = []
    # OCR engine load is a one-off cold start; do it outside the case timings and report it once.
    cold_ocr = None
    try:
        from fixtures import image_bytes, text_image

        t0 = time.perf_counter()
        parse_document(_write(tmp / "warm.png", image_bytes(text_image(["warm up the engine"], size=30, width=800), "PNG")), "image/png", settings)
        cold_ocr = round(time.perf_counter() - t0, 2)
    except Exception as exc:
        cold_ocr = f"failed: {type(exc).__name__}"

    for case in cases:
        for rep in range(case["repeats"]):
            user = uuid.uuid4()
            did = uuid.uuid4()
            mime = MIME[case["path"].suffix]
            row = {"case": case["name"], "kind": case["kind"], "run": rep + 1, "file_bytes": case["bytes"], "mime": mime, "ocr_involved": case["ocr"]}
            sampler = MemorySampler()
            with sampler:
                t_total = time.perf_counter()
                try:
                    # Independent stage timings first (extraction and chunking separately), then the real end-to-end ingest.
                    t0 = time.perf_counter()
                    parsed = await anyio.to_thread.run_sync(parse_document, case["path"], mime, settings)
                    row["extract_s"] = round(time.perf_counter() - t0, 3)
                    row["pages"] = len(parsed.pages)
                    row["extracted_chars"] = sum(len(p.text) for p in parsed.pages)
                    row["ocr_pages"] = getattr(parsed.report, "ocr_pages", 0)
                    row["extract_ocr_s_if_ocr"] = row["extract_s"] if row["ocr_pages"] else 0.0
                    t0 = time.perf_counter()
                    chunks = chunk_pages(parsed.pages, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
                    row["chunk_s"] = round(time.perf_counter() - t0, 3)
                    row["chunks_planned"] = len(chunks)
                    t0 = time.perf_counter()
                    async with rag_db.SessionLocal() as db:
                        out = await pipeline.ingest(db, svc, document_id=did, user_id=user, filename=case["path"].name, mime_type=mime, path=case["path"])
                    row["ingest_total_s"] = round(time.perf_counter() - t0, 2)  # includes a second parse+chunk (the pipeline's own)
                    row["pipeline_timings"] = out["timings"]  # parse_s (= parse+chunk), embed_s, upsert_s (verify excluded)
                    row["chunks"] = out["chunks"]
                    row["status"] = "indexed"
                    t0 = time.perf_counter()
                    async with rag_db.SessionLocal() as db:
                        res = await pipeline.retrieve(db, svc, user_id=user, query=f"What is the verification code for document {case['tag']}?", document_ids=[did], top_k=6)
                    row["verify_retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    marker = f"QZV-{case['tag']}-7731"
                    ranks = [i for i, c in enumerate(res["chunks"], 1) if marker in c["text"] or (case["ocr"] and case["tag"] in c["text"].replace(" ", ""))]
                    row["retrieval_verified"] = bool(ranks)
                    row["planted_sentence_rank"] = ranks[0] if ranks else None
                except errors.ServiceError as exc:
                    row.update(status="refused", error_code=exc.code, error=str(exc)[:160])
                except Exception as exc:
                    row.update(status="failed", error_code=type(exc).__name__, error=str(exc)[:160])
                row["wall_s"] = round(time.perf_counter() - t_total, 2)
            row.update(sampler.report())
            if row.get("chunks") and row.get("pipeline_timings", {}).get("embed_s"):
                row["embed_chunks_per_s"] = round(row["chunks"] / row["pipeline_timings"]["embed_s"], 1)
            rows.append(row)
            print(json.dumps({k: row.get(k) for k in ("case", "run", "status", "file_bytes", "pages", "extracted_chars", "chunks", "extract_s", "chunk_s", "ingest_total_s", "peak_rss_mb", "retrieval_verified", "error_code")}))
    write_result("formats", {"meta": {"ocr_engine_cold_load_and_first_image_s": cold_ocr, "limits": {
        "backend MAX_UPLOAD_BYTES": 25 * 1024 * 1024, "rag-core MAX_FILE_BYTES": settings.MAX_FILE_BYTES,
        "MAX_CHUNKS_PER_DOCUMENT": settings.MAX_CHUNKS_PER_DOCUMENT, "CHUNK_SIZE": settings.CHUNK_SIZE, "CHUNK_OVERLAP": settings.CHUNK_OVERLAP,
        "OCR_RENDER_DPI": getattr(settings, "OCR_RENDER_DPI", None)}, "embedding_model": svc.embeddings.model,
        "note": "synthetic documents generated by this script; ocr rows use the real RapidOCR engine; vector store is in-database brute force"}, "rows": rows})
    await engine.dispose()


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


# ------------------------------------------------------------------ embedding

async def cmd_embedding(args):
    setup_env(args.db)
    rag_db, pipeline, settings, svc, engine, warm = await boot(args, ocr=False)
    from rag_core.embeddings import DOCUMENT, QUERY, LocalEmbeddings

    rng = random.Random(SEED)
    words = "refund shipping invoice warranty customer policy delivery account balance contract terms support returns payment order product".split()
    texts = [" ".join(rng.choice(words) for _ in range(150))[:1000] for _ in range(256)]
    res = {"meta": {"model": svc.embeddings.model, "dimensions": svc.embeddings.dimensions, "chunk_chars": 1000, "chunks": len(texts),
                    "model_load_s": round(warm, 2), "default_batch_size": settings.EMBEDDING_BATCH_SIZE, "threads": settings.EMBEDDING_THREADS}, "batch_sizes": {}}
    first = await svc.embeddings.embed(texts[:1], QUERY)
    norm = math.sqrt(sum(x * x for x in first[0]))
    res["meta"]["query_vector_norm"] = round(norm, 4)
    q_lat = []
    for i in range(100):
        t = time.perf_counter()
        await svc.embeddings.embed([f"What is the policy about {rng.choice(words)} and {rng.choice(words)}?"], QUERY)
        q_lat.append((time.perf_counter() - t) * 1000)
    res["single_query_embedding_ms"] = stats(q_lat)
    for bs in (1, 4, 8, 16, 32, 64):
        prov = LocalEmbeddings(model=settings.EMBEDDING_MODEL, cache_dir=os.environ["EMBEDDING_CACHE_DIR"], batch_size=bs, threads=settings.EMBEDDING_THREADS)
        await prov.warm_up()
        await prov.embed(texts[:8], DOCUMENT)  # warm
        runs = []
        for _ in range(3):
            sampler = MemorySampler()
            with sampler:
                t = time.perf_counter()
                await prov.embed(texts, DOCUMENT)
                runs.append(time.perf_counter() - t)
        best, med = min(runs), sorted(runs)[1]
        res["batch_sizes"][str(bs)] = {"runs_s": [round(r, 2) for r in runs], "chunks_per_s_median": round(len(texts) / med, 1),
                                       "chunks_per_s_best": round(len(texts) / best, 1), "model_batches_per_call": math.ceil(len(texts) / bs),
                                       "ms_per_model_batch_median": round(med / math.ceil(len(texts) / bs) * 1000, 1), "resources_last_run": sampler.report()}
        print(bs, res["batch_sizes"][str(bs)]["chunks_per_s_median"], "chunks/s")
    write_result("embedding", res)
    await engine.dispose()


# ------------------------------------------------------------------ lifecycle

async def cmd_lifecycle(args):
    setup_env(args.db)
    rag_db, pipeline, settings, svc, engine, warm = await boot(args, ocr=False)
    from rag_core.db import Chunk
    from sqlalchemy import func, select, text

    rng = random.Random(SEED)
    words = "refund shipping invoice warranty customer policy delivery account balance contract terms support returns payment order product".split()

    def doc(n_chars, salt):
        out, n = [], 0
        while n < n_chars:
            p = " ".join(rng.choice(words) for _ in range(rng.randrange(40, 110))) + f" {salt}.\n\n"
            out.append(p)
            n += len(p)
        return "".join(out)[:n_chars].encode()

    user, did = uuid.uuid4(), uuid.uuid4()
    tmp = Path(tempfile.mkdtemp(prefix="bench_life_"))
    res = {"meta": {"embedding_model": svc.embeddings.model, "dimensions": svc.embeddings.dimensions}}
    p1 = tmp / "v1.txt"
    p1.write_bytes(doc(300_000, "alpha"))
    t = time.perf_counter()
    async with rag_db.SessionLocal() as db:
        out = await pipeline.ingest(db, svc, document_id=did, user_id=user, filename="v1.txt", mime_type="text/plain", path=p1)
    res["initial_ingest"] = {"seconds": round(time.perf_counter() - t, 2), "chunks": out["chunks"], "timings": out["timings"], "source_bytes": p1.stat().st_size}
    async with rag_db.SessionLocal() as db:
        total_chunks = await db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == did))
        with_vec = await db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == did, Chunk.embedding.is_not(None)))
        sizes = {}
        for tbl in ("rag_chunks", "rag_documents", "rag_ingestions"):
            sizes[tbl] = (await db.execute(text(f"select pg_total_relation_size('{tbl}')"))).scalar()
        text_bytes = (await db.execute(text("select coalesce(sum(octet_length(text)),0) from rag_chunks"))).scalar()
        emb_bytes = (await db.execute(text("select coalesce(sum(pg_column_size(embedding)),0) from rag_chunks"))).scalar()
    res["storage"] = {"chunk_rows": total_chunks, "chunk_rows_with_vector": with_vec, "vectors_in_store": with_vec, "dimensions": svc.embeddings.dimensions,
                      "raw_vector_bytes_float32_estimate": with_vec * svc.embeddings.dimensions * 4, "db_relation_bytes_total": sizes,
                      "chunk_text_bytes": text_bytes, "embedding_column_bytes_as_stored_in_pg(JSON)": emb_bytes,
                      "note": "local store keeps vectors as a JSON column in rag_chunks; Pinecone storage/billing is NOT measured and may differ"}
    t = time.perf_counter()
    async with rag_db.SessionLocal() as db:
        res["audit_after_ingest"] = await pipeline.audit_document(db, svc, document_id=did, user_id=user)
    res["audit_seconds"] = round(time.perf_counter() - t, 3)
    # reindex from stored rows
    t = time.perf_counter()
    async with rag_db.SessionLocal() as db:
        res["reindex"] = {"result": await pipeline.reindex_document(db, svc, document_id=did, user_id=user), "seconds": round(time.perf_counter() - t, 2)}
    # idempotent re-ingest of identical content (what a retry does)
    t = time.perf_counter()
    async with rag_db.SessionLocal() as db:
        await pipeline.ingest(db, svc, document_id=did, user_id=user, filename="v1.txt", mime_type="text/plain", path=p1)
    res["reingest_identical_seconds"] = round(time.perf_counter() - t, 2)
    # replacement by a shorter version (stale tail vectors must go)
    p2 = tmp / "v2.txt"
    p2.write_bytes(doc(120_000, "beta"))
    t = time.perf_counter()
    async with rag_db.SessionLocal() as db:
        out2 = await pipeline.ingest(db, svc, document_id=did, user_id=user, filename="v2.txt", mime_type="text/plain", path=p2)
    res["replacement_by_shorter_version"] = {"seconds": round(time.perf_counter() - t, 2), "old_chunks": out["chunks"], "new_chunks": out2["chunks"]}
    async with rag_db.SessionLocal() as db:
        rows_after = await db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == did))
        res["replacement_by_shorter_version"]["rows_after"] = rows_after
        res["audit_after_replacement"] = await pipeline.audit_document(db, svc, document_id=did, user_id=user)
    t = time.perf_counter()
    async with rag_db.SessionLocal() as db:
        deleted = await pipeline.delete_document(db, svc, document_id=did, user_id=user)
    res["deletion"] = {"seconds": round(time.perf_counter() - t, 3), "returned": deleted}
    async with rag_db.SessionLocal() as db:
        res["deletion"]["rows_remaining"] = await db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == did))
    print(json.dumps({k: v for k, v in res.items() if k != "storage"}, indent=1, default=str)[:2500])
    write_result("lifecycle", res)
    await engine.dispose()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["stages", "formats", "embedding", "lifecycle"])
    ap.add_argument("--db", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    asyncio.run({"stages": cmd_stages, "formats": cmd_formats, "embedding": cmd_embedding, "lifecycle": cmd_lifecycle}[args.command](args))


if __name__ == "__main__":
    main()
