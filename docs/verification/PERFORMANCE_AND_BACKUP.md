# Performance, load tests and backup/restore — recorded runs

Date: 2026-10-23. One Windows 11 laptop (12 logical CPUs), PostgreSQL 18 on localhost, Python 3.13. Nothing here touches Pinecone, the real LLM
gateway or any deployed service. **Measured** results are labelled as such; anything else is an extrapolation and says so.

## 1. Ingestion size ladder (`scripts/perf_ingest.py`) — measured

Plain-text documents through the real pipeline (spool → parse → chunk → embed → rows + vectors → verify) with the **fake** embedding provider,
so parsing, chunking, memory and database cost are isolated. Raw: `evaluation/results/20261023-163149-perf-ingest.json`.

| Size | Outcome | Chunks | Seconds | Peak process RSS |
|---|---|---|---|---|
| 100 KB | indexed | 143 | 0.16 | 93 MB |
| 1 MB | indexed | 1,456 | 0.76 | 125 MB |
| 5 MB | **refused** `DOCUMENT_TOO_LARGE` (7,300 chunks > limit 5,000), non-retryable | – | 0.22 | 125 MB |
| 10 MB | refused (14,548 chunks) | – | 1.15 | 148 MB |
| 25 MB | refused (36,378 chunks) | – | 2.73 | 221 MB |

Real embedding throughput (BAAI/bge-small-en-v1.5, ONNX CPU, default settings: batch 4, threads unset), 1,000-character chunks, nothing else running:
**34.8 chunks/s** (256 chunks in 7.35 s; `20261023-215350-perf-ingest.json`).

> **Correction.** The first version of this document reported 3.7 chunks/s. That sample's timing window had included a ~31 s model download, so it was wrong by about 10x. The corrected figure above replaces it everywhere it was quoted (course material, question bank, quick revision).

Consequences, stated carefully: at the measured rate a 5,000-chunk document needs about 2.5 minutes of embedding **on this laptop** (extrapolation from the 256-chunk sample). The deployed free instance has 0.1 CPU and 512 MB; the repository's own notes estimate 1–2 s per chunk there ([Documented], not reproduced), i.e. hours for a maximal document. The chunk limit is therefore a safeguard sized for that hardware, and it makes plain text of 5 MB or more fail fast with a clear error rather than after a long wait.

## 2. Concurrent load tests (`scripts/load_test.py`) — measured

**Method.** The real four services run as local processes against two throw-away PostgreSQL databases (names must start with `loadtest_`), with the real local embedding model and the in-database vector store (no Pinecone). The LLM is the mock gateway with a fixed 800 ms delay per completion (Agent Core makes about 4 per turn). Rate limiting is off; the per-user stream cap stays on (3). Resources are sampled once a second (RSS and CPU of every service process, and the job queue read from the database). Each result file in `evaluation/results/*-load-*.json` records the full configuration and the machine.

**Concurrent ingestion** (all documents unique text; `WORKER_CONCURRENCY` / `INGEST_CONCURRENCY` as shown):

| Scenario | Submitted → ready | Wall | Docs/min | Upload→ready p50 / p95 | Errors | Peak RSS rag-core / backend | Queue peak (pending/running) |
|---|---|---|---|---|---|---|---|
| 3 users × 2 docs × 60 KB, concurrency 1/1 | 6 → 6 | 25 s | 14.3 | 13.1 s / 25.2 s | 0 | 345 MB / 171 MB | 5 / 1 |
| 4 users × 3 docs × 200 KB, concurrency 1/1 | 12 → 12 | 98.5 s | 7.3 | 57.5 s / 90.1 s | 0 | 359 MB / 172 MB | 11 / 1 |
| 4 users × 3 docs × 200 KB, concurrency 2/2 | 12 → 12 | 98.8 s | 7.3 | 65.9 s / 98.7 s | 0 | 418 MB / 171 MB | 10 / 2 |
| 2 users × 2 scanned PDFs × 3 pages, **OCR on** | 4 → 4 | 23.7 s | 10.1 | 17.4 s / 23.6 s | 0 | **526 MB** / 111 MB | 3 / 1 |

Reading it: ingestion is CPU-bound in RAG Core (about 7.7 cores busy at concurrency 1); raising the concurrency to 2 gave **no throughput gain** (7.3 docs/min either way) and about 60 MB more memory, so the default of 1 is the right setting on this hardware. Uploads themselves return in about 0.16 s; the rest is the queue draining. With OCR enabled RAG Core peaked at 526 MB, **above the 512 MB of the deployed free instance** (measured here with the engine and the embedding model both loaded), so OCR cannot be enabled there without more memory.

**Concurrent chat** (each user has one small indexed document; every turn streams a grounded answer):

| Scenario | Turns | Wall | Turns/min | Time to first token p50 / p95 | Turn total p50 / p95 | Errors |
|---|---|---|---|---|---|---|
| 8 users × 3 turns | 24 / 24 | 12.6 s | 114 | 2.51 s / 3.60 s | 3.82 s / 4.88 s | 0 |
| 24 users × 3 turns | 72 / 72 | 15.6 s | 278 | 2.69 s / 6.19 s | 3.99 s / 7.51 s | 0 |
| 6 users × 3 turns **while 12 documents (100 KB) ingest** | 18 / 18 | 13.3 s | 81 | 2.60 s / 3.95 s | 3.93 s / 5.29 s | 0 |

In the mixed run all 12 documents also became ready (51.9 s). At this scale ingestion did not visibly slow chat (first-token p50 2.60 s vs 2.51 s idle). Peak RSS in the 24-user run: backend 172 MB, agent-core 125 MB, rag-core 340 MB, llm 102 MB.

**A defect found by these runs.** In every ingestion run (1 of 6, 12, 12 and 4 uploads) one upload returned HTTP 500: the local-storage path check used `Path.resolve()`, which on Windows can return a `\\?\C:\...` path while another request is creating the directory, so a valid key was rejected. Fixed with a lexical check and regression tests (`backend/tests/unit/test_security_primitives.py`); the four ingestion scenarios were rerun and all documents succeeded, and the table above is from those reruns. Linux deployments were not affected by this specific race.

**What these runs do not show.** Only one machine and one run per configuration (no repeats, so no variance estimate); no Pinecone or Neon network latency; a mock model whose delay is fixed; local brute-force vector search over small collections; no test of the per-user stream cap under load (covered by unit tests); no soak test (the longest run was under two minutes); and none of this predicts the deployed free tier, whose CPU is a small fraction of this laptop's. `approx_chunks_per_second` in the JSON is derived from bytes ÷ 900 and is only indicative (meaningless for the OCR scenario). **No capacity or scalability claim follows from them.**

## 3. Backup / restore (local PostgreSQL) — measured

`pg_dump -Fc` of the perf database → `pg_restore` into a fresh database: identical row counts (2 documents, 1,599 chunks), identical MD5 over all chunk text ordered by id, the same two indexes on `rag_chunks`; dump size 526 KB. This proves the procedure on a local database; it was not run against the managed production database. Object-store contents (the `s3` backend) need their own bucket versioning/replication, which this repository does not configure or test; with the `db` storage backend file bytes are inside the database dump. Vectors are derived data: they can be rebuilt from stored chunk rows (`POST /api/v1/rag/documents/{id}/reindex`), and `GET .../audit` reports where rows and vectors disagree (`docs/verification/FEATURE_MATRIX.md`, items 19 and 30).
