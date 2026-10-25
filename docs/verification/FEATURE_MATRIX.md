# Feature-to-test traceability matrix

Maintained during the implementation pass that started at commit `31bad7e`. Statuses: **IMPLEMENTED_AND_TESTED** (integration/e2e evidence), **IMPLEMENTED_UNIT_TESTED**, **IMPLEMENTED_NOT_INTEGRATION_TESTED**, **PARTIALLY_IMPLEMENTED**, **BLOCKED**, **NOT_IMPLEMENTED**. Nothing is production-verified: no production deployment, real Pinecone index, or real LLM gateway was touched.

## Baseline (before any code change)

Environment: Windows 11, Python 3.13, repo at `31bad7e` (working tree: only the generated course files were untracked).

| Suite | SQLite (default) | Isolated local PostgreSQL 18 (throwaway cluster, port 55432) |
|---|---|---|
| rag-core | 87 passed, 4 skipped | 91 passed, 0 skipped |
| agent-core | 199 passed | – (no database) |
| llm | 92 passed | – |
| backend | 218 passed, 4 skipped | 217 passed, 5 skipped (Redis/S3/migration-on-SQLite gates) |
| tests/contract | 52 passed | not run |
| frontend `npm run lint` (oxlint) | clean | – |
| frontend `npm run build` (tsc + vite) | succeeds | – |
| ruff / mypy | not installed in the project; not configured | – |

Not run in the baseline: Playwright e2e, `scripts/rag_benchmark.py` (needs live Pinecone + gateway), any load test.

## Current test results (end of the pass, 2026-10-23, local machine, working tree on top of `31bad7e`, nothing committed)

| Suite | SQLite (default) | Isolated local PostgreSQL 18 (port 55432) |
|---|---|---|
| rag-core | 196 passed, 5 skipped (all five need PostgreSQL concurrency/FTS) | 201 passed, 0 skipped |
| agent-core | 206 passed | – |
| llm | 92 passed | – |
| backend | 264 passed, 5 skipped (Redis x3, S3 gate, migration-on-SQLite) | 264 passed, 5 skipped |
| tests/contract | 52 passed | – |
| frontend `npm run lint` / `npm run build` | clean / succeeds | – |
| Playwright e2e (isolated stack, mock gateway) | 23 passed (desktop + mobile) | – |

Notes: one backend SQLite run in this pass reported `1 failed, 263 passed`; the failing test was not captured and seven
later full runs (three of them concurrent) passed, so it is an **unreproduced intermittent failure of unknown origin**, not a fixed bug.
`ruff`/`mypy` are not installed or configured in the project; no static type check was run on the Python code.
The 5 backend skips need Redis (`TEST_REDIS_URL`) and an S3 endpoint (`TEST_S3_*`); S3 behaviour is covered with `moto` instead.

## Matrix

Legend: PASS = implemented and covered by tests that ran here; PARTIAL = works but a stated part is untested or limited;
BLOCKED = needs something not available here. Every PASS is *hermetic* evidence (mock gateway, local vector store, scripted or
local components) unless stated; none is production evidence.

| # | Feature | Status | Evidence (what ran) | Limits |
|---|---|---|---|---|
| 1 | API routing / auth parity (Vercel rewrite, cookies, refresh) | PASS | `backend/tests/integration/test_frontend_contract.py` (every `api.ts` call matches a backend route+method), `test_auth.py`, `test_refresh_rotation.py`, Playwright | Live Vercel/Render routing not exercised |
| 2 | Logout-on-cold-start (refresh) bug | PASS | frontend `api.ts` fix; Playwright | – |
| 3 | Chat stream readiness / overload | PASS | `test_stream_limits.py` (per-user cap, slot release on failure/disconnect, leak expiry), `test_reliability.py` (partial output, whitelisted codes, agent down, disconnect, crash recovery) | Cap is per process; with several backend instances the cap is per instance |
| 4 | "Uploaded file could not be found" | PASS (hermetic) / PARTIAL (deployment) | `test_durability_versions.py`: durable `db`/`s3` backends, jobs reference objects not bytes, `FILE_MISSING` non-retryable with diagnostics, `STORAGE_UNAVAILABLE` retryable, legacy payload compatibility | The deployed service still needs `STORAGE_BACKEND=s3` or `db`; not changed in `render.yaml`, not deployed |
| 5 | OCR scanned / mixed PDFs | PASS | `rag-core/tests/test_ocr_extraction.py` with real RapidOCR (ONNX): scanned, mixed, degraded, timeouts, per-page budget, page numbers kept | Off by default (`OCR_ENABLED`); accuracy not benchmarked (no CER measurement); Latin-script text only tested |
| 6 | Image uploads (PNG/JPEG) | PASS | OCR tests; `test_hostile_inputs.py`; backend content sniffing `test_security_primitives.py` | – |
| 7 | Partial-extraction warnings | PASS | `ExtractionReport` in status API (`test_ocr_extraction.py`; the API-level test runs on PostgreSQL only) | – |
| 8 | Text normalisation (de-hyphenation, margins, whitespace) | PASS | `test_textclean.py` | Heuristics; English-centric |
| 9 | Heading-aware chunking | PASS | `test_chunking_strategies.py`; retrieval run `RETRIEVAL_EVALUATION.md` | Not the default; no clear quality win on the synthetic set |
| 10 | Semantic chunking | PASS (function) | `test_chunking_strategies.py` | No quality win measured; more embedding calls; not the default |
| 11 | Table extraction (CSV, DOCX) | PASS | `test_tables_extraction.py` (self-describing rows, header repeats, merged cells, exact identifier lookup end to end) | **PDF tables are not specially extracted or tested**; they are read as text |
| 12 | Durable object storage (`local`/`s3`/`db`) | PASS (hermetic) | `test_durability_versions.py` incl. `moto` S3; `test_infrastructure.py` | Real S3/R2 bucket not exercised |
| 13 | Bounded memory upload path | PASS | `test_integrity_and_spool.py` (temp-file spool); `PERFORMANCE_AND_BACKUP.md` peak RSS | Measured on text only, one machine |
| 14 | Durable + resumable jobs | PASS | `test_ingestion_jobs.py`, `test_reliability.py` (stale-running requeue, bounded retries, delete redrive) | Single-worker timings only |
| 15 | Chunk / document limits | PASS | `perf_ingest.py` run: 5 MB+ of plain text refused with `DOCUMENT_TOO_LARGE` (limit 5000 chunks) | **A 25 MB upload is accepted by the backend but text over ~5 MB is refused by RAG Core** – a configured safeguard, not a bug; real-embedding speed is the reason to keep it |
| 16 | Deduplication | PASS | `test_durability_versions.py` (incl. concurrent identical uploads under a PG advisory lock, PG only) | Per-user, content-hash based |
| 17 | Versioning / replace / restore / purge | PASS | `test_durability_versions.py`, Playwright | `replaces_id` has no DB foreign key (documented) |
| 18 | Complete deletion (blob + rows + vectors) | PASS | `test_durability_versions.py`, `test_reliability.py`, rag-core lifecycle tests | Pinecone deletion tested against the Pinecone-emulating server, not real Pinecone |
| 19 | Reindex from stored chunks | PASS | `test_integrity_and_spool.py` (3 tests) | New this pass; same-model repair only; not wired into the backend UI |
| 20 | Embedding model / dimension / metric integrity | PASS | `test_integrity_and_spool.py`, `vectorstore` metric check | Real index metadata not queried here |
| 21 | Retrieval evaluation (dataset, metrics, CIs, comparison) | PASS (tooling) | `test_evaluation.py`; `RETRIEVAL_EVALUATION.md` | Synthetic corpus, one run; no production-quality claim |
| 22 | Citations and abstention | PASS | `agent-core/tests/test_evidence_checks.py`, orchestrator tests; `scripts/rag_benchmark.py` refusal check fixed | `rag_benchmark.py` needs live services and was **not run** |
| 23 | Security: hostile inputs | PASS | `test_hostile_inputs.py` (zip bomb, ratio, fake DOCX, pixel flood, garbage images) | – |
| 24 | Security: dependency audit | PASS after fix | `pip-audit --local`: Pillow, pypdf, certifi advisories found, upgraded, re-audit clean (`SECURITY_AUDIT.md`) | Audit of the local venv, not of the production images; requirements are unpinned (`>=`) |
| 25 | Observability (trace ids, queue gauges, job metrics) | PASS | `test_observability.py` | No dashboards/alerts configured |
| 26 | Backup / restore | PASS (local) | `pg_dump`/`pg_restore` round trip, identical counts and text checksum (`PERFORMANCE_AND_BACKUP.md`) | Not tested on the managed production database; blob store backup not covered |
| 27 | Performance ladder | PARTIAL | `scripts/perf_ingest.py` (100 KB–25 MB, fake embeddings) + real-embedding throughput sample | No concurrent-user load test, no PDF/OCR ladder, no Pinecone latency |
| 28 | LLM service readiness | PASS (existing) | `/ready` checks configured/reachable/authenticated/model-available; `llm/tests` | A deep (chat-completion) readiness probe exists only as the CLI `healthcheck --chat`, deliberately not on the HTTP probe |
| 29 | Real-service verification (Pinecone, FreeLLMAPI, deployed stack) | BLOCKED | – | Not touched by design (no credentials used, no deploys) |
