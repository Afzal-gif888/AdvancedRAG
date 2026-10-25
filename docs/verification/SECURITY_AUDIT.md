# Security audit notes (two passes, 2026-10-23)

This is a code-and-test review plus dependency audits, **not a penetration test** (none was performed, by me or anyone else). Nothing was run against
a deployed system, no production data or secrets were touched, and `.env` files were deliberately not opened.

## Verified by tests that ran

| Area | Evidence |
|---|---|
| Hostile uploads: zip bomb (size), high compression ratio, fake DOCX, pixel-flood image, truncated/garbage images | `rag-core/tests/test_hostile_inputs.py` — typed, non-retryable refusals |
| Polyglot files: a PDF with a ZIP appended is read as a PDF only (the hidden member is never read); declared as DOCX it is refused; an HTML/script file declared `.txt` is plain text | `test_hostile_inputs.py::test_a_pdf_zip_polyglot_...`, `::test_a_text_file_with_html_...`; type is decided by content in `backend/tests/unit/test_security_primitives.py::test_content_sniffing` |
| Parser resource exhaustion: page cap before any page is read; **new** wall-clock budget for native PDF text extraction (`PDF_PARSE_TIMEOUT_SECONDS`, pages after it are reported unread); table-extraction budget (`PDF_TABLE_TIMEOUT_SECONDS`); OCR page/pixel/time budgets | `test_hostile_inputs.py::test_the_pdf_parse_budget_...`, `test_pdf_tables.py::test_table_time_budget_...`, `test_ocr_extraction.py` |
| Storage path confinement: `..`, absolute, Windows-drive, empty, sibling-prefix (`storage-evil`) and "root itself" keys are rejected; ordinary keys accepted; 40 concurrent first uploads all succeed | `backend/tests/unit/test_security_primitives.py` (**fixed a real bug**, below) |
| Prompt injection — structural defence: 11 delimiter look-alikes (case, whitespace, newline-split, forged `<source>` blocks) and attribute injection through filename/section cannot create or close a real `<source>`/`<evidence>` block; document text is kept verbatim as data | `agent-core/tests/test_evidence_checks.py`, `test_units.py` |
| Cross-user isolation: retrieval, tables, ingestion status, dedup, replace/restore/delete, reindex, audit and the new keyword-ranking queries are scoped to the owner | `rag-core/tests/test_rag_core.py`, `test_tables.py`, `test_ingestion_lifecycle.py`, `test_index_audit.py`, `test_keyword_ranking.py`; `backend/tests/security/`; `tests/contract/` |
| Per-user concurrent-stream cap, **now shared across replicas** when `RATE_LIMIT_BACKEND=redis` (atomic Lua script on the Redis clock; falls back to the per-process cap, with a log line, if Redis is unreachable) | `backend/tests/integration/test_stream_limits.py` (cap shared by two replicas, atomic under 20 simultaneous attempts, expiry — run on an in-process Redis with the real Lua engine) |
| Downloads are attachments with `nosniff`; secrets never reach logs; hostile/forged `traceparent` is replaced; the rate limiter fails closed outside development; internal token enforced | `test_documents.py`, `test_reliability.py`, `test_observability.py`, `llm/tests/test_api_and_config.py` |
| Production configuration is accepted by the services' own fail-fast rules; a typo'd key or a wildcard CORS origin is rejected | `scripts/verify_production_config.py`, `tests/contract/test_production_config.py` |

## Dependency management

- **Audit:** `pip-audit --local` on the development virtualenv found advisories in Pillow 12.3.0, pypdf 6.19.0 and certifi; all were upgraded
  (Pillow 13.0.0, pypdf 6.21.x, certifi 2026.10.x) and the re-audit reported no known vulnerabilities. I could not independently verify the advisory text; the tool's output is the evidence.
  Adding `pdfplumber` (and its `pdfminer.six`, `cryptography`) was audited the same way (clean).
- **Conflict found and resolved:** every `fastembed` release (0.8.1 and 0.9.0 checked) declares `pillow<13`, which contradicts the Pillow floor the audit requires. `pip check` flagged it locally, and a fresh install of `requirements.txt` would have failed to resolve. Resolution: `fastembed` is installed with `--no-deps` from `rag-core/requirements-fastembed.txt`, and its other dependencies are listed in `requirements.txt`. fastembed uses Pillow only for image embeddings (never called here); the text-embedding path is exercised by the whole test suite with Pillow 13. **This overrides an upstream-declared bound; remove it when fastembed lifts the cap.**
- **Untested version found:** the Dockerfile asked for `rapidocr-onnxruntime>=1.3`, but the version all OCR tests and measurements used is 1.2.3 (the newest for Python 3.13); the image on Python 3.12 would have pulled an untested 1.4.x. Now pinned to the tested version.
- **Pinning:** `constraints/<service>.txt` (backend 45, rag-core 65, agent-core 23, llm 23 packages) fix every package to the version the tests ran against where one exists (the rest to the resolver's choice, listed in each file's header), generated for CPython 3.12 / Linux x86-64 by `scripts/make_constraints.py`, which also proves the pins are mutually consistent for that platform. All four Dockerfiles install with `-c constraints/<service>.txt`. The requirements files keep their ranges (what the code needs); the constraints say what is built.
- **Not verified:** a built image. The Docker daemon was not running when this was written; see `docs/verification/FEATURE_MATRIX.md` for the status of the image build.

## Bugs found by this work

- `LocalStorage._path` used `Path.resolve()` for the confinement check. On Windows `resolve()` can return the `\\?\C:\...` form while another request is creating the directory, so a valid key was rejected (`ValueError` → HTTP 500) under concurrent first uploads: 1 failed upload in each of four load-test runs. Found by `scripts/load_test.py`, reproduced, fixed with a purely lexical check, regression tests added; reruns had 0 errors. Linux deployments were not affected by this specific race.

## Residual risks and gaps (not fixed)

- No parser sandbox: pdf/docx/image parsers run in the worker process. The new time budgets are **cooperative** (checked between pages); a single pathological page can still run unbounded because a Python thread cannot be killed. Isolation needs a subprocess with limits.
- Prompt injection: the structural defence (delimiters, system/user separation) is tested with a mock model only. Whether a real model obeys plain-language instructions that stay inside the evidence is **unmeasured**.
- Cross-user trust rests on one shared internal token between services; a compromised service could pass any `user_id`.
- The shared stream cap needs Redis; the three tests that use a real Redis server were skipped (no Redis available locally); the same script ran on an in-process Redis implementation.
- No SAST/secret scanning/container scanning/DAST. `ruff`/`mypy` are not configured.
- Polyglot coverage is a few cases, not a corpus; antivirus scanning does not exist.
- Authentication, CORS, cookies and deployment settings were exercised through hermetic tests, not against the deployed services.
