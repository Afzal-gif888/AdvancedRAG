# Module 3: Chunking and Document Parsing

> Part of [ADVANCED_RAG_2026_MASTER_COURSE.md](../../ADVANCED_RAG_2026_MASTER_COURSE.md). Builds on [Module 1](MODULE_1_SYSTEM_ARCHITECTURE.md) (ingestion flow) and [Module 2](MODULE_2_EMBEDDINGS_AND_RETRIEVAL.md) (what an embedding is and its 512-token limit).
> Labels: **[Code-read]** I read the source; **[Ran]** I executed it in this session; **[Documented]** a repo doc says so, not reproduced by me; **[Missing]** not implemented; **[Unknown]** could not determine.

> **Update after the implementation pass (2026-10-23, working tree on top of `31bad7e`, uncommitted).** This module was first written against `31bad7e`.
> The sections below have been rewritten where the code changed: OCR and image uploads, partial-extraction warnings, text normalisation,
> heading-aware and semantic chunking, streaming to RAG Core, durable storage (`local`/`s3`/`db`) instead of the base64 job payload, upload
> deduplication, document versions/replace/restore, and reindex-from-chunks. Evidence for each is in
> [docs/verification/FEATURE_MATRIX.md](../verification/FEATURE_MATRIX.md). Everything here was verified with hermetic tests (local vector store,
> mock gateway, throw-away PostgreSQL); **none of it was run against Pinecone, the real LLM gateway or a deployment**, and none of it is committed.
> Line numbers from `31bad7e` that appear in older text have shifted; search by function name.

---

## 1. Learning objectives

1. Describe the full life of an uploaded file: validate, store, parse, chunk, embed, persist, verify, delete.
2. Name the parsing library for each supported format and its known limitations (OCR is optional and off by default; multi-column PDFs; PDF tables are read as text unless `PDF_TABLES` is enabled).
3. Explain fixed-size, recursive, token-based, semantic and structure-aware chunking, and say which ones AdvancedRAG implements, which is the default, and what the evaluation did and did not show.
4. Reason about chunk size and overlap, and predict how each choice changes retrieval and generation.
5. Explain tokenization limits and why `fit_to_token_limit` exists.
6. Explain the 25 MB question honestly: where the limit applies, what actually bounds a document, and why the base64-in-job-payload workaround was retired and what durable storage replaces it.
7. Verify, step by step, that a document was parsed, chunked, embedded and stored correctly.
8. Debug missing text, missing vectors and stale embeddings.

## 2. Prerequisites

Module 1 §5.1 (the upload trace) and Module 2 §3 (embeddings, 512-token limit). Basic Python string slicing.

---

## 3. Fundamentals

### 3.1 Why chunk at all?

An embedding model has an input limit (512 tokens here) and, more importantly, one vector can only "average" so much meaning. A 50-page PDF embedded as one vector would be a blurry summary of everything; a question about one sentence would not match it well. And the LLM cannot be handed the whole file for every question (cost, latency, context window, accuracy).

So we cut documents into **chunks**: pieces small enough to embed precisely and to paste several into a prompt, large enough to keep a complete thought.

### 3.2 The ingestion lifecycle (the whole pipeline in one table)

| Stage | What happens | Where (all [Code-read]) | Failure → code |
|---|---|---|---|
| 1. Validate | extension allowlist, magic-byte check, size ≤ 25 MB, non-empty, quotas, filename sanitised | `backend/app/services/document_service.py::upload` | 415 / 413 / 422 / 409 |
| 1b. Deduplicate | same user + same content hash returns the existing document (PostgreSQL advisory lock closes the race) | `document_service.upload` | HTTP 200 with `duplicate: true` |
| 2. Store | original saved under a server-generated key in the document's recorded backend (`local`, `s3` or `db`); for `db` the blob, document and job commit atomically | `document_service.py`, `integrations/storage.py`, `db/models/stored_object.py` | rollback + delete blob on commit failure |
| 3. Queue | `documents` row + `background_jobs` row in one commit; the job carries **no file bytes** | `document_service.py` | – |
| 4. Hand over | worker streams the stored object to a temp file (sha256 checked) and posts it to RAG Core, which spools it to disk | `workers/document_jobs.py`, `integrations/rag_core.py`, `rag_core/main.py:_spool` | `FILE_MISSING` (permanent, with diagnostics), `STORAGE_UNAVAILABLE` (retryable), `RAG_SERVICE_UNAVAILABLE` |
| 5. Parse | file → list of `Page(number, text, method, confidence, headings)` plus an `ExtractionReport` (native vs OCR pages, failed/low-confidence pages, warnings); text clean-up | `rag_core/parsing.py`, `textclean.py`, `ocr.py` | `PARSE_FAILED`, `NO_EXTRACTABLE_TEXT`, unsupported |
| 6. Chunk | `CHUNKING_STRATEGY`: `recursive` (default, 1200/200 chars), `structure` (heading-aware) or `semantic` | `rag_core/chunking.py` | – |
| 7. Token guard | split any chunk > 512 tokens | `chunking.py::fit_to_token_limit`, `pipeline.py::ingest` | – |
| 8. Chunk cap | reject > 5,000 chunks | `pipeline.py::ingest` | `DOCUMENT_TOO_LARGE` |
| 9. Embed | in slices, batch size 4 (Render) | `pipeline.py::ingest` | `EMBEDDING_*` |
| 10. Record extent | durable "may-exist ids" before writing | `pipeline.py::ingest`, `ingestion.py::_Control.record_extent` | – |
| 11. Persist | delete old chunk rows, insert new; CSV table; vector upsert | `pipeline.py::ingest` | rollback + `discard_index` |
| 12. Verify | read back first/middle/last + 1 per 100-vector request | `pipeline.py::verification_ids` | `VECTOR_INDEX_UNVERIFIED` |
| 13. Commit | ownership re-check under row lock, then commit | `ingestion.py::_Control.before_commit` | `IngestionCancelled` |
| 14. Ready | document `ready` | `document_jobs.py::ingest_document` | – |

### 3.3 Supported formats and the actual libraries

| Format | Library / method | What you get | Known limitations |
|---|---|---|---|
| PDF | `pypdf.PdfReader`, `page.extract_text()` per page; pages with (almost) no text layer can go to OCR | one `Page` per PDF page, with page number and a `method` (`native` or `ocr`) | reading order follows pypdf (multi-column layouts may interleave); encrypted PDFs rejected (`PARSE_FAILED`); tables become flat text **unless `PDF_TABLES` is enabled** (§3.3b). OCR only when `OCR_ENABLED=true`; native extraction has a wall-clock budget (`PDF_PARSE_TIMEOUT_SECONDS`) that is checked between pages |
| DOCX | `python-docx` (`iter_inner_content`) | paragraphs and tables **in document order**; heading paragraphs recorded so heading-aware chunking can use them; repeated header rows in tables not indexed as data | no images, headers/footers, footnotes; zip structure checked first (bomb guard) |
| CSV | `csv` module with `Sniffer` | rows as `column: value \| ...` lines plus the **full table** for exact analysis; repeated header rows dropped | first row is a header only if it looks like one |
| TXT, MD | decode `utf-8-sig`, `cp1252`, `latin-1`; clean-up | text; Markdown `#` headings are recognised for heading-aware chunking | – |
| Images (PNG/JPEG) | Pillow decode + OCR engine (RapidOCR/ONNX) | one `Page` per image, with engine-reported confidence | needs **both** `OCR_ENABLED=true` in RAG Core **and** `png,jpg,jpeg` in the backend's `ALLOWED_UPLOAD_EXTENSIONS` (default is `pdf,txt,md,csv,docx`); pixel limit `IMAGE_MAX_PIXELS`; without OCR enabled images are refused as unsupported |

[Code-read + Ran: `rag-core/tests/test_ocr_extraction.py`, `test_tables_extraction.py`, `test_hostile_inputs.py`.]

### 3.3b Tables in PDFs (`PDF_TABLES`, default `off`)

`rag_core/pdftables.py` + `parsing._structure_tables` use pdfplumber to find tables on text-layer pages and render each row like CSV/DOCX rows (`Header: value | Header: value`), with the text above and below the table kept in order.

- **`lines`**: tables drawn with ruling lines. Cell boundaries come from the lines, so this is the reliable case.
- **`auto`**: `lines`, then borderless tables inferred from text alignment on pages where no ruled table was found. Heuristic: a candidate is accepted only if it has >= 3 rows, at most 12 columns, most rows fill most columns, and cells are short, so prose and lists are normally rejected; nothing can guarantee that a page of aligned text is or is not a table.
- **Multi-page tables:** a table that starts near the top of a page with the same column count as the previous page's last table is a continuation if its first row repeats the header (dropped) or contains a numeric cell (a record, kept). A continuation inherits the previous header; a headerless table never gets guessed column names.
- **Scanned pages:** OCR rows whose cells are separated by wide gaps (`" | "`) and line up across >= 3 consecutive lines are rendered the same way, and the page is flagged `TABLE_FROM_OCR_LAYOUT` because the relationships are inferred from positions. A row with a missing cell ends the run (no structure is inferred across a gap).
- **Failure behaviour:** never fatal. A missing library, a damaged PDF, a page error or the time budget (`PDF_TABLE_TIMEOUT_SECONDS`) leaves the page's plain text and adds a warning. The extraction report lists each structured table (`page`, `kind`, `rows`, `columns`, `header`, `continued`).
- **Evidence:** 17 tests in `test_pdf_tables.py` (bordered, borderless, multi-page with repeated and missing headers, empty cells, prose not mistaken for a table, malformed PDFs, budget, no pdfplumber, scanned table). On the retrieval benchmark the 62 PDF-table questions showed no measurable difference with or without structure (they are easy either way), so the default stays `off`: the benefit is fidelity, not a demonstrated retrieval gain. Not handled: merged cells spanning rows, nested tables, rotated tables, tables as images other than the OCR-layout case.

### 3.4 Scanned PDFs, OCR and image-based documents

A scanned PDF is a picture of text; `pypdf` returns no text for it. With `OCR_ENABLED=false` (the default, and the way production is configured) nothing changed: such pages are dropped and a document with no text at all fails with the permanent `NO_EXTRACTABLE_TEXT`.

With `OCR_ENABLED=true` the pipeline (`rag_core/ocr.py`, `_ocr_pages` in `parsing.py`) works like this:

1. A page with fewer than `OCR_MIN_NATIVE_CHARS` native characters is an OCR candidate (so a *mixed* PDF keeps its native pages and OCRs only the scanned ones).
2. `pypdfium2` renders the page at `OCR_RENDER_DPI` (capped by `OCR_MAX_PIXELS`); a blank page is detected and skipped, not reported as a failure.
3. The engine (RapidOCR on ONNX Runtime, self-hosted, no network) recognises text under a per-page timeout (`OCR_PAGE_TIMEOUT_SECONDS`) and a per-document budget (`OCR_DOCUMENT_TIMEOUT_SECONDS`, `OCR_MAX_PAGES_PER_DOCUMENT`).
4. The only confidence recorded is what the engine reports; below `OCR_MIN_CONFIDENCE` the page is flagged.
5. The `ExtractionReport` lists native/OCR/failed/low-confidence pages and warnings (e.g. "12 of 40 pages could not be read"); it is stored with the document and returned by the document status API, so a partially read document is **no longer silently "ready"** (the old gap).

**Measured accuracy** (`scripts/eval_ocr.py`, `docs/verification/OCR_ACCURACY.md`; synthetic samples rendered from known text, so ground truth is exact): mean character error rate **0.019** over 44 English samples in 11 conditions (noise, blur, down-sampling, rotation up to 5 degrees, JPEG q30, four fonts); accented Latin about 0.05-0.10; Chinese 0.00; **Russian, Greek and Korean were not read** (CER 0.78-0.87). Two findings matter more than the mean: one clean sample **lost a whole line** (an article-number line) with no warning, and page confidence correlates only 0.11 with accuracy, so a high score does not mean a complete page. Threshold tuning did not help. Page order and numbers were preserved on scanned and mixed PDFs. Not measured: handwriting, photographs, other layouts, long documents, production-hardware speed. [Ran]

### 3.5 Text normalization

`normalize()` still converts line endings, removes NULs and collapses whitespace. With `TEXT_CLEANUP=true` (default) `rag_core/textclean.py` additionally: removes control characters and soft hyphens, expands ligatures, **de-hyphenates words broken at a line end** (while keeping real compounds, identifiers such as part numbers, e-mail addresses and URLs intact), and removes **repeated page headers/footers** (only on documents with enough lines to tell, `min_lines=5`). Unicode forms are normalised conservatively. These are heuristics, English-centric, and covered by `test_textclean.py`; a wrong guess can still merge two words, which is why the originals stay in storage.

### 3.6 Metadata and attribution

Chunk rows keep `document_id`, `user_id`, `chunk_index`, `page`, `section` (heading path such as "Travel > Hotels", only where the source has reliable headings) and `extraction_method` (`native`/`ocr`). Citations show `[n] filename (page p)` and, when a section exists, the section; the snippet is still the first 300 characters. The document row records the embedding model/dimensions, the pipeline settings used (`pipeline_info`) and the extraction report. Not kept: author, dates, language.

---

## 4. Technical deep dive

### 4.1 Chunking strategies

| Strategy | Idea | Pros | Cons | In AdvancedRAG? |
|---|---|---|---|---|
| **Fixed-size (characters)** | cut every N characters | trivial, predictable | splits sentences/words/table rows | No (only as the last-resort fallback inside `_split`) |
| **Token-based** | cut every N model tokens | respects the model's limit exactly | needs the tokenizer; still ignores structure | **Only as a guard**: oversized chunks are re-split by `fit_to_token_limit` |
| **Recursive (separator-preferring)** | try big separators first (paragraph), fall back to smaller (line, sentence, clause, word) | keeps paragraphs/sentences together | still size-driven; no notion of headings | **Yes, the main strategy** (`SEPARATORS = "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "`) |
| **Structure-aware** | split by headings/sections/table rows | chunks align with document meaning; the heading path is stored as `section` and (with `EMBED_SECTION_CONTEXT`) prepended to the text that is embedded, not to the stored text | needs reliable structure detection per format | **Yes, `CHUNKING_STRATEGY=structure`** (`chunk_pages_structured`): Markdown and DOCX headings, PDF headings located by `locate_headings`; a parent heading directly followed by a child is folded. Not the default |
| **Semantic** | split where embedding similarity between adjacent sentences drops | topic-coherent | costs one embedding per sentence at ingest; unstable; harder to test | **Yes, `CHUNKING_STRATEGY=semantic`** (`semantic_groups`, breakpoint at the `SEMANTIC_BREAKPOINT_PERCENTILE`, bounded by `SEMANTIC_MAX_SENTENCES`). Not the default |
| **Sliding window with overlap** | consecutive chunks share some text | facts near a boundary survive in at least one chunk | duplicates storage and may duplicate evidence | **Yes** (200 chars) |

All three strategies exist and are unit-tested, but **the evaluation does not justify changing the default**: on the synthetic corpus in `docs/verification/RETRIEVAL_EVALUATION.md` heading-aware chunking helped only when fused with BM25 and semantic chunking did not beat recursive (and embeds ~7x more texts per document). Say exactly that; do not claim either is "better".

### 4.2 The actual algorithm, step by step

`chunk_pages(pages, size, overlap)` processes each page independently (so a chunk never spans pages):

1. `_split(text, size, SEPARATORS)` recursively breaks text into pieces ≤ `size`: it takes the first separator found in the text, splits on it (keeping the separator attached), and recurses with the *next* separators on any piece still too big. If no separator exists it slices every `size` characters.
2. `_merge(pieces, size, overlap)` packs consecutive pieces into chunks up to `size`. When the next piece would overflow, it emits the chunk and **seeds the next chunk with a tail** of the previous one (the last `overlap` characters), trimmed to start at a line boundary if the tail contains a newline, otherwise at a word boundary.
3. Chunk text is `.strip()`ed and indices are assigned sequentially across the document.

Because the overlap tail is *added to* the next chunk, **a chunk can reach `size + overlap`** characters; the docs say so explicitly (`docs/rag-architecture.md §1`), and my run showed a 161-character chunk with `size=160`. [Ran]

### 4.3 Demonstration with the real code

I ran the repository's own `chunk_pages` on a 400-character sample with a small size (160) and overlap (30) so boundaries are visible. [Ran, `rag_core.chunking.chunk_pages`]

Sample:

```text
Section 2: Travel and expenses
Hotel stays are reimbursed up to 180 euros per night in all European cities. Flights must be booked in economy class unless the flight time exceeds 9 hours, in which case premium economy is allowed.

Section 3: Information security
Laptops must use full-disk encryption. Security incidents must be reported within 24 hours of discovery. Lost devices are wiped remotely.
```

**Repository's recursive chunker (size 160, overlap 30):**

| # | len | text |
|---|---|---|
| 0 | 107 | `Section 2: Travel and expenses\nHotel stays are reimbursed up to 180 euros per night in all European cities.` |
| 1 | 145 | `in all European cities. Flights must be booked in economy class unless the flight time exceeds 9 hours, in which case premium economy is allowed.` |
| 2 | 60 | `premium economy is allowed.\n\nSection 3: Information security` |
| 3 | 161 | `3: Information security\nLaptops must use full-disk encryption. Security incidents must be reported within 24 hours of discovery. Lost devices are wiped remotely.` |

**Naive fixed-size (160 characters, step 130) on the same text:**

| # | text (cut points in bold reasoning) |
|---|---|
| 0 | `Section 2: ... Flights must be booked in economy class unless the f` ← cuts mid-word |
| 1 | ` in economy class ... Section 3: Information security\nLaptops must use full-disk ` ← mixes two sections |
| 2 | `ty\nLaptops must use ... wiped remotely.` ← starts mid-word |
| 3 | ` remotely.` ← useless fragment |

What to notice (and say in an interview):

- Recursive chunking kept sentences whole and put "Section 3" content in its own chunk; fixed-size glued two topics and cut words ("the f", "ty").
- Chunks 0 and 1 overlap on "in all European cities." That repetition is deliberate.
- **A real imperfection:** chunk 3 begins with `3: Information security`; the overlap tail was trimmed to a *word* boundary, which dropped the word "Section" from the heading. Overlap is not heading-aware. With the production size (1200) this affects a smaller share of text but the same mechanism applies. This is a good "limitations" talking point: *structure-aware chunking would keep the heading with its content.*
- Why retrieval changes: a question "What is the hotel limit?" matches chunk 0 strongly (short, focused) in the recursive version; in fixed-size, the same fact is split between chunks 0 and 1 (`...reimbursed up to 180 euros per night in all European cities. Flights must be booked in economy class unless the f`) so chunk 0 contains the hotel fact but diluted with flight text. Bigger chunks dilute; smaller chunks lose context.

### 4.4 Chunk size and overlap: the trade-off

| Choice | Retrieval | Generation | Cost |
|---|---|---|---|
| Small chunks (200–400 chars) | precise matches, but a fact may need context from neighbours | many tiny sources; the LLM may lack context | more vectors (more embeddings, more Pinecone storage) |
| Large chunks (1500+ chars) | blurry vectors, lower precision; more topics per chunk | rich context per source, fewer sources fit | fewer vectors; but 512-token limit reached sooner |
| Overlap 0 | boundary facts can be cut in half | – | cheapest |
| Overlap ~15–20% (200/1200 ≈ 17%) | boundary facts survive | duplicate text may appear in two retrieved chunks, wasting context | ~17% more embeddings |
| Overlap too big | near-duplicate chunks crowd the top-k | repeated evidence | high |

AdvancedRAG's default: **1200 characters, overlap 200, final context ≤ 12,000 characters (≈ top 6 chunks)** [Code-read: `config.py`]. 1200 characters ≈ 250–300 English tokens (rule of thumb ≈ 4 characters per token; illustrative, depends on text). The config validator requires `CHUNK_OVERLAP < CHUNK_SIZE` (`check_safety`). The authors' guard test freezes these values (`test_frozen_contract.py`) because changing them changes every stored vector. [Documented]

There is **no experiment in the repo that compares chunk sizes** [Missing], so "1200/200 is optimal" is not a claim you can make; "it is the configuration the baseline was measured with" is.

### 4.5 Tokenization and model limits

A **token** is a sub-word unit produced by the model's tokenizer (BERT-style WordPiece for bge). "unbelievable" may be 3 tokens. Prose averages ~1.3 tokens per word; **numbers, codes and tables use many more tokens per character**: `ZRy-SL18` or `1,234.56 | 789` fragment heavily.

The embedding model ignores everything beyond 512 tokens (silent truncation). That is dangerous: the text looks indexed but the tail is invisible to semantic search. AdvancedRAG's guard (`fit_to_token_limit`, `chunking.py::fit_to_token_limit`) counts real tokens (using the model's tokenizer copy with truncation disabled, `embeddings.py::LocalEmbeddings._load`) and splits any chunk > `max_tokens` in half recursively; it renumbers chunk indices afterwards. A regression test exists (`test_rag_quality.py`, "no silent embedding truncation"). [Code-read; the selected subset of this file passed in my run — see master course "Test runs"]

Note the **different limits** in the pipeline: embedding model 512 tokens; reranker 1,024 tokens per query+passage pair (Pinecone docs); the LLM's own context window (Module 5); `MAX_CONTEXT_CHARS = 12,000` characters of evidence.

### 4.6 Duplicate chunks and repeated context

Sources of duplication, and what the code does:

| Source | Handled? |
|---|---|
| Overlap between consecutive chunks | By design; no dedup afterwards |
| The same file uploaded twice | **Yes, now.** The upload path looks up `(user_id, checksum_sha256)` among live documents; a match returns the existing document (`duplicate: true`) and queues nothing. Concurrent identical uploads are serialised by a PostgreSQL advisory lock (that test runs on PostgreSQL only). Per user: another user's identical file is a separate document and never reveals the first. [Ran: `test_durability_versions.py`] |
| Boilerplate repeated on every page (headers/footers) | No. Each page's header text becomes part of every page's chunk |
| Near-duplicate documents (Aurora X1 vs X2 spec sheets) | Intentional in the benchmark; handled by reranking and grading, not by dedup |

Effect: duplicates can fill several of the 6 final slots with the same evidence, lowering diversity and wasting the 12,000-character budget. The `_merge` helper in the orchestrator dedups by `chunk_id` only for the correction loop (`orchestrator.py::_merge`), not by content. Upload dedup is implemented as above; deduplicating *retrieved chunks* by normalised-text hash before building context is still **not** implemented.

### 4.7 Persistence: what is stored where

| Data | Store | Key |
|---|---|---|
| Original file bytes | storage (`documents/<user_id>/<document_id>`) | server-generated |
| Document metadata, status | backend `documents` | `id` |
| Chunk text | RAG DB `rag_chunks.text` (+ GIN full-text index) | `<document_id>:<n>` |
| Document summary (model, dims, counts) | RAG DB `rag_documents` | `document_id` |
| Vectors | Pinecone namespace `user-<user_id>` | `<document_id>:<n>` |
| CSV full table | RAG DB `rag_tables` (JSONB), ≤ 200,000 rows × ≤ 100 columns | `document_id` |
| Ingestion attempt/lock | RAG DB `rag_ingestions` | `document_id` |

### 4.8 Updates, deletion, re-indexing, stale embeddings

- **Re-ingest of the same `document_id`** is idempotent: old `rag_chunks` rows deleted and replaced, the same deterministic vector ids are overwritten, ids beyond the new chunk count are deleted (`pipeline.py::ingest`), and the highest chunk count ever indexed (`max_chunk_count`) lets later deletes cover every id that may exist. [Code-read]
- **Who triggers re-ingest?** `POST /documents/{id}/reprocess` still works only for `failed` documents, and now reads the file from the document's recorded durable storage. **Versioning exists**: `POST /documents/{id}/replace` uploads a new version that shares the `logical_id`; the current version keeps answering until the new one is indexed, then an atomic switch marks the old one `SUPERSEDED` (identical content is a no-op). `GET /documents/{id}/versions` lists versions, `POST /documents/{id}/restore` makes an old one current again, and a retention job (`SUPERSEDED_RETENTION_DAYS`, default 7) purges superseded versions. `replaces_id` deliberately has no database foreign key (SQLite migration limits). [Ran: `test_durability_versions.py`, Playwright]
- **Reindex without the file**: RAG Core `POST /api/v1/rag/documents/{id}/reindex?user_id=` re-embeds a document from its stored chunk rows, overwriting the same deterministic vector ids with the current model (it never deletes, never touches other users, refuses while an ingestion runs). Intended for repairing lost or model-mismatched vectors; not exposed in the UI. [Ran: `test_integrity_and_spool.py`]
- **Delete** is soft first (`status=deleted`, detached from conversations immediately), then a `DOCUMENT_DELETE` job removes vectors (`DELETE /rag/documents/{id}`), the stored object from whichever backend holds it, and any legacy base64 job payload, and finally hard-deletes the row; a reconciler re-drives failed deletes up to `MAX_DELETE_REDRIVES = 5`. Deleting a document also covers its superseded versions. [Ran: `test_durability_versions.py`, `test_reliability.py`]
- **Stale embeddings** arise when the model or chunking parameters change while old vectors remain. The repo's answer is procedural: "never mix models in one index; new index + full re-ingest" (`docs/rag-architecture.md §6`). Retrieval now **refuses** documents whose recorded embedding model or dimension differs from the current provider (`EMBEDDING_MODEL_MISMATCH`, HTTP 409; mixed sets are filtered and counted), and `/ready` reports the integrity; the repair path is the reindex endpoint above. (Module 2 §3.5.) Also: "Documents indexed before 2026-10-08 keep their old chunks until they are reprocessed" [Documented].

### 4.9 Large documents and the 25 MB question

There are **three different limits**; people conflate them.

| Limit | Value | Where | What it bounds |
|---|---|---|---|
| Upload size | 25 MiB (+ 64 KiB multipart slack) | backend `MAX_UPLOAD_BYTES`, `BodySizeLimitMiddleware` | bytes of the file |
| RAG Core file size | 30 MiB | `MAX_FILE_BYTES` (`rag-core/config.py`) | bytes received by RAG Core |
| **Chunks per document** | **5,000** | `MAX_CHUNKS_PER_DOCUMENT` | **extracted text** |
| Ingestion time | 1 h of active polling | backend `INGEST_MAX_SECONDS` | wall-clock work |

The **binding limit is usually the chunk cap, not the file size.** With ~1,000 new characters per chunk (1200 minus ~200 overlap, illustrative), 5,000 chunks ≈ 5 million characters of text, roughly 5 MB of plain text. Therefore:

- A **25 MB scanned or image-heavy PDF** has almost no extractable text: with OCR off it passes the size limit, then fails with `NO_EXTRACTABLE_TEXT`; with OCR on it is bounded by `OCR_MAX_PAGES_PER_DOCUMENT` and the time budgets (not ladder-tested).
- A **25 MB plain-text/CSV file** (~25 million characters) would produce on the order of 25,000 chunks (25 MiB / ~1000 ≈ 26,000, illustrative arithmetic [Ran]) and be rejected with `DOCUMENT_TOO_LARGE`, but *after* it has been uploaded, stored, parsed and chunked. **[Ran, `scripts/perf_ingest.py`]**: on one laptop, 1 MB of text indexed (1,456 chunks, 0.8 s with fake embeddings); 5, 10 and 25 MB were refused with `DOCUMENT_TOO_LARGE` (7,300 / 14,548 / 36,378 chunks) in 0.2–2.7 s at a peak RSS of 125–221 MB. With the real embedding model on that laptop the measured rate was 34.8 chunks/s (a 256-chunk sample, solo; an earlier figure of 3.7 chunks/s was wrong because its timing window included a model download), so a 5,000-chunk document is roughly 2.5 minutes of embedding there (extrapolated). The deployed free instance (0.1 CPU) is documented at 1-2 s per chunk, which is what the cap is sized for. See `docs/verification/PERFORMANCE_AND_BACKUP.md`.
- The documented worst case that **succeeded**: an 812-page, 5.6 MB PDF with 4,824 chunks: ≈ 175 s parse, ≈ 885 s embed, ≈ 39 s upsert [Documented, `docs/rag-architecture.md`, measured on the author's setup; not reproduced by me]. On Render free (0.1 CPU) the docs estimate 1–2 s per chunk, so 5,000 chunks ≈ 2 h, beyond the 1 h ceiling [Documented].
- CSV exact-analysis table is skipped silently if the file has > 200,000 rows or > 100 columns (`parse_table` returns `None`); the file is still chunk-indexed if it fits the chunk cap.

**Memory (updated).** The upload is no longer read whole into memory on its way to RAG Core: the backend streams the stored object to a temporary file and RAG Core spools the multipart body to disk (`_spool`) and parses from the path (`test_integrity_and_spool.py`). Parsing still materialises the extracted text and chunk objects, so memory grows with the document (221 MB peak at 25 MB of text in the ladder run). The older analysis follows for context. FastAPI documents that `UploadFile` is a `SpooledTemporaryFile` (memory up to a threshold, then disk), while a parameter typed `bytes` loads everything into memory (source 1). The backend uses `UploadFile` and its `_scan` reads the upload in 1 MiB steps for hashing [Code-read]. But RAG Core does `data = await file.read(MAX_FILE_BYTES + 1)`: the **whole file in memory** (`rag-core/main.py`), then `parse` operates on that `bytes`, then page strings, chunk objects, and 384 Python floats per vector. Illustrative CPython estimate (not measured here): a list of 384 Python floats ≈ 384 × (8-byte pointer + 24-byte float object) ≈ 12 KB, so 5,000 vectors held in memory ≈ 60 MB, before counting text and chunk objects. The docs report a measured peak of 324 MiB for a 1,000-row CSV and a 30-page PDF under the Render free limits (512 MB) [Documented]; the 25 MB case was not measured in the repo [Unknown].

### 4.10 From base64-in-the-job to durable storage (the "file could not be found" fix)

**What was wrong at `31bad7e`.** With `STORAGE_BACKEND=local` on Render's ephemeral disk, `document_service.upload` base64-encoded the whole file into `background_jobs.payload` (x4/3: 25 MiB became 33.33 MiB), kept it for up to 30 days, and did not clear it on delete. Worse, `reprocess` and the reconciler enqueued new jobs with *no* payload, so after a restart wiped the disk they failed with "The uploaded file could not be found". [Code-read at `31bad7e`; reproduced in a test before the fix.]

**What the code does now.**

- Each document records the backend that holds its bytes (`documents.storage_backend`: `local`, `s3` or `db`; migration 0005, additive and backward compatible). New jobs reference the object (`storage_key`) and carry no bytes.
- `db` stores the blob in the application database (`stored_objects`) in the **same transaction** as the document and job rows, so there is no window where the job exists without its file; `s3` uses any S3-compatible bucket (tested with `moto`, not a real bucket).
- The worker verifies the sha256 recorded at upload when it reads the object back. A missing or corrupt object raises `FILE_MISSING` (permanent, with diagnostics naming the backend and key); an unreachable store raises the retryable `STORAGE_UNAVAILABLE`.
- Jobs queued by the old version (with `file_data`) are still processed, and the payload is scrubbed after use or on delete.
- Delete and purge remove the object from the backend recorded on the row.

**What is *not* fixed by code alone.** `render.yaml` still sets `STORAGE_BACKEND=local` with `ALLOW_LOCAL_STORAGE_IN_PRODUCTION=true`; production config was deliberately not changed. Until an operator selects `db` or `s3`, a Render restart still loses new uploads, but the failure is now explicit and diagnosable instead of a mystery, and nothing is duplicated into the job table. [Documented config; not deployed]

**Interview framing.** "I found the root cause (ephemeral disk plus jobs that did not carry or reference durable bytes), made storage a per-document, recorded property, made the db backend atomic with the job, verified integrity on read, kept old jobs working, and left the production switch to the operator because it changes infrastructure."

Base64 arithmetic is still worth knowing: output length = 4*ceil(n/3); 25 MiB becomes 34,952,536 characters (33.33 MiB). [Ran]

---

## 5. AdvancedRAG implementation: file-by-file

| File | Role |
|---|---|
| [`document_service.py`](../../backend/app/services/document_service.py) | validate, store, enqueue, soft delete |
| [`workers/document_jobs.py`](../../backend/app/workers/document_jobs.py) | start/join ingestion, poll, error mapping |
| [`rag_core/parsing.py`](../../rag-core/rag_core/parsing.py) | bytes → pages; CSV table |
| [`rag_core/chunking.py`](../../rag-core/rag_core/chunking.py) | split/merge/token guard |
| [`rag_core/pipeline.py`](../../rag-core/rag_core/pipeline.py) `ingest` | the sequence in §3.2 |
| [`rag_core/ingestion.py`](../../rag-core/rag_core/ingestion.py) | lock, heartbeat, stages, cleanup |
| [`rag_core/embeddings.py`](../../rag-core/rag_core/embeddings.py) | model, tokenizer, batching |
| [`rag_core/tables.py`](../../rag-core/rag_core/tables.py) | exact CSV analysis (declarative; no SQL/code execution) |

Design properties you should be able to state: **deterministic ids** (re-run overwrites), **extent recorded before writes** (cleanup knows everything that may exist), **read-back verification** before `ready`, **ownership re-check at commit**, **cleanup on every failure path**, **bounded retries**, **sanitized error codes** (parser internals never reach the user).

### Implementation status (after the implementation pass)

| Capability | Status |
|---|---|
| PDF/DOCX/CSV/TXT/MD parsing | Implemented and tested |
| Recursive page-aware chunking (default), token-limit guard | Implemented and tested |
| Heading-aware (`structure`) and `semantic` chunking | Implemented, unit-tested, evaluated on a synthetic corpus; not the default; no quality win demonstrated |
| OCR for scanned/mixed PDFs and PNG/JPEG | Implemented, off by default, real-engine tests; CER measured on synthetic samples (about 2% English, some scripts unreadable, a line can be missed silently) |
| Partial-extraction warnings (`ExtractionReport`) | Implemented; surfaced by the status API |
| Text normalisation (hyphenation, ligatures, repeated margins) | Implemented, on by default |
| Table extraction | CSV and DOCX tables; PDF tables via `PDF_TABLES=lines|auto` (default off; ruled tables reliable, borderless heuristic, scanned tables inferred from OCR positions and flagged) |
| Upload dedup, versioning/replace/restore/purge | Implemented and tested |
| Durable storage (`local`/`s3`/`db`), integrity check, explicit `FILE_MISSING` | Implemented and tested hermetically; production still configured with `local` |
| Streaming to RAG Core (temp file spool) | Implemented and tested |
| Reindex from stored chunks | Implemented and tested; no UI |
| Retrieved-chunk content dedup | Missing |
| Performance with a real 25 MB file | Partially measured: parse/chunk/refusal path only (`PERFORMANCE_AND_BACKUP.md`); PDF/OCR ladder and real-embedding end-to-end not measured |

---

## 6. Worked examples

### 6.1 Compare chunkers (educational; the production code is `rag_core/chunking.py`)

```python
# runnable
SEPS = ("\n\n", "\n", ". ", " ")

def fixed(text, size, overlap):
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]

def recursive(text, size, seps=SEPS):
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(seps):
        if sep in text:
            out = []
            parts = text.split(sep)
            for j, part in enumerate(parts):
                piece = part + (sep if j < len(parts) - 1 else "")
                out += recursive(piece, size, seps[i + 1:]) if len(piece) > size else [piece]
            return out
    return [text[k:k + size] for k in range(0, len(text), size)]

def merge(pieces, size):
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) > size:
            chunks.append(cur.strip()); cur = ""
        cur += p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

doc = "Hotel limit is 180 euros. Flights are economy.\n\nLaptops need encryption. Report incidents in 24 hours."
f = fixed(doc, 40, 0)
r = merge(recursive(doc, 40), 60)
assert any(c.endswith(("lim", "180 e", "Fligh")) or c.startswith(("it is", "uros")) for c in f) or len(f) > len(r) - 1
assert all("180 euros" in c or "180" not in c for c in r)      # the fact stays whole in the recursive version
assert "Laptops need encryption." in r[1] or "Laptops need encryption." in " ".join(r)
print(len(f), len(r))
```

### 6.2 Base64 expansion and what it means for a job row

```python
# runnable
import base64, math

def b64_len(n): return 4 * math.ceil(n / 3)

assert b64_len(25 * 1024 * 1024) == 34_952_536
assert b64_len(3) == 4 and b64_len(1) == 4 and b64_len(2) == 4
raw = bytes(range(256)) * 10
assert len(base64.b64encode(raw)) == b64_len(len(raw))
ratio = b64_len(25 * 1024 * 1024) / (25 * 1024 * 1024)
assert abs(ratio - 4 / 3) < 1e-6
print("33.33 MiB:", round(b64_len(25 * 1024 * 1024) / 2**20, 2))
```

### 6.3 What would 5,000 chunks mean?

```python
# runnable
def approx_chunks(text_chars, size=1200, overlap=200):
    step = size - overlap            # new characters per chunk (approximation)
    return -(-text_chars // step)    # ceiling division

assert approx_chunks(5_000_000) == 5000          # ~5 MB of text hits the cap
assert approx_chunks(25 * 1024 * 1024) == 26215  # 25 MiB of text would be ~5x over the cap
print("ok")
```

(This is arithmetic, not a measurement: real chunks vary with separators, paragraph lengths and the token guard splitting extra chunks.)

---

## 7. Common mistakes

1. Claiming "semantic chunking" or "heading-aware chunking" because chunks "look sensible". The code is recursive by separators.
2. Believing the 25 MB limit is what bounds a document. The 5,000-chunk cap (≈ 5 MB of text) and the ingestion time usually bind first.
3. Treating `ready` as proof that all pages are searchable (silent skip of empty pages).
4. Forgetting that overlap increases chunk length and storage (a chunk can be `size + overlap`).
5. Choosing chunk size without measuring on your own questions.
6. Letting the embedding model silently truncate (the reason for the token guard).
7. Assuming re-uploading a corrected file replaces the old one. It creates a second document.
8. Using `bytes` instead of `UploadFile` for uploads (loads everything in memory), or reading a "streamed" file fully into memory later.
9. Parsing quality issues blamed on the embedding model: hyphenation, ligatures, repeated headers, two-column interleaving, tables flattened. Always inspect the extracted text first.
10. Changing `CHUNK_SIZE`/`CHUNK_OVERLAP` without re-ingesting everything and re-running the benchmark.

---

## 8. Interview questions

### Beginner

**B1. Why do we chunk documents?** Model input limits, precision of one vector per idea, and prompt budget: retrieve a few relevant pieces, not the whole file.

**B2. What formats does AdvancedRAG support and with what libraries?** PDF (pypdf, plus optional OCR via pypdfium2 + RapidOCR), DOCX (python-docx), CSV (csv module), TXT/MD (decoded text), and PNG/JPEG through OCR when OCR is enabled and the backend allowlist includes them.

**B3. What is chunk overlap for?** So that a fact straddling a boundary appears whole in at least one chunk. Here 200 characters.

**B4. What happens to a scanned PDF?** OCR off (default): no text layer → `NO_EXTRACTABLE_TEXT`, permanent. OCR on: the scanned pages are rendered and recognised under time/page budgets, tagged `method=ocr` with the engine's confidence, and any pages that failed or were low-confidence are listed as warnings on the document.

**B5. Which chunking method does it use?** By default recursive, separator-preferring, page-aware, character-based with overlap, plus a token-limit guard. Heading-aware and semantic strategies exist behind `CHUNKING_STRATEGY` but are not the default.

### Intermediate

**I1. Walk through what happens to an uploaded PDF.** (Use §3.2.) Mention deterministic ids, extent, verification, commit-time ownership check.

**I2. How does chunk size affect retrieval and generation?** Small = precise but context-poor and more vectors; large = blurry vectors but richer context and fewer sources fit in the 12,000-character budget. Overlap protects boundaries but duplicates content.

**I3. Why is there a token guard if chunks are limited to 1,200 characters?** Characters ≠ tokens. Numbers, codes and tables take many more tokens per character, so a 1,200-char chunk can exceed 512 tokens and be silently truncated by the embedding model. The guard re-splits.

**I4. How would you verify that a document was indexed correctly?** (Section 10, E3.5.) Compare chunk count in `rag_chunks`, vector count in Pinecone namespace, spot-check a chunk's text against the source, fetch a vector and confirm 384 dimensions, unit norm, metadata, and that a query for a sentence from the document retrieves it. The project's benchmark performs exactly these checks for its test corpus [Code-read: `rag_benchmark.py::main`].

**I5. How does re-ingestion avoid duplicates?** Deterministic ids `<document_id>:<n>`; old chunk rows deleted; upsert overwrites; leftover ids beyond the new count deleted using `max_chunk_count`. This prevents duplicates *within* a document; two uploads of the same file by the same user are deduplicated at upload (the existing document is returned).

**I6. What are the failure modes during ingestion and how are they handled?** Transient (RAG down, Pinecone 5xx) → retry with backoff, bounded attempts; permanent (parse failed, no text, too large) → fail fast with a sanitized code; crash → heartbeat/stale detection and cleanup; delete during ingestion → cancel at the next checkpoint and discard writes.

### Advanced

**A1. A user uploads a 25 MB PDF. Predict what happens and where it can fail.** Passes upload checks; stored durably (the job holds only a reference); the worker streams it to RAG Core, which spools it to disk; pypdf parses (single-threaded; documented ≈ 0.2 s/page on the author's machine for the 812-page case); if text > ~5M characters → `DOCUMENT_TOO_LARGE`; if embedding is slow (Render 0.1 CPU ≈ 1–2 s per chunk) → hits the 1 h ceiling → `INGESTION_TIMEOUT` (not retried); if memory > 512 MB → process killed → heartbeat stale → retry. Be explicit that you are predicting from code and docs, not from a run.

**A2. Redesign ingestion for 25 MB files on a small instance.** Already done: stream upload to object storage, pass a storage reference (not bytes) to the worker, spool to disk in RAG Core, keep the per-document lock/extent logic, no base64 payload. Still open: stream-parse page by page (a generator) instead of materialising all pages, resumable per-batch checkpoints within one attempt (a failed attempt restarts from the beginning), and a separate CPU-heavy embed worker. Say which are done and which are not.

**A3. Compare fixed, recursive, semantic and structure-aware chunking for contract documents.** Contracts have numbered clauses and defined terms: structure-aware (split by clause, keep clause number/heading as metadata and in the chunk text) is best; recursive is a fair fallback; semantic adds cost with unclear benefit; fixed-size is worst. Evaluate with a labeled question set; do not assume.

**A4. How would you detect that half of a PDF's pages were scanned?** Implemented: the parser records native/OCR/failed/low-confidence pages in the `ExtractionReport` (stored on the document, returned by the status API) and, with OCR on, reads the scanned pages. With OCR off, the pages without text are listed as skipped in the report instead of vanishing silently.

**A5. Why store chunk text in PostgreSQL rather than Pinecone metadata?** Smaller index, easier deletion, FTS available on the same rows, avoids metadata size limits; cost: an extra lookup per query and two stores to keep consistent (hence verify/cleanup logic).

---

## 9. Follow-up questions

- What is your overlap as a percentage, and how many extra embeddings does it cost? (200/1200 ≈ 17%; roughly 1.2× vectors for the same text.)
- How do you chunk a table? (One self-describing line per row; overlap starts on a line boundary; CSV also stored as a table for exact aggregation.)
- Why is the first row of a CSV sometimes not treated as a header? (Header only if ≥ 2 distinct non-empty non-numeric names.)
- What encoding problems can occur? (UTF-8 BOM, Windows-1252 smart quotes; decode order utf-8-sig → cp1252 → latin-1 never fails but may mis-decode.)
- How do you handle a document in French? (`bge-small-en` is English-only [Documented]; a multilingual model + new index + re-ingest.)
- How do you test large-document ingestion without a real 25 MB file? (Generate synthetic files of controlled size/complexity; measure parse/embed/upsert separately; use the mock Pinecone used by `test_ingestion_speed.py`.)

## 10. Practical exercises

### E3.1 Implement overlap-aware fixed chunking with tests

*Problem.* `chunk(text, size, overlap)` returns chunks of at most `size` characters where each chunk starts `size - overlap` after the previous one. Reject `overlap >= size`.

```python
# runnable
def chunk(text, size, overlap):
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("need 0 <= overlap < size")
    step = size - overlap
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        if i + size >= len(text):
            break
        i += step
    return out

assert chunk("abcdefghij", 4, 1) == ["abcd", "defg", "ghij"]
assert chunk("abc", 10, 2) == ["abc"]
assert chunk("", 4, 1) == []
try:
    chunk("x", 3, 3)
except ValueError:
    pass
else:
    raise AssertionError
print("ok")
```
*Complexity:* O(n) time, O(n·size/step) space (duplicated overlap). *Edge cases:* empty text, overlap ≥ size, last short chunk. *Follow-up:* "How would you avoid cutting words?" (snap to the nearest whitespace, as `_merge` does.)

### E3.2 Content-hash dedup of chunks

```python
# runnable
import hashlib, re

def norm(t): return re.sub(r"\s+", " ", t.strip().lower())

def dedupe(chunks):
    seen, out = set(), []
    for c in chunks:
        h = hashlib.sha256(norm(c).encode()).hexdigest()
        if h not in seen:
            seen.add(h); out.append(c)
    return out

assert dedupe(["Page 1 Confidential", "Hotel 180", "page 1   confidential"]) == ["Page 1 Confidential", "Hotel 180"]
print("ok")
```
*Follow-up:* exact-hash dedup misses near-duplicates; discuss MinHash/SimHash or embedding similarity > 0.98 (and its cost).

### E3.3 Upload-level idempotency by checksum

```python
# runnable
import hashlib

class Store:
    def __init__(self): self.by_hash, self.docs = {}, {}
    def upload(self, user, name, data: bytes):
        h = hashlib.sha256(data).hexdigest()
        key = (user, h)
        if key in self.by_hash:
            return self.by_hash[key], False       # existing document, nothing queued
        doc_id = f"doc{len(self.docs) + 1}"
        self.docs[doc_id] = name; self.by_hash[key] = doc_id
        return doc_id, True

s = Store()
a, created_a = s.upload("u1", "a.pdf", b"same")
b, created_b = s.upload("u1", "copy.pdf", b"same")
c, created_c = s.upload("u2", "a.pdf", b"same")      # another user: never shared
assert (a, created_a, b, created_b) == ("doc1", True, "doc1", False) and c != a and created_c
print("ok")
```
*Production note:* dedup must be per user (tenant isolation) and race-safe (unique index on `(user_id, checksum_sha256)` plus `INSERT ... ON CONFLICT`).

### E3.4 Debugging: "the answer is in the PDF but never found"

Checklist: (1) Download/extract the text of that page with pypdf; is the sentence present? (2) If missing: scanned page or font-encoding issue → OCR; (3) Is the sentence split across two chunks? Print neighbouring chunks; (4) Is the chunk > 512 tokens (truncated)? Count tokens; (5) Is the vector present in Pinecone (fetch `<doc>:<n>`)? (6) Does a query that *quotes* the sentence retrieve it? If yes, the problem is the user's phrasing vs the chunk (query rewriting/hybrid); if no, indexing is wrong.

### E3.5 Verify an ingested document (a runnable plan with SQL)

With read-only access to the RAG DB:

```sql
-- 1. the document was indexed and by which model
SELECT document_id, filename, chunk_count, max_chunk_count, page_count, embedding_model, embedding_dimensions
FROM rag_documents WHERE document_id = :doc;

-- 2. chunk rows match the recorded count and indices are contiguous
SELECT count(*) AS n, min(chunk_index) AS lo, max(chunk_index) AS hi
FROM rag_chunks WHERE document_id = :doc;      -- expect n = chunk_count, lo = 0, hi = n - 1

-- 3. no empty or oversized chunks
SELECT count(*) FILTER (WHERE length(text) = 0) AS empty,
       max(length(text)) AS longest              -- expect <= CHUNK_SIZE + CHUNK_OVERLAP (token-split chunks are shorter)
FROM rag_chunks WHERE document_id = :doc;

-- 4. the ingestion record says succeeded
SELECT status, stage, chunk_count, error_code, timings FROM rag_ingestions WHERE document_id = :doc;
```
Then in Pinecone: `describe_index_stats` namespace `user-<user_id>` vectorCount should include this document's `chunk_count` vectors; fetch ids `<doc>:0`, `<doc>:<n/2>`, `<doc>:<n-1>` and check `len(values) == 384`, norm ≈ 1, metadata contains only `document_id`, `chunk_index`, `page`. Finally query a sentence copied from the middle of the document and confirm that chunk is in the top 6. (The repo's benchmark script does the equivalent for its fixed corpus.) [I did not execute these queries; there is no database in this session.]

### E3.6 Test large-document ingestion (design)

Generate synthetic inputs: (a) 5 MB plain text of 5,000 chunks (boundary), (b) 5.1 MB (expect `DOCUMENT_TOO_LARGE`), (c) 25 MB text-poor PDF (expect `NO_EXTRACTABLE_TEXT`), (d) 25 MB CSV with 300k rows (expect table skipped). Use the mock vector store and fake embeddings from `rag-core/tests` so no external cost is incurred; assert on error codes, peak RSS (`tracemalloc`/`psutil`) and stage timings. Treat any numbers you obtain as your measurements for your machine, not as production claims.

### E3.7 Fix the heading-trimming overlap (stretch)

Modify `_merge` so that when the overlap tail begins mid-line it starts at the previous *line* start if that line is shorter than 120 characters (a heading heuristic). Write a test using the sample in §4.3 asserting chunk 3 starts with `Section 3: Information security`. This is a design exercise: do not change production code without re-running the benchmark gate (`docs/rag-architecture.md §6`).

## 11. Revision notes

- Pipeline: validate → store → job → parse → chunk → token guard → cap 5,000 → embed → record extent → persist → verify → commit → ready.
- Parsers: pypdf, python-docx, csv, text decoding; optional OCR (pypdfium2 + RapidOCR) for scans and PNG/JPEG, off by default.
- Chunking: default recursive by separators, page-aware, 1200/200, can reach 1400; token guard at 512; optional `structure` and `semantic` strategies (not default, no demonstrated win).
- Binding limit is chunks (≈ 5 MB of text), not 25 MB of bytes.
- Base64 = ×4/3 (25 MiB → 33.33 MiB): the retired workaround. Now: per-document storage backend (`local`/`s3`/`db`), jobs reference objects, sha256 verified, `FILE_MISSING` explicit; production config still `local`.
- Upload dedup per user; versions via replace/restore with an atomic switch; reprocess still only for failed docs; reindex from chunks.
- Re-ingest idempotent via deterministic ids + extent.

## 12. Assessment (attempt before reading the answer key)

1. List the supported formats, their parsing libraries, and one limitation of each.
2. With `size=100, overlap=20`, what is the maximum length of a chunk produced by `chunk_pages`, and why?
3. Compute the base64 length of a 10 MiB file. What fraction larger is it?
4. A 20 MiB text file of ordinary prose is uploaded. Predict the outcome and the error code. Explain with numbers.
5. Explain why a 1,200-character chunk might exceed 512 tokens, and what the code does.
6. A user uploads the same PDF twice. What exists afterwards in the backend DB, RAG DB and Pinecone? How is it prevented, and what about two simultaneous uploads?
7. A PDF with 40 pages, 12 scanned. What do users see with OCR off and with OCR on?
8. List the steps to confirm a document's content is parsed, chunked, embedded and stored.
9. After the Render free instance restarts, a user clicks "Retry" on a failed document that was uploaded with local storage. What happens, and what changes with `db` or `s3` storage?
10. Argue for or against changing `CHUNK_SIZE` to 600. How would you test the decision?
11. What is the difference between `UploadFile` and `bytes` parameters, and where does the project read a whole file into memory anyway?
12. Why is the vector extent recorded before the first vector is written?

---

## 13. ANSWER KEY (do not read until you have attempted the assessment)

**A1.** PDF – pypdf (no OCR, reading order may interleave columns, encrypted rejected); DOCX – python-docx (headings are plain paragraphs; images/headers/footers not extracted); CSV – csv module with delimiter sniffing (header heuristic; very large tables skipped for exact analysis); TXT/MD – decode utf-8-sig → cp1252 → latin-1 (Markdown structure not used). Images are unsupported.

**A2.** Up to `size + overlap` = 120 characters. After a chunk is emitted, `_merge` sets `current = tail` and then unconditionally does `current += piece` for the piece that triggered the overflow, with no size check on that first addition; the size test only guards *further* pieces. So the first piece after a boundary can be up to `size` long on top of a tail of up to `overlap`. The docs say the same ("a chunk can reach size + overlap") and I observed 161 characters for size 160 / overlap 30. *Common error:* assuming strictly ≤ size.

**A3.** 10 MiB = 10,485,760 bytes; 4·⌈10,485,760/3⌉ = 4 × 3,495,254 = 13,981,016 characters ≈ 13.33 MiB, 33.3% larger.

**A4.** 20 MiB ≈ 20.97 million characters. At ~1,000 new characters per chunk (illustrative) ≈ 20,972 chunks > 5,000 → `DOCUMENT_TOO_LARGE` (permanent, 413). The file passes the 25 MB upload limit, is stored, queued, parsed and chunked first. In local-storage mode it is also copied to the job payload (~26.7 MiB).

**A5.** Characters ≠ tokens: numbers, IDs and tables tokenize into many more tokens per character, so 1,200 characters can exceed 512 tokens; the embedding model would silently truncate the tail. `fit_to_token_limit` counts real tokens with the model's tokenizer and recursively halves chunks above the limit.

**A6.** Now: the second upload returns the existing document (`duplicate: true`): one `documents` row, one stored file, one job, one set of chunks and vectors. (At `31bad7e`: two of everything.) Races between simultaneous identical uploads are closed with a PostgreSQL advisory lock; the same file from a different user is a separate document.

**A7.** OCR off: the document is `ready` with a visible warning that 12 of 40 pages had no extractable text, and questions about them abstain. OCR on: those pages are recognised (method `ocr`, engine confidence recorded) and any that fail or score low are listed. (At `31bad7e` the pages vanished silently; that was the bug.)

**A8.** Check document row (model, dims, counts), chunk-row count/contiguity, no empty/oversized chunks, ingestion record status, Pinecone namespace count and sampled vector (384-d, unit norm, metadata), spot-check source text vs chunk, and a retrieval test with a quoted sentence. (See E3.5.)

**A9.** Reprocess reads the file from the document's recorded storage. If that is `local` on Render's ephemeral disk and a restart wiped it, the worker raises the permanent `FILE_MISSING` with diagnostics (backend, key) and the user is told to upload again. With `db` or `s3` storage the file survives the restart and the retry succeeds. The production config still uses `local`.

**A10.** Against: no measurements exist; changing the size changes every vector, requires a new index/re-ingest, and the benchmark gate must pass (`docs/rag-architecture.md §6`). For: smaller chunks might help precision on fact-lookup questions. Test: build a labeled question set (answerable, paraphrased, multi-sentence), index a copy of the corpus with 600/100 in a separate namespace/index, compare Hit@K/MRR/answer faithfulness, token cost and latency; accept only if retrieval does not regress and cost is acceptable.

**A11.** `UploadFile` (a `SpooledTemporaryFile`) keeps small uploads in memory and spills to disk; `bytes` loads all into memory. The backend uses `UploadFile` and hashes in 1 MiB steps. At `31bad7e` three places still read whole files (base64 for local storage, the worker's `read_bytes`, RAG Core's `file.read`); now the worker streams the stored object to a temp file and RAG Core spools the body to disk, so those are gone for the upload path, but parsing still holds the extracted text in memory.

**A12.** If the process crashes after writing some vectors but before recording what it may have written, cleanup cannot know which ids to delete, leaving orphan vectors. Recording `vector_extent` (committed) first guarantees that every possibly-written id (`0..extent-1`) can be removed by the failure path, a later retry, or a delete.

---

## 14. Completion criteria (self-check)

- [ ] You reproduced the §4.3 chunk demo yourself with the repo's `chunk_pages`.
- [ ] You can explain the 25 MB question using the three limits.
- [ ] You can state precisely what is and is not implemented (OCR off by default with measured, limited accuracy; semantic/heading chunking present but not proven better; PDF tables only with `PDF_TABLES`, benefit unproven for retrieval; production storage still `local`).
- [ ] You can explain the base64 workaround, why it was retired, and how `db`/`s3` storage fixes the root cause.
- [ ] You ran the runnable blocks; all printed `ok`.

## Sources (accessed 2026-10-23)

1. FastAPI, "Request Files" (`UploadFile`, `SpooledTemporaryFile`): https://fastapi.tiangolo.com/tutorial/request-files/
2. Pinecone, "Rerank results" (reranker token limits): https://docs.pinecone.io/guides/search/rerank-results
3. BAAI/bge-small-en-v1.5 model card (512 tokens): https://huggingface.co/BAAI/bge-small-en-v1.5
4. OWASP, File Upload Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html
5. RFC 4648 (Base64 encoding: 3 bytes → 4 characters), from standards knowledge, not re-fetched.
