# RAG architecture: frozen baseline

**Status: FROZEN on 2026-10-08.** The parameters below are the configuration that was verified end to end against the real stack: the running RAG Core, Pinecone `advancerag-bge384`, Pinecone rerank, Agent Core, the LLM service and FreeLLMAPI.

Do not change a frozen value as a side effect of other work. The guard test `rag-core/tests/test_frozen_contract.py` fails if a default changes. A deliberate change follows the procedure in [section 6](#6-changing-a-frozen-value).

## 1. Pipeline

```
Upload ─► Backend (storage + job) ─► RAG Core /ingestions (async; see "Ingestion lifecycle" below)
            parse ─► chunk ─► token-limit guard ─► embed (bge-small, passage) ─► Pinecone upsert + chunk rows (PostgreSQL)

Question ─► Agent Core: retrieval needed? (documents attached, not small talk)
            └─ follow-up? rewrite into a standalone query (LLM, fast model, temperature 0)
         ─► RAG Core /retrieve
            embed query (bge-small, query instruction)
            ├─ semantic: Pinecone, namespace user-<id>, filter document_id ∈ attached, top 20
            └─ keyword:  PostgreSQL full-text (GIN index), OR semantics, ts_rank_cd, top 20
            ─► RRF (k=60) ─► top 20 ─► cross-encoder rerank ─► top 6 ─► numbered context ≤ 12,000 characters
         ─► Agent Core: grounded prompt ([n] sources, cite, "not in the documents" rule, ignore instructions in sources)
         ─► LLM service ─► FreeLLMAPI (streamed) ─► citations extracted ([n], [n, m], 【n】; only numbers that were provided)
```

| Stage | Owner | Implementation |
|---|---|---|
| Parsing | RAG Core `parsing.py` | PDF via pypdf: per page; encrypted PDFs rejected. DOCX via python-docx: paragraphs and tables **in document order**; tables with a header row become `Header: value \| …` lines; merged cells are deduplicated. CSV via the `csv` module: delimiter sniffed (`, ; tab \|`); rows become `column: value \| …`; an all-numeric or duplicate first row is not used as a header. TXT and MD: decoded (utf-8-sig, then cp1252, then latin-1) and whitespace-normalised. |
| Chunking | `chunking.py` | Recursive splitting on separators (paragraph → line → sentence → clause → word), page-aware, never across pages. Overlap starts on a **line** boundary where possible (a table or CSV row is never cut in half), otherwise on a word boundary. A chunk can reach size + overlap. |
| Token guard | `chunking.fit_to_token_limit` | Any chunk longer than the model's token limit (512, read from the tokenizer) is split before embedding. Nothing is silently truncated. |
| Embeddings | `embeddings.py` | fastembed (ONNX, CPU). `passage_embed` for chunks, `query_embed` for queries. Unit-normalised. The output dimension is checked on every batch. |
| Vector store | `vectorstore.py` | Pinecone REST. Deterministic ids `<document_id>:<n>`. Metadata `{document_id, chunk_index, page}` only; chunk text never goes to Pinecone. Re-ingest overwrites and deletes stale ids. Delete removes every id up to the highest chunk count ever indexed. |
| Keyword | `pipeline._keyword` | PostgreSQL `to_tsvector('english')` with a GIN index, `plainto_tsquery` rewritten to OR, ranked by `ts_rank_cd`. Scoped to the user and the attached documents. |
| Fusion | `pipeline.rrf` | Reciprocal rank fusion, k = 60. |
| Rerank | `rerank.py` | Pinecone hosted `bge-reranker-v2-m3`. If it fails, the RRF order is used and recorded as `reranker: skipped`. **No score threshold** (see section 4). |
| Degradation | `pipeline.retrieve` | Vector store unavailable (a retryable error): keyword-only results, recorded as `retrieval_strategy: keyword_only_degraded`. Permanent errors are raised, not masked. |
| Context | `pipeline.retrieve` | `[n] filename (page p)` header and the chunk text, in rerank order, capped at 12,000 characters. |
| Citations | Agent Core `extract_citations` | Built only from source numbers that were actually provided, so they cannot be invented. |

### Ingestion lifecycle (orchestration only; changes nothing above)

Large documents take many minutes to index on CPU. The measured worst case is an 812-page,
5.6 MB PDF with 4,824 chunks: about 175 s to parse, about 885 s to embed and about 39 s to
upsert. No HTTP timeout bounds that work.

1. **Upload.** The backend stores the file, creates the document (`uploaded`) and queues an
   ingestion job. The request returns in well under a second.
2. **Start or join.** The worker asks RAG Core for the document's ingestion status.
   - If an attempt is already `running`, the worker joins it; if it already `succeeded`, it just
     records that.
   - Otherwise it posts the file to `POST /api/v1/rag/ingestions`, which returns 202 at once.
3. **Per-document lock.** `rag_ingestions` holds one row per document. Claiming an attempt is a
   compare-and-set on `attempt_id`, which holds across processes and replicas. A second request
   for a document whose attempt is alive joins it and never starts a second ingestion.
4. **Background run.** RAG Core indexes in a background task that heartbeats every 10 s.
   - Its stages are `parsing`, `embedding`, `writing`, `upserting`, `verifying`, `committing` and `done`.
   - `verifying`: before a document can become `ready`, its first, middle and last vectors, and the
     first vector of every 100-vector upsert request, must be read back from the vector store (at
     most 54 ids for 5,000 chunks, one Pinecone `GET /vectors/fetch` in the user's namespace;
     eventually consistent, so up to `INGEST_VERIFY_SECONDS`, 20 s). Otherwise the attempt fails
     with the retryable `VECTOR_INDEX_UNVERIFIED` and removes what it wrote. A concurrent delete
     stops the check at once.
   - CSV files are also stored as a table (`rag_tables`: columns + all rows, at most
     `TABLE_MAX_ROWS` 200,000 rows and `TABLE_MAX_COLUMNS` 100), in the same transaction as the
     chunks, replaced on re-ingest and deleted with the document (FK cascade). Chunking and
     embeddings are unchanged.
   - Embedding and upserts run in slices; each slice is a cancellation checkpoint.
   - The vector extent (every id that may be written) is committed **before** the first vector
     is written.
   - The commit re-checks, under a row lock, that the attempt still owns the document.
5. **Polling.** The worker polls `GET /api/v1/rag/ingestions/{id}` every 2 s.
   - Each poll renews the job's lease (`locked_at`), so a long but alive ingestion never looks
     crashed. A job that lost its lease stops without touching any state.
   - The document stays `processing` the whole time, then becomes `ready`.
6. **Failure.** The attempt removes this document's vectors (ids 0..extent-1 in the owner's
   namespace) and chunk rows, then records a sanitized code. The code is either retryable
   (e.g. `VECTOR_STORE_UNAVAILABLE`) or permanent (e.g. `PARSE_FAILED`, `NO_EXTRACTABLE_TEXT`,
   `DOCUMENT_TOO_LARGE`).
   - The backend retries only retryable codes, at most 3 attempts. The document ends `failed`
     with "… No indexed data was retained."
   - A RAG-only cleanup job is then queued as a backstop. It is skipped if the user has
     reprocessed the document meanwhile.
7. **Crash.** An attempt whose heartbeat is older than 120 s (and that isn't running in the
   answering process) reports `stale`. Once that is confirmed over 30 s, the worker clears it
   and retries; the next claim takes over, and the recorded extent covers any leftovers.
   - A worker crash is recovered by a stale-lease sweep that runs every 60 s; a lease counts as
     stale after 5 min. The recovered job joins the RAG attempt, which kept running.
   - A superseded attempt never deletes writes that belong to the newer attempt.
8. **Delete during ingestion.** `DELETE /api/v1/rag/documents/{id}` removes the attempt row and
   every vector up to the extent. The running attempt stops at its next checkpoint or commit
   check and discards what it wrote; the backend job ends `cancelled`.

The ingestion ceiling (`INGEST_MAX_SECONDS`, 1 h) counts active polling time only. If the host
sleeps, RAG Core is frozen too, so time asleep must not become a timeout.

### Exact table analysis (additive; changes nothing above)

`POST /api/v1/rag/tables/describe {user_id, document_ids}` returns the caller's stored tables
(columns with type, distinct count, missing count, examples or numeric range).
`POST /api/v1/rag/tables/query {user_id, document_id, queries[1..5]}` computes over **every row**
of one table the caller owns (404 otherwise). Queries are declarative: `count`, `distribution`,
`crosstab`, `stats`, `top_terms`, at most 5 filters (`eq ne contains not_contains gt gte lt lte
in`) and `limit` ≤ 50. There is no SQL and no code execution; an unknown column or wrong
operation returns 422 `TABLE_QUERY_INVALID`. Agent Core uses this for quantitative questions
(docs/agent-core-architecture.md §7).

### Ingestion speed (measured 2026-10-10; changes no stored value)

Where the time goes, from the production logs (Render free, 0.1 CPU) and local measurements:

| Stage | Share of one ingestion | Evidence |
|---|---|---|
| Embedding (CPU inference of the quantized bge-small ONNX model) | about 99% on Render free, 74-95% on a laptop | production: 74.3 s of about 75 s for 40 chunks; the 812-page worst case above: 885 of about 1,100 s |
| PDF parse (pypdf, single-threaded Python) | up to 16% for real PDFs | 812 pages: about 175 s |
| Pinecone upserts | 0.3-0.4 s per 100-vector request, 3-4% sequentially | production logs; 39 s for 4,824 chunks |
| Tokenizing for the length check, chunking, CSV table parse, DB writes, verification | under 1% | 0.3 ms per chunk; chunking 10-20 ms; table parse 15-54 ms |

What changed, and what each change was measured to do (same documents, same machine, vectors compared
afterwards):

- **Length-sorted embedding batches.** A batch is padded to its longest text, so similar lengths
  per batch waste less compute. 22-24% less embedding time. Each text's vector does not depend on
  its batch: **bit-identical** (max difference 0.0 over every variant and run), and results come back
  in the caller's order.
- **`EMBEDDING_BATCH_SIZE` default 32 -> 4.** On 12 cores, batch 4 is 26% faster than 32; on one
  thread it is up to 4x faster. Sorted batch sizes 2 and 4 are within noise of each other; 8 and 16
  are 6-26% slower. Render already used 4. Vectors identical.
- **Concurrent upserts** (`PINECONE_UPSERT_CONCURRENCY`, default 5 = one round trip per 500-vector
  slice, at most 8): the upsert stage of a 410-chunk PDF went from 1.96 s to 0.53 s.
- **Bounded retries** (`PINECONE_MAX_ATTEMPTS`, default 3, exponential backoff 0.5 s / 1 s, Retry-After
  honoured up to 5 s) for upsert, fetch and delete only, and only for network errors, 429 and 5xx.
  Searches are not retried. One transient Pinecone error used to fail the whole attempt and throw
  away all of its embedding work (up to an hour on Render free).
- **Stronger verification** before `ready`: one vector of every 100-vector upsert request is read
  back (at most 54 ids for 5,000 chunks, still one fetch call), so a request that was acknowledged
  but not stored cannot be missed.

End to end, 12-core laptop, simulated Pinecone latency from production, old code with its defaults
against new code with its defaults (fastest of two interleaved runs; the machine was shared, so
single runs vary by up to 3x): 1,000-row CSV (209 chunks) 18.7 -> 14.3 s (-24%), 30-page PDF
(124 chunks) 11.0 -> 6.8 s (-38%), 100-page PDF (410 chunks) 35.6 -> 22.4 s (-37%). With Render's
own settings (one thread, batch 4, slice 32) only sorting and the concurrent upserts apply:
1,000-row CSV 45.4 -> 42.4 s (-7%, one run each). Ids, metadata and vectors were identical to the
old code on the same bytes.

Not changed, and why:

- **Embedding is still CPU-bound.** On Render free (0.1 CPU) a document needs about 1-2 s per chunk,
  so the 5,000-chunk limit would take about two hours, more than the one-hour ingestion ceiling.
  No code change fixes that; it needs more CPU.
- **Overlapping embedding with upserting** would hide at most 3-4% and complicates the cleanup
  guarantees. **Batched token counting** was no faster on one core (0.123 s against 0.125 s).
  **Faster PDF extraction** (another library) would change the extracted text and so the chunks.
  **Resuming a failed attempt** without re-embedding is the largest remaining win for big documents
  but needs durable vectors per slice and a looser "a failed attempt leaves nothing" rule.

## 2. Frozen parameters

| Parameter | Value | Where |
|---|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5` | `EMBEDDING_MODEL` (RAG Core); baked into `rag-core/Dockerfile` |
| Dimension | **384** | Derived from the model, never set by hand |
| Pinecone index | `advancerag-bge384`: dense, cosine, serverless aws/us-east-1 | `PINECONE_INDEX` |
| Namespace | `user-<user_id>` (one per user) | `vectorstore.namespace_for` |
| Chunk size / overlap | 1200 / 200 characters | `CHUNK_SIZE`, `CHUNK_OVERLAP` |
| Max token length | 512 (bge tokenizer) | `LocalEmbeddings.max_tokens` |
| Candidates per retriever | 20 | `RETRIEVAL_CANDIDATES` |
| RRF k | 60 | `RRF_K` |
| Reranker | `pinecone` / `bge-reranker-v2-m3` | `RERANKER`, `RERANK_MODEL` |
| Final top k | 6 | `RETRIEVAL_TOP_K` (RAG Core and Agent Core) |
| Context cap | 12,000 characters | `MAX_CONTEXT_CHARS` |
| Max chunks per document | 5,000 | `MAX_CHUNKS_PER_DOCUMENT` |
| Query rewrite | on, for follow-ups (when there is history) | `QUERY_REWRITE_ENABLED` (Agent Core) |

The old `advancerag` index (1024-d, llama-text-embed-v2) is **not compatible** and must not be used.

## 3. Verified baseline (2026-10-08)

**Benchmark run against the real stack** (`python scripts/rag_benchmark.py`, run from the repo root with the stack running):
- Fixed corpus: an 8-page PDF handbook, two near-duplicate DOCX spec sheets (Aurora X1 and X2), TXT meeting notes, an MD runbook, a 240-row CSV price list, and a 27k-character distractor document.
- 69 chunks in total.

**Pinecone check:**
- 69 vectors in the benchmark namespace, equal to 69 chunk rows.
- Sampled vectors are 384-d and unit-norm.
- Metadata matches the chunk, and contains no text.
- Each stored vector equals bge-small(chunk text) (cosine > 0.999).

**Retrieval**: rank of the correct chunk over 20 questions, including paraphrases with no shared keywords, CSV rows deep in the file, and the X1/X2 distractor.

| Stage | hit@1 | hit@6 | MRR |
|---|---|---|---|
| Semantic (Pinecone) | 18/20 | 20/20 | 0.950 |
| Keyword (PostgreSQL full-text) | 15/20 | 20/20 | 0.837 |
| RRF | 16/20 | 20/20 | 0.900 |
| **Final (after rerank)** | **19/20** | **20/20** | **0.975** |

- The only miss is genuinely ambiguous: "What temperature range can *the sensor* operate in?" matches both the X1 and the X2. Both are in the top 6.
- Retrieve latency through the running service (Pinecone query plus rerank): p50 1.1 s, max 2.3 s.
- User isolation: another user asking with the same document ids gets 0 chunks.

**End to end** (Agent Core → RAG Core → Pinecone → LLM service → FreeLLMAPI):
- **7/7** answerable questions got the correct fact, cited to the correct file. This includes a follow-up that needed the query rewrite.
- **2/2** unanswerable questions were answered with "the documents do not contain this".

**Fixes that produced this baseline** (made in the core RAG audit):
- DOCX tables kept in document order;
- self-describing table and CSV rows;
- overlap that respects line boundaries;
- the token-limit guard;
- keyword fallback when the vector store is down;
- citation formats `【n】` and `[n, m]`.

## 4. Decisions on record

- **No reranker score threshold.** Measured: a correct paraphrased answer scored **0.007**, while an on-topic unanswerable question scored **0.38**, so every threshold drops real answers. "Not in the documents" is handled by the grounded prompt, which declined 2/2 unanswerable questions.
- **Hybrid search is kept, even though semantic search alone looks stronger on this corpus.** Keyword search finds exact codes such as SKUs and error codes, and it is the fallback when Pinecone is down.
- **Chunk text lives in PostgreSQL, not in Pinecone metadata.** This keeps Pinecone small and makes deletion complete.
- **One Pinecone namespace per user.** This is hard isolation. Check the namespace limit of your Pinecone plan before you have many users.

## 5. Known limits (accepted for now)

- OCR exists (`rag_core/ocr.py`, RapidOCR on ONNX Runtime, pypdfium2 rendering) but is **off by default**. With it off, scanned PDFs fail with `NO_EXTRACTABLE_TEXT`. Measured on labelled synthetic samples (docs/verification/OCR_ACCURACY.md): mean character error rate about 2% for English text across 11 conditions; accented Latin characters are often lost; Cyrillic, Greek and Korean are not read; a line can be dropped without any warning, and the engine's confidence does not predict errors.
- PDF layout follows pypdf's text order. Multi-column layouts may interleave.
- Section headings stay in the chunk text. With `CHUNKING_STRATEGY=structure` or `semantic` (not the default) the heading path is also stored as `section` and, if `EMBED_SECTION_CONTEXT`, prepended to the text that is embedded (the stored and returned text stays the bare chunk). The default `recursive` strategy is unchanged and is the one the frozen parameters and the baseline in section 3 describe.
- PDF tables are structured only when `PDF_TABLES` is `lines` (ruled tables) or `auto` (also borderless, heuristic); the default `off` keeps plain text. Tables on scanned pages are inferred from OCR positions and flagged. See rag_core/pdftables.py and docs/verification/RETRIEVAL_EVALUATION.md (no retrieval benefit shown, fidelity verified by tests).
- The model is English-only (`bge-small-en`).
- CSVs indexed before migration 0003 have no stored table until they are reprocessed; quantitative questions about them fall back to chunk retrieval.
- Documents indexed before 2026-10-08 keep their old chunks until they are reprocessed. As of the freeze there were none.

## 5a. Additions after the freeze (2026-10-23, uncommitted; none changes a stored vector, the model, the dimension or the metric)

- **Extraction**: `parsing.py` returns pages with a method (`native`/`ocr`), engine-reported confidence and an `ExtractionReport` (native/OCR/blank/unprocessed/failed pages, warnings) stored on `rag_documents.extraction` and returned by the status API. `textclean.py` (on by default, `TEXT_CLEANUP`) fixes ligatures, soft and line-end hyphens and repeated page headers/footers. Zip, pixel and OCR budgets bound hostile input.
- **Chunking strategies** behind `CHUNKING_STRATEGY` (`recursive` default | `structure` | `semantic`). The recorded comparison is in `docs/verification/RETRIEVAL_EVALUATION.md`; it does not justify changing the default.
- **Streaming**: uploads are spooled to a temp file and parsed from the path; the backend streams the stored object to RAG Core.
- **Integrity**: documents record the embedding model and dimension; retrieval refuses mismatches (`EMBEDDING_MODEL_MISMATCH`, 409), `/ready` reports them, the Pinecone dimension and metric are checked, and `POST /api/v1/rag/documents/{id}/reindex?user_id=` re-embeds from stored chunk rows (same deterministic vector ids, never deletes).
- **Observability**: ingestion/OCR/degraded-retrieval counters and an ingestion-duration histogram; W3C `traceparent` propagation (see `shared/advancerag_common/observability.py`).
- **Keyword ranking (behaviour change, not a stored value):** `KEYWORD_RANKING` now defaults to `idf`: the PostgreSQL OR candidates are re-scored with BM25 using the user's own chunk statistics. The original `ts_rank_cd` ignores term rarity; on evaluation dataset v1.2 it found none of the exact-identifier chunks in its top 20 and made the hybrid worse than dense search alone (Hit@1 0.737 -> 0.598). With the new default plus a cross-encoder, Hit@6 0.958 -> 0.992 and MRR 0.908 -> 0.939 on that dataset (all of the gain on identifier lookups; nothing got worse elsewhere). Roll back with `KEYWORD_RANKING=cover_density`. Evidence and limits: docs/verification/RETRIEVAL_EVALUATION.md. Cost: four database round trips per query instead of one. `RRF_KEYWORD_WEIGHT` (default 1.0) is available but was not needed.
- **Audit and repair:** `GET /api/v1/rag/documents/{id}/audit?user_id=` (read-only) compares chunk rows, chunk indexes and the vectors that can actually be read back, and the recorded embedding model; `POST .../reindex` re-embeds from stored chunks. Missing chunk rows cannot be rebuilt from vectors (Pinecone holds no text); the audit says to reprocess the original file.
- **Parser budgets:** `PDF_PARSE_TIMEOUT_SECONDS` (native extraction; checked between pages, so one pathological page can still run long), `PDF_TABLE_TIMEOUT_SECONDS`, and the OCR page/pixel/time limits.
- **Tables in PDFs:** `PDF_TABLES=off|lines|auto` (default `off`), implemented with pdfplumber; see section 5.
- Migrations: RAG Core `0004_extraction_metadata`; backend `0005_document_durability` (additive and backward compatible; rollback with `alembic downgrade`).

## 6. Changing a frozen value

1. Explain why the change is needed, with a failing example.
2. Change the value, and update `test_frozen_contract.py` and this document in the same change.
3. If the change affects embeddings or chunking, you need a new Pinecone index whose dimension matches the model, plus a full re-ingest. Never mix vectors from two models in one index.
4. Run `python scripts/rag_benchmark.py`. The baseline gate must pass, and the retrieval numbers must not regress.
5. Record the new baseline in section 3.
