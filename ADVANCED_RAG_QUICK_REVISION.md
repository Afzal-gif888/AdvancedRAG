# AdvancedRAG Quick Revision

Part of the [master course](ADVANCED_RAG_2026_MASTER_COURSE.md). Last-minute notes. **Updated after the implementation pass (2026-10-23, uncommitted working tree on `31bad7e`); evidence in [FEATURE_MATRIX](docs/verification/FEATURE_MATRIX.md); all new evidence is hermetic, none is production.** Evidence tags: **[code]** read in source, **[doc]** documented in the repo and not reproduced by me, **[tests]** a repo test passed in my run (2026-10-23, commit `31bad7e`), **[missing]**, **[unknown]**. If a point feels shaky, open the linked module.

## 1. Architecture ([Module 1](docs/course/MODULE_1_SYSTEM_ARCHITECTURE.md))

```text
Browser (React, Vercel) ─HTTPS/SSE─► Backend (FastAPI): auth, chats, documents, shares, job queue + in-process worker
Backend ─Bearer INTERNAL_SERVICE_TOKEN─► Agent Core (stateless): classify, rewrite, grade, prompt, verify, cite
Backend worker ─► RAG Core (ingest)        Agent Core ─► RAG Core (retrieve)        Agent Core ─► LLM service ─► FreeLLMAPI
RAG Core: parse, chunk, local embeddings (bge-small, 384-d), Pinecone (vectors), Postgres (chunk text + FTS), RRF, rerank
Stores: Neon×2 (app DB; RAG DB), Pinecone (vectors only, namespace user-<id>), file storage (local or S3), Redis (rate limits)
```

- Four backend services because of separate secrets/failure domains; cost: hops and config. A modular monolith is a valid alternative.
- Free-tier machinery (wake relay, keep-awake, long timeouts) is a hosting workaround, not a design goal. The base64 job payload was replaced by durable per-document storage (`local`/`s3`/`db`); production config still says `local`.
- Status of deployment is **[unknown]** to me. Say "configured for", not "running on".

## 2. Document ingestion flow ([Module 1 §5.1](docs/course/MODULE_1_SYSTEM_ARCHITECTURE.md), [Module 3](docs/course/MODULE_3_CHUNKING_AND_DOCUMENT_PARSING.md))

1. Upload: allowlist (`pdf,txt,md,csv,docx`), magic bytes, ≤ 25 MiB, quotas, sanitised name, server key → `201 uploaded`.
2. Document row + job row in **one commit**; blob deleted if commit fails.
3. Worker: claim (`SKIP LOCKED` + CAS) → `processing` → join or start RAG attempt (`202`) → poll + heartbeat.
4. RAG Core: parse (pypdf / python-docx / csv / text) → recursive page-aware chunks 1200/200 → token guard (512) → cap 5,000 chunks → embed → **record extent** → chunk rows + vector upsert → read-back verify → ownership re-check → commit → `ready`.
5. Failure: cleanup of partial writes; transient → retry (30 s, 60 s, ... ≤ 15 min, ≤ 3); permanent → `failed` with a safe message.
6. Delete: soft delete → job removes vectors, chunks, file → hard delete; reconciler re-drives.

## 3. Question-answering flow ([Module 5](docs/course/MODULE_5_LLM_BEHAVIOR_AND_GENERATION.md))

`save msgs` → `classify` (small talk? named file? CSV stats?) → `rewrite follow-up` → `retrieve` → `grade evidence` → `(corrected query ≤1)` → `generate (stream, T=0.3)` → `verify` → `strip invalid [n]` → `citations` → `save`.
Typical **4 LLM calls**, worst **8**. Abstain (insufficient evidence on a document question) skips generation: fixed sentence *"I couldn't find sufficient evidence for this answer in your Knowledge Base."*

## 4. Key configuration (frozen baseline) [code]

| Item | Value |
|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5`, 384-d, unit-normalised, local ONNX, 512-token input |
| Pinecone index | `advancerag-bge384`, cosine (metric is **not** checked by code), namespace `user-<id>` |
| Metadata in Pinecone | `document_id`, `chunk_index`, `page` only (no text) |
| Chunking | 1200 chars, overlap 200, a chunk can reach size+overlap; never crosses pages |
| Candidates / RRF / final | 20 + 20 → RRF k=60 → rerank → **6**; context ≤ **12,000** chars |
| Reranker | Pinecone `bge-reranker-v2-m3`; failure → RRF order |
| Keyword search | PostgreSQL FTS `ts_rank_cd`, OR query, GIN index (**not BM25**) |
| pgvector | **not used** |
| Limits | 25 MiB upload; 5,000 chunks/doc; 500 docs & 2 GB per user; ingestion ceiling 1 h active |
| Auth | Argon2id; JWT 15 min + session row; refresh 7 d rotating w/ reuse detection |
| LLM settings | answers T=0.3; checks T=0; retries ≤ 2 (1→8 s); `AGENT_FAST_MODEL` empty = default route |

## 5. Embedding and retrieval fundamentals ([Module 2](docs/course/MODULE_2_EMBEDDINGS_AND_RETRIEVAL.md))

- cos = a·b/(‖a‖‖b‖). Unit vectors: dot = cos; ‖a−b‖² = 2−2cos → same ranking.
- Same model for query and passage (different entry points). Same dimension ≠ same model.
- Three dimension checks (config, batch, index) plus a model-name check on read: mismatched documents are refused (`EMBEDDING_MODEL_MISMATCH`) and repaired by reindex from stored chunks **[tests]**.
- Dense for paraphrase; keyword for exact codes; fuse by **ranks** (RRF), then cross-encoder rerank.
- RRF = Σ 1/(60+rank). Example: A,B,C,D + C,A,E,B → A, C, B, E, D.
- bge scores cluster ~0.6–1: use for ranking, not thresholds. Project measured 0.007 (correct) vs 0.38 (unanswerable) **[doc]** → no threshold.

## 6. Chunking trade-offs ([Module 3](docs/course/MODULE_3_CHUNKING_AND_DOCUMENT_PARSING.md))

- Small chunks: precise, context-poor, more vectors. Large: blurry, rich. Overlap saves boundary facts, costs ~17% more vectors.
- Default: recursive + page-aware + token guard. Also implemented: `structure` (heading-aware) and `semantic` strategies (not default; no proven win), optional OCR for scans/images (off by default; CER ~2% English, some scripts unreadable), PDF tables (`PDF_TABLES`, default off), text clean-up, upload dedup, versions (replace/restore/purge). **Not** done: chunk-level dedup.
- Binding limit is chunk count (≈ 5 MB of text) before 25 MB of bytes.
- Base64 = ×4/3 (25 MiB → 33.33 MiB): the retired workaround. Now jobs reference stored objects; sha256 checked; `FILE_MISSING` (permanent) vs `STORAGE_UNAVAILABLE` (retryable) **[tests]**.
- Partial extraction is reported (`ExtractionReport`: native/OCR/failed/unprocessed pages) and shown in the UI **[tests]**.
- Verify an ingest: `rag_documents` row, chunk count/contiguity, ingestion status, Pinecone namespace count, sampled vector (384-d, unit norm, metadata), quoted-sentence query.

## 7. Evaluation formulas ([Module 4](docs/course/MODULE_4_EVALUATION_AND_BENCHMARKING.md))

| Metric | Formula |
|---|---|
| Hit@K | ≥ 1 relevant in top K |
| Precision@K | relevant in top K / K |
| Recall@K | relevant in top K / all relevant |
| MRR | mean 1/rank of first relevant (0 if none) |
| nDCG@K | DCG/IDCG; DCG = Σ gain_i / log₂(i+1) |
| Wilson 95% (k/n) | centre (p+z²/2n)/(1+z²/n), half-width z√(p(1−p)/n+z²/4n²)/(1+z²/n) |
| Percentile (nearest rank) | element ⌈p/100·n⌉ of sorted list |

Worked: relevant {a,b,c}, results [a,x,b,y,z]: P@3 = R@3 = 2/3; P@5 = 0.4, R@5 = 2/3. nDCG@4 of [3,2,0,1] = 0.9854. Wilson 19/20 = [0.76, 0.99]. Unanswerables: report **false-answer rate** and **over-refusal rate**.

**Historical benchmark ([doc], not reproduced):** 69 vectors = 69 chunks; Hit@1 19/20; Hit@6 20/20; MRR 0.975 (arithmetic consistent: one question at rank 2); retrieval p50 1.1 s, **max 2.3 s (not 2.5)**; 20 questions, author-written, dev=test, one run. The script's refusal regex does not match the current abstain sentence **[ran]**.
**Tests I ran:** rag-core 87 passed/4 skipped, agent-core 199, llm 92, backend 218 passed/4 skipped, contract 52. Skips = PostgreSQL-only concurrency/FTS tests and Redis/S3 tests.

## 8. LLM behaviour ([Module 5](docs/course/MODULE_5_LLM_BEHAVIOR_AND_GENERATION.md))

- Attention = softmax(QKᵀ/√d)V. Temperature divides logits: [2,1,0] → T=1 [0.665,0.245,0.090]; T=0.3 [0.964,0.034,0.001].
- Prompt: system = rules; user = fenced `<evidence><source id= document= page=>` + question; delimiter look-alikes neutralised.
- Hallucination despite good retrieval: ambiguity/near-duplicates, partial/conflicting evidence, parametric override, position effects, injected text.
- Defences: grade → abstain, grounded prompt, strip invalid `[n]`, verify/revise, numbers check for CSV, no tools, UI image blocking.
- Injection test uses a mock LLM → proves prompt structure only **[unknown for real model]**.
- Failures: retries only in LLM service; read timeout not retried; stream retries only before first token; partial output kept; five public error codes.
- No token counting; token/cost accounting partial (stream usage off by default).

## 9. Security risks ([Module 6](docs/course/MODULE_6_SECURITY_AND_RELIABILITY.md))

Implemented & tested: allowlist+magic bytes, sanitised names, server keys, 404-on-foreign, 4-layer isolation, JWT+session, rotation/reuse, rate limits (Redis, fail-closed), body limits, CORS, log redaction, fail-fast prod config.
**Gaps:** shared internal token trusts any `user_id`; no parser sandbox/uncompressed-size cap; unpinned deps; no concurrent-stream cap; payload retention ≥ 30 d incl. after delete; email enumeration at register; no backup procedure; injection untested on real model; PG-only tests skipped in my run.
Never say "secure". Say "controls, tests, known gaps".

## 10. Scaling ([Module 7](docs/course/MODULE_7_PERFORMANCE_AND_SCALING.md))

- Ingestion bottleneck = embedding CPU (≈ 99% on Render free **[doc]**). Retrieval p50 1.1 s **[doc]**; end-to-end **[unknown]**.
- Little's Law: L = λW (100 in flight at 20 s → 5/s). Utilisation ≤ ~70%.
- 25 MB: three limits (bytes, 5,000 chunks, 1 h). 100 concurrent turns: suspects = LLM quotas, RAG CPU, DB pool; add stream cap + global limiter.
- Small app needs: persistent storage, CPU, limits, backups, measurements. Not Kubernetes.

## 11. Frequently asked interview questions (see the [question bank](ADVANCED_RAG_INTERVIEW_QUESTION_BANK_2026.md))

1. Walk through a question end to end. 2. How is ingestion reliable? 3. Why hybrid? 4. Is it BM25 / pgvector? (No/No.) 5. How do you stop hallucination? 6. How is data isolated? 7. What are the weaknesses? 8. Audit your benchmark. 9. Handle a 25 MB file. 10. Scale to 100 users.

## 12. Common technical mistakes

Calling Postgres FTS "BM25" · claiming pgvector/semantic chunking/OCR · treating `ready` as "all pages searchable" · quoting 2.5 s · calling 19/20 "95% accuracy" · using a score threshold for abstention · same-dimension model swap · averages instead of percentiles · unbounded `gather` · `WHERE` on the right table of a `LEFT JOIN` · trusting Content-Type · "temperature 0 stops hallucination" · saying "secure".

## 13. One-page final revision sheet

**30-second pitch.** AdvancedRAG lets you chat with your documents. Uploads are indexed in the background; questions run hybrid search (semantic + keyword), re-rank, and an LLM answers only from the retrieved passages with citations, or says it can't find evidence.

**Numbers to know.** 384-d · 1200/200 · top 20+20 → 6 · RRF k=60 · 12,000 chars · 5,000 chunks · 25 MiB · 15 min JWT / 7 d refresh · 4 typical, 8 worst LLM calls · benchmark 19/20, 20/20, 0.975, p50 1.1 s, max 2.3 s (20 q, 69 chunks, documented).

**Say, don't claim.** "The code does X (file). The repo's test for it passed in my run / I haven't verified Y. The known gap is Z and here's how I'd test and fix it."

**Measured (hermetic, one laptop, synthetic data):** retrieval on 263 questions: the old Postgres full-text leg made the hybrid *worse than dense alone* (Hit@1 0.737 -> 0.598) and found **0 of 12 exact-identifier chunks**; BM25-weighted candidates (`KEYWORD_RANKING=idf`, now the default) + cross-encoder: Hit@6 0.958 -> 0.992, MRR 0.908 -> 0.939, all of it from identifier lookups, nothing worse elsewhere. OCR CER 0.019 on English samples (Cyrillic/Greek/Korean unreadable; a line can vanish silently; confidence r=0.11 with accuracy). Embedding 34.8 chunks/s; text >= 5 MB refused at 5,000 chunks. Bounded load tests (6-12 docs ingesting, 8-24 chat users, mixed, OCR): 0 errors after fixing a Windows path race; ingest concurrency 2 gave no gain; OCR peaked at 526 MB RSS. `pip-audit` (Pillow/pypdf/certifi upgraded), fastembed `pillow<13` conflict resolved with `--no-deps`, versions pinned in `constraints/`. **Not** measured: Pinecone/Neon latency, real model latency, soak, anything in production.

**Top fixes if asked "what next?"** (1) switch production to `db`/`s3` storage (operator guide exists); (2) per-service credentials and signed user context; (3) run the parser in a sandboxed subprocess (time budgets exist but are cooperative); (4) a global LLM limiter (the per-user stream cap is now shared across replicas via Redis); (5) repeat the retrieval evaluation on real documents and the production reranker before trusting `idf`; (6) soak and Pinecone/Neon latency tests, SLOs; (7) chunk-level dedup; (8) backups with a tested restore on the managed database; (9) a CI job that rebuilds constraints and runs `pip-audit` on the built images; (10) adversarial injection suite on the real model.

**Formulas.** cos=a·b/‖a‖‖b‖ · RRF=Σ1/(60+r) · MRR=mean(1/r) · nDCG=DCG/IDCG · Little L=λW · base64=4⌈n/3⌉ · softmax(z/T) · attention=softmax(QKᵀ/√d)V.
