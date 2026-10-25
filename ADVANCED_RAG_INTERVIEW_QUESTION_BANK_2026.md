# AdvancedRAG Interview Question Bank (2026)

Part of the [master course](ADVANCED_RAG_2026_MASTER_COURSE.md). 172 distinct questions in 17 categories. Questions Q1–Q160 were written against commit `31bad7e`; where the implementation pass (2026-10-23, uncommitted) changed the facts, the *AdvancedRAG* line says so, and category 17 covers the new work. Evidence for the new work: [docs/verification/FEATURE_MATRIX.md](docs/verification/FEATURE_MATRIX.md).

**Read this first**

- These are **practice questions I wrote** for AI-engineer interviews, based on common topics and on this repository. They are **not** questions from any company's private interviews, and I make no claim that any will be asked.
- Each entry has: **Difficulty**, **Testing** (what the interviewer is checking), **Answer**, **AdvancedRAG** (how it connects to this project), **Follow-up**, **Avoid** (common mistakes).
- Project statements are tagged by evidence level as in the modules: *code-read*, *documented* (a repo doc says it; not reproduced), *missing*, *unknown*. Where I write "documented", do not present it as something you measured.
- Benchmark numbers are **historical documented results** (20 questions, 69 chunks, one run): Hit@1 19/20, Hit@6 20/20, MRR 0.975, retrieval p50 1.1 s, max 2.3 s. Not reproduced by me, not production performance.
- Do not claim accomplishments that are not yours. The behavioural section gives structure and honest facts about the project; replace with your real experience.
- Deeper treatment of each topic is in the modules: [1 Architecture](docs/course/MODULE_1_SYSTEM_ARCHITECTURE.md), [2 Embeddings and retrieval](docs/course/MODULE_2_EMBEDDINGS_AND_RETRIEVAL.md), [3 Chunking and parsing](docs/course/MODULE_3_CHUNKING_AND_DOCUMENT_PARSING.md), [4 Evaluation](docs/course/MODULE_4_EVALUATION_AND_BENCHMARKING.md), [5 LLM behaviour](docs/course/MODULE_5_LLM_BEHAVIOR_AND_GENERATION.md), [6 Security](docs/course/MODULE_6_SECURITY_AND_RELIABILITY.md), [7 Performance](docs/course/MODULE_7_PERFORMANCE_AND_SCALING.md), [8 Coding](docs/course/MODULE_8_CODING_AND_PRACTICAL_INTERVIEWS.md).

## Index

| # | Category | Questions |
|---|---|---|
| 1 | [System architecture](#1-system-architecture) | Q1–Q10 |
| 2 | [RAG fundamentals](#2-rag-fundamentals) | Q11–Q20 |
| 3 | [Embeddings and vector databases](#3-embeddings-and-vector-databases) | Q21–Q30 |
| 4 | [Chunking and parsing](#4-chunking-and-parsing) | Q31–Q40 |
| 5 | [Information retrieval](#5-information-retrieval) | Q41–Q50 |
| 6 | [LLMs and transformers](#6-llms-and-transformers) | Q51–Q60 |
| 7 | [Evaluation](#7-evaluation) | Q61–Q70 |
| 8 | [Security](#8-security) | Q71–Q80 |
| 9 | [Performance and scalability](#9-performance-and-scalability) | Q81–Q90 |
| 10 | [Python](#10-python) | Q91–Q100 |
| 11 | [SQL](#11-sql) | Q101–Q110 |
| 12 | [FastAPI](#12-fastapi) | Q111–Q120 |
| 13 | [Testing and debugging](#13-testing-and-debugging) | Q121–Q130 |
| 14 | [System design](#14-system-design) | Q131–Q140 |
| 15 | [Practical incidents](#15-practical-incidents) | Q141–Q150 |
| 16 | [Behavioural](#16-behavioural) | Q151–Q160 |
| 17 | [Implementation pass: what was built and verified](#17-implementation-pass-what-was-built-and-verified) | Q161–Q172 |

---

## 1. System architecture

**Q1 · Basic.** What does AdvancedRAG do, and what problem does it solve?
- *Testing:* can you explain the product in plain language.
- *Answer:* Users upload PDF/DOCX/TXT/MD/CSV files and chat with them. An LLM answers using passages retrieved from those files and cites file and page. It solves "LLMs don't know my private documents and may invent answers" by retrieving evidence first and refusing when evidence is missing.
- *AdvancedRAG:* abstains with a fixed sentence when evidence is insufficient (`prompts.ABSTAIN_MESSAGE`); OCR exists but is off by default, so scans still fail unless it is enabled.
- *Follow-up:* What would a plain ChatGPT-style chat get wrong here?
- *Avoid:* claiming it "understands" documents or never hallucinates.

**Q2 · Basic.** Name the services and each one's responsibility.
- *Testing:* architecture recall.
- *Answer:* Frontend (React on Vercel); Backend (auth, data, jobs, SSE, public API); Agent Core (classify, rewrite, grade, prompt, verify, cite; stateless); RAG Core (parse, chunk, local embeddings, Pinecone, Postgres full-text, RRF, rerank); LLM service (provider access, retries, streaming).
- *AdvancedRAG:* secrets are split: only the LLM service has the gateway key; only RAG Core has the Pinecone key.
- *Follow-up:* Which two services call RAG Core and for what?
- *Avoid:* forgetting that the backend worker calls RAG Core directly for ingestion.

**Q3 · Intermediate.** Walk through what happens from clicking "send" to seeing the answer.
- *Testing:* end-to-end tracing.
- *Answer:* SSE request → backend saves the user message and a pending assistant message → Agent Core classifies, optionally rewrites the query → RAG Core embeds the query, runs Pinecone and Postgres FTS, fuses with RRF, reranks, returns ≤ 6 chunks → Agent Core grades evidence, builds a fenced prompt, streams from the LLM service, verifies, strips invalid citations → backend persists the final message.
- *AdvancedRAG:* code path in Module 1 §5.2.
- *Follow-up:* Which steps use an LLM?
- *Avoid:* saying "the LLM searches the documents".

**Q4 · Intermediate.** Which steps are synchronous and which asynchronous, and why?
- *Testing:* sync/async reasoning.
- *Answer:* Upload returns 201 immediately; indexing is asynchronous because it takes seconds to hours. Chat is streamed so users see tokens early. Deletes are soft first, then a job removes vectors/files.
- *AdvancedRAG:* RAG Core returns 202 and runs a background task; the worker polls and heartbeats.
- *Follow-up:* Why poll instead of callback?
- *Avoid:* confusing `async/await` code with asynchronous business flow.

**Q5 · Intermediate.** Why two PostgreSQL databases instead of one?
- *Testing:* ownership boundaries.
- *Answer:* Separate lifecycle and ownership: the backend owns users/chats/jobs; RAG Core owns chunks, FTS index and ingestion locks. Cost: no cross-database transaction, so consistency relies on idempotent jobs, verification and reconcilers.
- *AdvancedRAG:* linked only by `document_id`/`user_id` convention.
- *Follow-up:* What breaks if one write succeeds and the other fails?
- *Avoid:* claiming foreign keys across the two.

**Q6 · Advanced.** Defend or attack "four services was the right decision".
- *Testing:* trade-off maturity.
- *Answer:* For: separate secrets and failure domains, swap-ability of providers, independent deploys. Against: one developer, network hops, more config, public URLs on free Render, much wake/timeout code created by free hosting; a modular monolith with the same package boundaries is a legitimate alternative.
- *AdvancedRAG:* the single-VM Compose deployment shows the same code can run co-located.
- *Follow-up:* What would you merge first?
- *Avoid:* "microservices are better".

**Q7 · Advanced.** Explain the ingestion reliability design.
- *Testing:* distributed-systems thinking.
- *Answer:* Document+job in one commit; atomic claim (`SKIP LOCKED` + CAS); lease/heartbeat; per-document attempt lock; deterministic ids; extent recorded before writes; read-back verification; commit-time ownership check; cleanup on failure; bounded retries.
- *AdvancedRAG:* `job_service.py`, `ingestion.py`, `pipeline.ingest`.
- *Follow-up:* What if the worker dies mid-ingestion?
- *Avoid:* describing only retries.

**Q8 · Advanced.** How does the system degrade when dependencies fail?
- *Testing:* graceful degradation.
- *Answer:* Pinecone down → keyword-only; reranker down → RRF order; grader/verifier fail → fall back to the grounded prompt and `unverified`; RAG Core down → no answer rather than invention; LLM failure → partial text saved, message failed and retryable.
- *AdvancedRAG:* `pipeline.retrieve`, `grading.py`, `agent_service.py`; tested with mocks.
- *Follow-up:* Which degradation lowers answer quality silently?
- *Avoid:* saying "it falls back to the LLM's own knowledge" (it does not for document questions).

**Q9 · Advanced.** Why is Agent Core stateless, and what does it cost?
- *Testing:* state placement.
- *Answer:* Stateless services scale horizontally and restart safely; the backend sends history and ready documents each turn. Cost: repeated payloads, and the backend must assemble context.
- *AdvancedRAG:* `message_service._build_agent_request`.
- *Follow-up:* Where would you cache?
- *Avoid:* saying it has no data at all (it receives user data per request).

**Q10 · Advanced.** What parts of the architecture exist only because of free-tier hosting?
- *Testing:* separating domain from workaround.
- *Answer:* Wake helper and Vercel `/wake` relay, keep-awake loop, `*_WAKE_SECONDS`, long nested timeouts, ephemeral-disk workarounds (the base64 job payload was one; it has been replaced by durable `db`/`s3` storage in code, while production config still says `local`), tiny batch/thread settings, services calling public URLs.
- *AdvancedRAG:* `render.yaml`, `wake.py`, `vercel.json`, `document_service.py`.
- *Follow-up:* What do you delete on paid always-on hosting?
- *Avoid:* presenting them as design goals.

---

## 2. RAG fundamentals

**Q11 · Basic.** What is RAG?
- *Testing:* definition.
- *Answer:* Retrieval-Augmented Generation: retrieve relevant passages for a question and give them to an LLM to write a grounded answer. Two phases: indexing (parse, chunk, embed, store) and answering (embed query, retrieve, prompt, generate, cite).
- *AdvancedRAG:* both phases are real code paths.
- *Follow-up:* Why not just fine-tune?
- *Avoid:* "RAG trains the model".

**Q12 · Basic.** Why not paste the whole document into the prompt?
- *Testing:* limits and cost.
- *Answer:* Context limits, cost and latency scale with tokens, accuracy drops on long inputs, and multi-document knowledge bases don't fit.
- *AdvancedRAG:* evidence is capped at 12,000 characters (top 6 chunks).
- *Follow-up:* When is long-context better?
- *Avoid:* claiming long context makes RAG obsolete.

**Q13 · Intermediate.** RAG vs fine-tuning vs long-context prompting.
- *Testing:* choosing tools.
- *Answer:* RAG for private, changing data with citations and per-user access control; fine-tuning for style/format/behaviour; long context for small corpora. They combine.
- *AdvancedRAG:* per-user isolation is a main reason for RAG.
- *Follow-up:* Can fine-tuning enforce tenant isolation?
- *Avoid:* "fine-tuning teaches facts reliably".

**Q14 · Intermediate.** Why can RAG still hallucinate?
- *Testing:* failure analysis.
- *Answer:* Retrieval miss, ambiguous/near-duplicate sources, partial evidence, conflicting sources, parametric override, long-context position effects, injected text, citation without support.
- *AdvancedRAG:* grading, grounded prompt, verifier, invalid-marker stripping; benchmark's single miss was an ambiguous X1/X2 question (documented).
- *Follow-up:* Which defence catches a fluent unsupported sentence?
- *Avoid:* "low temperature fixes it".

**Q15 · Intermediate.** What is "grounded generation"?
- *Testing:* concept + implementation.
- *Answer:* Instructing the model to answer only from supplied evidence and cite it, abstain otherwise, and treat evidence as data.
- *AdvancedRAG:* `GROUNDING_RULES` in `prompts.py`.
- *Follow-up:* How do you test groundedness?
- *Avoid:* assuming the prompt guarantees compliance.

**Q16 · Intermediate.** What is abstention and why is it hard?
- *Testing:* product trade-offs.
- *Answer:* Declining when evidence is insufficient. Hard because over-refusal and false answers trade off; scores don't separate answerable from unanswerable.
- *AdvancedRAG:* no score threshold (a correct answer scored 0.007, an unanswerable 0.38, documented); an LLM grader decides, and the abstain answer skips generation.
- *Follow-up:* How do you measure both error types?
- *Avoid:* using a cosine threshold.

**Q17 · Advanced.** What is self-correcting (agentic) RAG, as implemented here?
- *Testing:* depth beyond naive RAG.
- *Answer:* Classify the request, rewrite follow-ups, retrieve, grade sufficiency, retry with a corrected query (bounded), generate, then verify claims against evidence and revise or abstain.
- *AdvancedRAG:* up to 8 LLM calls per turn (typically 4); every check has a fallback.
- *Follow-up:* What does it cost in latency and money?
- *Avoid:* implying unbounded loops.

**Q18 · Advanced.** Why is the retrieval scope the whole Knowledge Base, not the conversation's attachments?
- *Testing:* product decision reading.
- *Answer:* The product contract: nothing has to be attached; every ready document is searchable in every chat; a named file narrows scope.
- *AdvancedRAG:* `document_ids=[]`, `library_documents` in `message_service._build_agent_request`; `classification.py`.
- *Follow-up:* Risks of searching everything?
- *Avoid:* assuming older docs describe current behaviour.

**Q19 · Advanced.** How do you handle follow-up questions ("and its price?")?
- *Testing:* conversational retrieval.
- *Answer:* Rewrite to a standalone query using history, guard against entity drift, fall back to attaching the subject; reuse earlier answers for pure reformatting requests.
- *AdvancedRAG:* `query_resolution.py` (`rejection_reason`, `_fallback`), `followup.py`.
- *Follow-up:* What if the rewrite introduces a wrong entity?
- *Avoid:* retrieving with the raw pronoun query.

**Q20 · Advanced.** Where is exactness needed, and how does the project handle CSV analytics?
- *Testing:* knowing RAG's limit.
- *Answer:* Counting/aggregation over a table can't come from 6 retrieved snippets. The project stores the CSV as a table and executes declarative, validated operations; the LLM only plans and explains.
- *AdvancedRAG:* `tables.py`, `structured.py`; numbers in the answer must appear in computed results.
- *Follow-up:* Why not let the LLM write SQL?
- *Avoid:* "RAG can answer any aggregate".

---

## 3. Embeddings and vector databases

**Q21 · Basic.** What is an embedding?
- *Testing:* fundamentals.
- *Answer:* A fixed-length vector produced by a model so that similar meanings are close.
- *AdvancedRAG:* 384-d `BAAI/bge-small-en-v1.5`, local ONNX via fastembed.
- *Follow-up:* Why 384 numbers?
- *Avoid:* "each dimension means something".

**Q22 · Basic.** Cosine vs dot product vs Euclidean distance.
- *Testing:* math.
- *Answer:* Cosine = angle; dot product = angle × lengths; Euclidean = straight-line distance. For unit vectors they rank identically (‖a−b‖² = 2 − 2cos).
- *AdvancedRAG:* all vectors are L2-normalised in `_normalize`.
- *Follow-up:* Compute for [3,4] and [4,3].
- *Avoid:* mixing normalised and unnormalised vectors.

**Q23 · Intermediate.** Why must query and document embeddings use the same model?
- *Testing:* embedding spaces.
- *Answer:* Each model defines its own space; cross-model distances are meaningless.
- *AdvancedRAG:* one provider for both; query vs passage entry points.
- *Follow-up:* What if the dimension is the same but the model differs?
- *Avoid:* thinking dimension equality is enough.

**Q24 · Intermediate.** How does the project guard against dimension mismatch, and what can't it catch?
- *Testing:* defensive design.
- *Answer:* Three checks (config vs model catalog, per-batch output length, Pinecone index dimension at first use and in `/ready`). It can't catch a same-dimension model swap; the stored per-document `embedding_model` isn't compared on read.
- *AdvancedRAG:* `embeddings.py::LocalEmbeddings.__init__`, `vectorstore.py::PineconeStore._client`.
- *Follow-up:* Write the guard.
- *Avoid:* saying "it's fully protected".

**Q25 · Intermediate.** What do Pinecone namespaces and metadata filters do here?
- *Testing:* isolation and filtering.
- *Answer:* One namespace per user (hard partition) plus a `document_id $in` filter to scope documents; metadata holds only document id, chunk index, page.
- *AdvancedRAG:* `namespace_for`, `PineconeStore.query`.
- *Follow-up:* What's the limit on namespaces?
- *Avoid:* stating a number; the plan limit is unknown from the repo.

**Q26 · Intermediate.** Why is chunk text stored in PostgreSQL rather than in Pinecone metadata?
- *Testing:* data placement.
- *Answer:* Smaller index, simpler complete deletion, FTS on the same rows, avoids metadata limits; cost: extra lookup and two stores to keep consistent.
- *AdvancedRAG:* `rag_chunks.text` + GIN index; query uses `includeMetadata: false`.
- *Follow-up:* What if the RAG DB is lost but Pinecone survives?
- *Avoid:* assuming vectors can reconstruct text.

**Q27 · Advanced.** Pinecone vs pgvector for this system.
- *Testing:* technology choice.
- *Answer:* pgvector keeps vectors with chunks (transactional, SQL filters, one fewer vendor) but needs memory/index tuning and fits Neon's small free storage poorly; Pinecone is managed ANN with namespaces and hosted reranking. Choose by scale, cost, ops skill.
- *AdvancedRAG:* pgvector is **not used**; there's no vector column in migrations. The alternative is a design discussion.
- *Follow-up:* Max dimensions for a pgvector index?
- *Avoid:* claiming the project uses pgvector.

**Q28 · Advanced.** Explain HNSW in two minutes.
- *Testing:* ANN knowledge.
- *Answer:* Layered proximity graph; start at sparse top layers, greedily move toward the query, descend, refine at the base layer; parameters trade recall, memory, build time.
- *AdvancedRAG:* Pinecone hides it; the dev-only local store is brute force.
- *Follow-up:* What is recall in ANN?
- *Avoid:* saying ANN is exact.

**Q29 · Advanced.** How would you migrate to a new embedding model without downtime?
- *Testing:* operations.
- *Answer:* New index with the new dimension/metric; backfill by re-embedding stored chunk text; shadow-evaluate with the benchmark; switch reads; keep the old index for rollback; delete later.
- *AdvancedRAG:* documented procedure: new index + full re-ingest; frozen values guarded by `test_frozen_contract.py`.
- *Follow-up:* What if originals are gone?
- *Avoid:* mixing models in one index.

**Q30 · Advanced.** Estimate vector storage for 1 M chunks at 384-d float32.
- *Testing:* arithmetic.
- *Answer:* 384 × 4 = 1,536 B/vector → ≈ 1.5 GB raw, plus index overhead and metadata.
- *AdvancedRAG:* a document is capped at 5,000 chunks ≈ 7.7 MB raw.
- *Follow-up:* How does quantisation help?
- *Avoid:* forgetting index overhead.

---

## 4. Chunking and parsing

**Q31 · Basic.** Why chunk documents?
- *Testing:* fundamentals.
- *Answer:* Model input limits, precision of one vector per idea, prompt budget.
- *AdvancedRAG:* 1,200 characters, 200 overlap.
- *Follow-up:* What if chunks are too big?
- *Avoid:* "to save storage".

**Q32 · Basic.** What is overlap for?
- *Testing:* boundary effects.
- *Answer:* Keeps facts at boundaries whole in at least one chunk; costs ~17% more vectors here.
- *AdvancedRAG:* overlap tail starts on a line or word boundary.
- *Follow-up:* Downsides?
- *Avoid:* huge overlaps causing duplicate evidence.

**Q33 · Intermediate.** Compare fixed-size, recursive, token-based, semantic and structure-aware chunking.
- *Testing:* breadth.
- *Answer:* Fixed: trivial but cuts words/rows. Recursive: prefers paragraph → line → sentence → word. Token: respects model limit. Semantic: splits at topic shifts, costs embeddings. Structure-aware: uses headings/tables.
- *AdvancedRAG:* recursive + page-aware + token guard; **not** semantic, **not** heading-aware.
- *Follow-up:* Show a boundary failure from the real code.
- *Avoid:* claiming semantic chunking.

**Q34 · Intermediate.** How can a 1,200-character chunk exceed the embedding model's limit?
- *Testing:* tokenisation.
- *Answer:* Numbers/codes/tables use many tokens per character; the model has a 512-token limit and silently truncates.
- *AdvancedRAG:* `fit_to_token_limit` re-splits using the real tokenizer.
- *Follow-up:* What does silent truncation do to retrieval?
- *Avoid:* assuming 4 chars/token always.

**Q35 · Intermediate.** What are PDF extraction's main limitations?
- *Testing:* practical parsing.
- *Answer:* Text layer only (scans need OCR), reading order in multi-column layouts, tables flattened, repeated headers/footers, hyphenation, fonts.
- *AdvancedRAG:* pypdf; empty pages are dropped silently; encrypted PDFs rejected.
- *Follow-up:* Design OCR support.
- *Avoid:* blaming embeddings for bad extraction.

**Q36 · Intermediate.** How are tables and CSVs handled?
- *Testing:* structure preservation.
- *Answer:* Each row becomes a self-describing line `column: value | ...`; DOCX tables are kept in document order; CSV also stored as a full table for exact analysis.
- *AdvancedRAG:* `parsing._row_text`, `rag_tables`.
- *Follow-up:* Header detection rule?
- *Avoid:* treating the first row as a header blindly.

**Q37 · Advanced.** A 25 MB document is uploaded. What happens?
- *Testing:* limits and bottlenecks.
- *Answer:* Passes the 25 MiB upload limit; the binding limits are the 5,000-chunk cap (≈ 5 MB of text) and the 1 h ingestion ceiling. A scanned 25 MB PDF fails `NO_EXTRACTABLE_TEXT`; 25 MB of prose fails `DOCUMENT_TOO_LARGE` after parse; embedding time is c × chunks.
- *AdvancedRAG:* worst documented success: 812 pages, 4,824 chunks (documented). I did not run a 25 MB test.
- *Follow-up:* What measurements do you need?
- *Avoid:* inventing timings.

**Q38 · Advanced.** Explain the base64 payload workaround, its cost, and alternatives.
- *Testing:* reading code critically.
- *Answer:* At `31bad7e`, with local storage, the file was base64-encoded into `background_jobs.payload` (×4/3: 25 MiB → 33.33 MiB), retained up to 30 days, with multiple in-memory copies. Alternatives: S3/R2, persistent volume, clear payload on completion, stream by reference.
- *AdvancedRAG:* **retired in the implementation pass**: jobs now reference a durable object (`local`/`s3`/`db`, recorded per document), legacy payloads are still processed and then scrubbed (`document_jobs.py`, `test_durability_versions.py`). Production config still selects `local`.
- *Follow-up:* Why did reprocess jobs fail after a restart, and what happens now?
- *Avoid:* defending it as a general design.

**Q39 · Advanced.** How do you handle duplicates?
- *Testing:* data quality.
- *Answer:* Dedup uploads by `(user, sha256)`, dedupe retrieved chunks by normalised-text hash, handle overlap deliberately.
- *AdvancedRAG:* upload dedup is now implemented per user on the content hash, race-safe on PostgreSQL via an advisory lock (tested). Dedup of *retrieved chunks* by content hash is still not implemented.
- *Follow-up:* Race-safe implementation?
- *Avoid:* claiming chunk-level dedup exists.

**Q40 · Advanced.** How do you update a document?
- *Testing:* lifecycle design.
- *Answer:* Re-ingest with deterministic ids so vectors overwrite, delete leftover ids, record max chunk count; or version documents.
- *AdvancedRAG:* re-ingest of the same id is idempotent, but there's no replace-content API: reprocess is for failed docs only, a new file is a new document.
- *Follow-up:* How do you avoid serving a half-updated document?
- *Avoid:* forgetting the limits: versions exist (replace/restore/purge) but `replaces_id` has no DB foreign key.

---

## 5. Information retrieval

**Q41 · Basic.** What is BM25?
- *Testing:* sparse retrieval.
- *Answer:* TF-IDF-family scoring with term-frequency saturation (k₁) and length normalisation (b).
- *AdvancedRAG:* not used; PostgreSQL FTS `ts_rank_cd` is used.
- *Follow-up:* Why does IDF matter?
- *Avoid:* saying the project uses BM25.

**Q42 · Intermediate.** Is PostgreSQL full-text search BM25?
- *Testing:* precision.
- *Answer:* No. `ts_rank`/`ts_rank_cd` use term frequency/proximity and optional length normalisation; PostgreSQL docs say ranking uses no global information (no IDF).
- *AdvancedRAG:* OR-ed `plainto_tsquery` + `ts_rank_cd`, GIN index.
- *Follow-up:* How would you add BM25?
- *Avoid:* "similar enough".

**Q43 · Intermediate.** Explain Reciprocal Rank Fusion with an example.
- *Testing:* hybrid ranking.
- *Answer:* Score = Σ 1/(k + rank), k = 60; rank-based so no score calibration. Example in Module 2 §4.4: A, C, B, E, D.
- *AdvancedRAG:* `pipeline.rrf`.
- *Follow-up:* Why k = 60?
- *Avoid:* adding raw scores.

**Q44 · Intermediate.** Bi-encoder vs cross-encoder.
- *Testing:* two-stage retrieval.
- *Answer:* Bi-encoder encodes separately (fast, indexable); cross-encoder scores pairs (accurate, slow).
- *AdvancedRAG:* Pinecone-hosted `bge-reranker-v2-m3` reranks ≤ 20 to 6; failure falls back to RRF order.
- *Follow-up:* What does recall of the first stage bound?
- *Avoid:* thinking the reranker finds new documents.

**Q45 · Intermediate.** Why hybrid retrieval?
- *Testing:* complementary strengths.
- *Answer:* Dense handles paraphrase; sparse handles exact tokens (SKUs, error codes) and is the fallback if the vector DB is down.
- *AdvancedRAG:* documented: semantic alone looked stronger on the benchmark (18/20 vs RRF 16/20); hybrid is kept for exact codes and degradation.
- *Follow-up:* When would you drop hybrid?
- *Avoid:* claiming the benchmark proves hybrid is better.

**Q46 · Intermediate.** Precision@K vs Recall@K trade-off.
- *Testing:* metrics.
- *Answer:* Larger K raises recall and lowers precision. Example [a,x,b,y,z] with relevant {a,b,c}.
- *AdvancedRAG:* benchmark reports hit@1/6 and MRR; no P@K/R@K.
- *Follow-up:* Choosing K=6 and 12,000 characters?
- *Avoid:* dividing recall by K.

**Q47 · Advanced.** Compute nDCG@4 for gains [3,2,0,1].
- *Testing:* metric fluency.
- *Answer:* DCG = 3 + 1.2619 + 0 + 0.4307 = 4.6925; IDCG = 4.7619; nDCG = 0.9854.
- *AdvancedRAG:* not computed in the repo.
- *Follow-up:* Exponential gain variant?
- *Avoid:* forgetting the log₂(i+1) denominator.

**Q48 · Advanced.** The correct chunk is rank 8 in dense and absent in keyword. Will it reach the final 6?
- *Testing:* pipeline reasoning.
- *Answer:* It is in the 20 candidates (rank 8 < 20); RRF gives 1/68; the reranker, reading query+passage jointly, may promote it. Candidate recall is the ceiling.
- *AdvancedRAG:* `RETRIEVAL_CANDIDATES=20`.
- *Follow-up:* What if it were rank 25?
- *Avoid:* "the reranker will fix it".

**Q49 · Advanced.** How do you debug "answer exists but retrieval misses it"?
- *Testing:* methodical debugging.
- *Answer:* Confirm extracted text, find the chunk, check boundaries and token truncation, check the vector exists, test a quoted-sentence query, compare dense vs keyword ranks, then query wording/rewrite.
- *AdvancedRAG:* `retrieve_done` logs strategy; benchmark records per-stage ranks.
- *Follow-up:* What if the quoted sentence retrieves it but the user's wording doesn't?
- *Avoid:* retuning the model first.

**Q50 · Advanced.** Why does AdvancedRAG avoid a similarity threshold?
- *Testing:* empirical thinking.
- *Answer:* Scores aren't calibrated; reranker scores of 0.007 (correct) vs 0.38 (unanswerable) overlap (documented); bge similarities cluster in ~0.6–1 per its model card.
- *AdvancedRAG:* `pipeline.py::retrieve`.
- *Follow-up:* How else to detect unanswerables?
- *Avoid:* thresholds from one anecdote.

---

## 6. LLMs and transformers

**Q51 · Basic.** What is a token?
- *Testing:* basics.
- *Answer:* Sub-word unit; limits and prices are in tokens (~4 chars/word-ish heuristic for English).
- *AdvancedRAG:* the code uses character caps and never counts LLM tokens.
- *Follow-up:* Why do numbers cost more?
- *Avoid:* "a token is a word".

**Q52 · Basic.** What does temperature do?
- *Testing:* sampling.
- *Answer:* Divides logits before softmax; lower is sharper. For [2,1,0]: T=1 → [0.665,0.245,0.090]; T=0.3 → [0.964,0.034,0.001].
- *AdvancedRAG:* answers 0.3, checks 0, follow-up transform 0.1.
- *Follow-up:* Does T=0 remove hallucination?
- *Avoid:* "temperature controls truthfulness".

**Q53 · Intermediate.** Explain self-attention.
- *Testing:* transformer basics.
- *Answer:* softmax(QKᵀ/√d)V; each token mixes values from others with learned weights; decoder models use a causal mask.
- *AdvancedRAG:* this is how the model "reads" retrieved passages.
- *Follow-up:* Why divide by √d?
- *Avoid:* hand-waving.

**Q54 · Intermediate.** Embedding model vs generation model.
- *Testing:* roles.
- *Answer:* Encoder → vector vs decoder → text.
- *AdvancedRAG:* local bge-small vs remote models via gateway; reranker is a cross-encoder.
- *Follow-up:* Could one model do both?
- *Avoid:* mixing their roles.

**Q55 · Intermediate.** How are retrieved passages inserted into the prompt?
- *Testing:* prompt design.
- *Answer:* Rules in system; fenced numbered `<source>` blocks in the last user message; delimiters neutralised; question last.
- *AdvancedRAG:* `prompts.py`, `context.py`.
- *Follow-up:* Why not the system message?
- *Avoid:* concatenating raw text.

**Q56 · Intermediate.** What is the lost-in-the-middle effect?
- *Testing:* long-context behaviour.
- *Answer:* Models often use information at the start/end better than in the middle (Liu et al., 2023).
- *AdvancedRAG:* best-ranked chunks first; context capped.
- *Follow-up:* Mitigations?
- *Avoid:* presenting as universal law.

**Q57 · Advanced.** What is indirect prompt injection and what does the project do about it?
- *Testing:* LLM security.
- *Answer:* Instructions inside retrieved data. Layers: role separation, fenced+neutralised evidence, explicit rules, no tools, UI image blocking, verification. No guarantee.
- *AdvancedRAG:* contract test uses a mock LLM: structure only.
- *Follow-up:* How would you test the real model?
- *Avoid:* "solved".

**Q58 · Advanced.** Why would you not use the runtime verifier as your evaluator?
- *Testing:* evaluation independence.
- *Answer:* It is part of the system; correlated errors; need independent judge and human labels.
- *AdvancedRAG:* `grading.verify`.
- *Follow-up:* Design an independent judge.
- *Avoid:* circular evaluation.

**Q59 · Advanced.** Describe the LLM call chain and its latency/cost.
- *Testing:* system-level reasoning.
- *Answer:* Typical 4 calls (rewrite, grade, generate, verify), worst case 8; the evidence is sent up to three times; grade precedes first token; verify follows streaming and may replace text.
- *AdvancedRAG:* `orchestrator.py`; `AGENT_FAST_MODEL` empty by default → same routed model.
- *Follow-up:* Where would you cut?
- *Avoid:* inventing prices.

**Q60 · Advanced.** How are provider failures handled?
- *Testing:* resilience.
- *Answer:* Retries only in the LLM service (≤ 2, backoff 1→8 s, `Retry-After` ≤ 10 s) for connect errors and 429/5xx; read timeouts not retried; stream retries only before first token; five public error codes; partial output kept.
- *AdvancedRAG:* `freellmapi.py`, `agent_service.py`.
- *Follow-up:* Why retry in one layer only?
- *Avoid:* retry stacking.

---

## 7. Evaluation

**Q61 · Basic.** Hit@K and MRR.
- *Testing:* definitions.
- *Answer:* Hit@K fraction with a relevant item in top K; MRR mean of 1/rank of first relevant.
- *AdvancedRAG:* final stage 19/20 Hit@1, 20/20 Hit@6, 0.975 MRR (documented).
- *Follow-up:* Rank pattern behind 0.975?
- *Avoid:* confusing with accuracy.

**Q62 · Basic.** Why isn't a working demo enough?
- *Testing:* quality mindset.
- *Answer:* No failure rate, no regression detection, hides failure types.
- *AdvancedRAG:* a frozen benchmark and guard tests exist.
- *Follow-up:* What's missing here?
- *Avoid:* "tests passed so it's good".

**Q63 · Intermediate.** Retrieval vs answer evaluation.
- *Testing:* separation.
- *Answer:* Retrieval: was the evidence found; answer: correct, faithful, cited, abstains properly.
- *AdvancedRAG:* the repo has retrieval metrics and a small E2E check.
- *Follow-up:* Retrieval up, answers worse: why?
- *Avoid:* one number for both.

**Q64 · Intermediate.** What does "19/20" mean statistically?
- *Testing:* uncertainty.
- *Answer:* Wilson 95% ≈ [0.76, 0.99]; one question difference is noise.
- *AdvancedRAG:* n = 20.
- *Follow-up:* Sample size for 90→95%?
- *Avoid:* reporting without intervals.

**Q65 · Intermediate.** How do you evaluate unanswerable questions?
- *Testing:* abstention metrics.
- *Answer:* False-answer rate on unanswerables and over-refusal rate on answerables, both with intervals.
- *AdvancedRAG:* benchmark has 2 E2E negatives and 6 retrieval-only probes: too few.
- *Follow-up:* Example numbers?
- *Avoid:* only testing answerable questions.

**Q66 · Intermediate.** What is test-set leakage?
- *Testing:* methodology.
- *Answer:* Tuning on the test set; questions written from chunk text; shared docs.
- *AdvancedRAG:* the benchmark's questions were visible during the fixes that "produced the baseline" (documented): it is a dev set.
- *Follow-up:* Fix?
- *Avoid:* calling it unbiased.

**Q67 · Advanced.** Audit the claim "Hit@6 20/20, MRR 0.975, so retrieval is excellent".
- *Testing:* critical reading.
- *Answer:* 69-chunk corpus (top-20 pool ≈ 29%), author-written questions, phrase proxy labels, single run, no intervals; consistent arithmetic but narrow.
- *AdvancedRAG:* Module 4 §6.
- *Follow-up:* What is the 2.3 vs 2.5 s issue?
- *Avoid:* repeating the number as production quality.

**Q68 · Advanced.** Design a fair retrieval experiment for chunk size.
- *Testing:* experimental design.
- *Answer:* Same corpus/questions, content labels, separate namespaces, dev/test, paired comparison, cost/latency too.
- *AdvancedRAG:* chunk values are frozen; change procedure in `docs/rag-architecture.md §6`.
- *Follow-up:* Which metric decides?
- *Avoid:* comparing on dev only.

**Q69 · Advanced.** Measure latency honestly.
- *Testing:* performance measurement.
- *Answer:* Percentiles with enough samples, warm/cold separation, concurrency, open-loop arrivals, per-stage timers, failures counted.
- *AdvancedRAG:* the documented latency is 20 retrieve calls (retrieval only); per-turn timing fields exist.
- *Follow-up:* p95 of 20 samples?
- *Avoid:* means only.

**Q70 · Advanced.** The benchmark's `REFUSAL` regex vs the abstain sentence.
- *Testing:* reading evaluation code.
- *Answer:* The regex didn't match "I couldn't find sufficient evidence...", so the E2E unanswerable gate would fail even for correct abstentions; the E2E part may predate the abstain contract.
- *AdvancedRAG:* verified by running the regex (Module 4 §5); fixed afterwards by recognising the abstain sentence directly. The fixed script was parse-checked but not run (it needs live Pinecone and a gateway).
- *Follow-up:* Fix?
- *Avoid:* trusting an old gate.

---

## 8. Security

**Q71 · Basic.** How are passwords stored?
- *Testing:* basics.
- *Answer:* Argon2id hashes; equal-time login for unknown users.
- *AdvancedRAG:* `core/security.py`.
- *Follow-up:* Why not SHA-256?
- *Avoid:* "encrypted".

**Q72 · Basic.** What file-upload controls exist?
- *Testing:* upload security.
- *Answer:* Allowlist, magic bytes, size limit, sanitised names, server-generated keys, attachment downloads with nosniff.
- *AdvancedRAG:* tests passed; no AV scanning.
- *Follow-up:* Zip bombs?
- *Avoid:* trusting Content-Type.

**Q73 · Intermediate.** How do access tokens and refresh tokens work here?
- *Testing:* auth design.
- *Answer:* 15-min JWT + session check; 7-day rotating refresh cookie with reuse detection and 30-s grace.
- *AdvancedRAG:* `deps.py`, `auth_service.py`.
- *Follow-up:* Why is logout immediate?
- *Avoid:* "JWTs can't be revoked" without mentioning session rows.

**Q74 · Intermediate.** IDOR and why 404 instead of 403.
- *Testing:* authorization.
- *Answer:* Owner-scoped queries; 404 avoids revealing existence.
- *AdvancedRAG:* security tests list endpoints; new routes could be missed.
- *Follow-up:* Auto-generate IDOR tests.
- *Avoid:* checking auth but not ownership.

**Q75 · Intermediate.** Tenant isolation in retrieval.
- *Testing:* multi-tenancy.
- *Answer:* Owner checks, namespace per user, document filter, user filter on rows.
- *AdvancedRAG:* contract test passed; weakness: internal callers supply `user_id` under one shared token.
- *Follow-up:* Fix?
- *Avoid:* single-layer isolation.

**Q76 · Intermediate.** Rate limiting and proxies.
- *Testing:* abuse prevention.
- *Answer:* Redis fixed window, fail-closed in production; client IP = N-th entry from the right of X-Forwarded-For.
- *AdvancedRAG:* `TRUSTED_PROXY_HOPS` must match reality (unverified).
- *Follow-up:* Fixed-window flaw?
- *Avoid:* trusting the left-most XFF.

**Q77 · Advanced.** Threat-model the system in five minutes.
- *Testing:* structured thinking.
- *Answer:* Assets, actors, boundaries; top risks: parser bombs, shared token, injection, DoS via streams, payload retention; evidence and status for each.
- *AdvancedRAG:* Module 6 register.
- *Follow-up:* Prioritise fixes.
- *Avoid:* "it's secure".

**Q78 · Advanced.** What sensitive data could leak through logs, and what's done?
- *Testing:* logging hygiene.
- *Answer:* Redaction regexes, route templates not raw paths, no question/document text in diagnostics; pattern-based redaction misses unlabeled secrets.
- *AdvancedRAG:* tests for share-token logging.
- *Follow-up:* Canary test.
- *Avoid:* logging bodies.

**Q79 · Advanced.** Deletion and retention.
- *Testing:* privacy.
- *Answer:* Delete must reach vectors, chunks, files, job payloads, caches, backups.
- *AdvancedRAG:* the base64 payload leak (job rows kept it ≥ 30 days) was fixed; message citation snippets of deleted documents are still retained, and there is no written retention policy.
- *Follow-up:* Test for it.
- *Avoid:* assuming soft-delete is deletion.

**Q80 · Advanced.** Handling a suspected cross-user data leak.
- *Testing:* incident response.
- *Answer:* Contain, collect request ids, determine scope, find root cause, fix, notify per policy, regression test, post-mortem.
- *AdvancedRAG:* `X-Request-ID` propagates across all services.
- *Follow-up:* What do you check first?
- *Avoid:* deleting logs.

---

## 9. Performance and scalability

**Q81 · Basic.** What is p95 latency?
- *Testing:* basics.
- *Answer:* 95% of requests are faster.
- *AdvancedRAG:* documented p50 1.1 s only.
- *Follow-up:* Samples needed for p99?
- *Avoid:* averages.

**Q82 · Basic.** What is the ingestion bottleneck?
- *Testing:* profiling mindset.
- *Answer:* Embedding CPU (≈ 99% on Render free, 74–95% on a laptop; documented).
- *AdvancedRAG:* batch size 4 beats 32 (documented).
- *Follow-up:* Scale how?
- *Avoid:* optimising parsing first.

**Q83 · Intermediate.** Little's Law applied.
- *Testing:* capacity math.
- *Answer:* L = λW; 100 in flight at 20 s → 5/s.
- *AdvancedRAG:* chat turns are long-lived.
- *Follow-up:* If W doubles?
- *Avoid:* ignoring tail latency.

**Q84 · Intermediate.** Async vs threads vs processes.
- *Testing:* concurrency.
- *Answer:* Async for I/O waits; threads for blocking calls; processes for CPU-bound Python.
- *AdvancedRAG:* `anyio.to_thread` for parsing/embedding; ONNX has its own threads.
- *Follow-up:* GIL?
- *Avoid:* "async makes CPU work faster".

**Q85 · Intermediate.** Connection pools.
- *Testing:* DB scaling.
- *Answer:* Bounded reused connections; replicas × pool ≤ DB limit; poolers need prepared statement care.
- *AdvancedRAG:* 5 + 10 per process; Neon pooled endpoint handled in `db.py`.
- *Follow-up:* Streams holding connections?
- *Avoid:* raising pools blindly.

**Q86 · Intermediate.** Caching in RAG.
- *Testing:* invalidation.
- *Answer:* Query embeddings, retrieval, answers; keys need user and versions; TTL.
- *AdvancedRAG:* no cache exists.
- *Follow-up:* Cache poisoning/leak?
- *Avoid:* keys without `user_id`.

**Q87 · Advanced.** Plan for 100 concurrent chat turns.
- *Testing:* systems analysis.
- *Answer:* Walk hops; likely limits: LLM gateway quotas, RAG Core CPU, DB pool, stream caps; add limits and measure.
- *AdvancedRAG:* Module 7 §5.2.
- *Follow-up:* What test passes it?
- *Avoid:* claiming it works.

**Q88 · Advanced.** Backpressure.
- *Testing:* overload behaviour.
- *Answer:* Bounded queues and fast rejection instead of unbounded waiting.
- *AdvancedRAG:* limits exist per user/IP; no global LLM limiter or stream cap.
- *Follow-up:* Implement.
- *Avoid:* retry storms.

**Q89 · Advanced.** Scaling plan from prototype to production.
- *Testing:* staged judgement.
- *Answer:* Storage + CPU + limits + backups first; then worker/RAG replicas, tracing, per-service creds; GPU/sharding later.
- *AdvancedRAG:* Module 7 §6.
- *Follow-up:* What not to build?
- *Avoid:* Kubernetes first.

**Q90 · Advanced.** Reduce p95 time-to-first-token.
- *Testing:* optimisation.
- *Answer:* Measure; skip/overlap grading, cache embeddings, warm services, trim evidence, faster check model.
- *AdvancedRAG:* first token waits for rewrite+retrieval+grading.
- *Follow-up:* Risk of overlapping grading?
- *Avoid:* guessing.

---

## 10. Python

**Q91 · Basic.** The GIL and its implications.
- *Testing:* runtime knowledge.
- *Answer:* One thread executes Python bytecode at a time; threads help I/O and native code that releases it.
- *AdvancedRAG:* parse in threads frees the event loop only.
- *Follow-up:* Multiprocessing trade-offs?
- *Avoid:* "Python can't do concurrency".

**Q92 · Basic.** Generators and why they matter for streaming.
- *Testing:* language features.
- *Answer:* Lazy iteration; SSE endpoints yield frames.
- *AdvancedRAG:* `stream_turn` is an async generator.
- *Follow-up:* Cleanup on cancellation?
- *Avoid:* building lists of everything.

**Q93 · Intermediate.** `asyncio.gather` pitfalls.
- *Testing:* async correctness.
- *Answer:* First exception propagates, others keep running unless cancelled; use `return_exceptions` or per-task guards; bound concurrency.
- *AdvancedRAG:* Pinecone upsert cancels siblings on failure.
- *Follow-up:* CancelledError semantics?
- *Avoid:* unbounded gather.

**Q94 · Intermediate.** Dataclass vs Pydantic model.
- *Testing:* tool choice.
- *Answer:* Dataclass: plain containers; Pydantic: validation/serialisation at boundaries.
- *AdvancedRAG:* Pydantic for APIs, dataclasses internally.
- *Follow-up:* Cost of validation?
- *Avoid:* validating internal hot loops.

**Q95 · Intermediate.** Context managers.
- *Testing:* resource handling.
- *Answer:* `with`/`async with` guarantee cleanup.
- *AdvancedRAG:* `timed()` and `session_scope()`.
- *Follow-up:* Write one.
- *Avoid:* manual close.

**Q96 · Intermediate.** Mutable default arguments.
- *Testing:* gotchas.
- *Answer:* Shared across calls; use `None`.
- *AdvancedRAG:* `payload or {}` pattern.
- *Follow-up:* Dataclass `field(default_factory)`.
- *Avoid:* `def f(x=[])`.

**Q97 · Intermediate.** Time complexity of dict, set, list operations.
- *Testing:* data structures.
- *Answer:* Dict/set O(1) average; list membership O(n); heap push/pop O(log n).
- *AdvancedRAG:* top-k via heap.
- *Follow-up:* Amortised append?
- *Avoid:* `in list` in hot loops.

**Q98 · Advanced.** Type hints and runtime validation.
- *Testing:* quality practices.
- *Answer:* Hints aid tools; they're not enforced at runtime; validate at I/O boundaries.
- *AdvancedRAG:* Pydantic models; mypy-like discipline unknown.
- *Follow-up:* Protocols?
- *Avoid:* trusting annotations.

**Q99 · Advanced.** Decorators and dependency injection in testing.
- *Testing:* design.
- *Answer:* Pass collaborators in; fake them in tests; avoid hidden globals.
- *AdvancedRAG:* `Services` container in RAG Core; `set_storage`, `set_backend` hooks.
- *Follow-up:* Global singletons downsides?
- *Avoid:* monkeypatching everything.

**Q100 · Advanced.** Memory efficiency for big files.
- *Testing:* practical Python.
- *Answer:* Stream in chunks, avoid multiple copies, generators, `memoryview`; measure with `tracemalloc`.
- *AdvancedRAG:* backend scan streams; worker/RAG Core load whole files.
- *Follow-up:* base64 overhead?
- *Avoid:* assuming `read()` is free.

---

## 11. SQL

**Q101 · Basic.** Index basics.
- *Testing:* fundamentals.
- *Answer:* Auxiliary structure to speed lookups at write cost.
- *AdvancedRAG:* `ix_background_jobs_status_run_after`, `ix_documents_user_id_created_at`, GIN full-text.
- *Follow-up:* Composite index order?
- *Avoid:* indexing everything.

**Q102 · Basic.** INNER vs LEFT JOIN.
- *Testing:* joins.
- *Answer:* Inner drops non-matches; left keeps left rows with NULLs.
- *AdvancedRAG:* users without ready documents.
- *Follow-up:* Filter placement pitfall.
- *Avoid:* WHERE on the right table after LEFT JOIN.

**Q103 · Intermediate.** GROUP BY with HAVING.
- *Testing:* aggregation.
- *Answer:* HAVING filters after aggregation (Module 8 E20).
- *AdvancedRAG:* document counts per status.
- *Follow-up:* Conditional aggregation.
- *Avoid:* WHERE with aggregates.

**Q104 · Intermediate.** Transaction isolation levels in brief.
- *Testing:* consistency.
- *Answer:* Read committed (PostgreSQL default), repeatable read, serializable; anomalies differ.
- *AdvancedRAG:* relies on row locks and CAS rather than serializable.
- *Follow-up:* Phantom reads?
- *Avoid:* claiming default is serializable.

**Q105 · Intermediate.** `SELECT ... FOR UPDATE SKIP LOCKED`.
- *Testing:* queue patterns.
- *Answer:* Lock rows for this transaction and skip locked ones, enabling concurrent workers.
- *AdvancedRAG:* `claim_next`, plus CAS for portability.
- *Follow-up:* Why also CAS?
- *Avoid:* assuming SQLite supports it.

**Q106 · Intermediate.** Compare-and-set pattern.
- *Testing:* concurrency.
- *Answer:* `UPDATE ... WHERE state = expected`; check rowcount.
- *AdvancedRAG:* job outcomes and ingestion claims.
- *Follow-up:* ABA problem?
- *Avoid:* read-then-write.

**Q107 · Advanced.** Atomic upload: document and job.
- *Testing:* transactional design.
- *Answer:* One transaction; blob cleanup if commit fails.
- *AdvancedRAG:* `document_service.upload`.
- *Follow-up:* Outbox pattern?
- *Avoid:* two commits.

**Q108 · Advanced.** Reading `EXPLAIN ANALYZE`.
- *Testing:* tuning.
- *Answer:* Look for seq scans, row estimates vs actual, loops, index usage.
- *AdvancedRAG:* whether the planner uses the GIN index for filtered FTS queries is unknown without running it.
- *Follow-up:* How to force/verify?
- *Avoid:* tuning blind.

**Q109 · Advanced.** Full-text search syntax and safety.
- *Testing:* PostgreSQL features.
- *Answer:* `to_tsvector`, `plainto_tsquery` (AND), `websearch_to_tsquery`, ranking functions; user text as bound params.
- *AdvancedRAG:* OR via replacing `&` in the generated tsquery.
- *Follow-up:* Injection risk?
- *Avoid:* string-building queries.

**Q110 · Advanced.** Schema migrations.
- *Testing:* ops.
- *Answer:* Versioned Alembic migrations; run as a release step; backwards-compatible changes.
- *AdvancedRAG:* backend 4, RAG 3; container migrates on start (single replica only).
- *Follow-up:* Zero-downtime column change?
- *Avoid:* editing applied migrations.

---

## 12. FastAPI

**Q111 · Basic.** What does FastAPI give you?
- *Testing:* framework basics.
- *Answer:* Type-driven validation, OpenAPI, dependency injection, async support.
- *AdvancedRAG:* all four services.
- *Follow-up:* Why disable `/docs` in production?
- *Avoid:* confusing with Flask.

**Q112 · Basic.** `UploadFile` vs `bytes`.
- *Testing:* memory.
- *Answer:* `UploadFile` spools to disk past a threshold; `bytes` loads everything.
- *AdvancedRAG:* backend uses `UploadFile`.
- *Follow-up:* Bounded read?
- *Avoid:* reading unbounded.

**Q113 · Intermediate.** Dependency injection.
- *Testing:* architecture.
- *Answer:* `Depends` for DB sessions, auth, services; override in tests.
- *AdvancedRAG:* `CurrentUser`, `DB`.
- *Follow-up:* Dependencies with `yield` and streaming?
- *Avoid:* globals.

**Q114 · Intermediate.** Validation and error handling.
- *Testing:* API quality.
- *Answer:* Pydantic constraints → 422; custom handlers produce uniform error envelopes and hide internals.
- *AdvancedRAG:* `install_error_handlers`, safe codes.
- *Follow-up:* 400 vs 422?
- *Avoid:* returning stack traces.

**Q115 · Intermediate.** Middleware ordering.
- *Testing:* request lifecycle.
- *Answer:* Outermost first; request context wraps everything; body limit before parsing.
- *AdvancedRAG:* `create_app` in `main.py`.
- *Follow-up:* Raw ASGI vs BaseHTTPMiddleware?
- *Avoid:* reading bodies in middleware carelessly.

**Q116 · Intermediate.** Background tasks vs a job queue.
- *Testing:* reliability.
- *Answer:* In-process tasks die with the process; a durable queue survives and retries.
- *AdvancedRAG:* Postgres queue + worker in the lifespan.
- *Follow-up:* When Celery?
- *Avoid:* `BackgroundTasks` for hour-long work.

**Q117 · Advanced.** Streaming responses and disconnects.
- *Testing:* SSE.
- *Answer:* Generator yields frames; handle `CancelledError`/`GeneratorExit`; persist partial state.
- *AdvancedRAG:* `stream_turn` marks cancelled and shields persistence.
- *Follow-up:* Proxy buffering?
- *Avoid:* forgetting `X-Accel-Buffering: no`.

**Q118 · Advanced.** Lifespan events.
- *Testing:* app lifecycle.
- *Answer:* Startup/shutdown for workers, clients, pools.
- *AdvancedRAG:* worker start, model warm-up, `close_clients`.
- *Follow-up:* Graceful shutdown?
- *Avoid:* leaking tasks.

**Q119 · Advanced.** Testing FastAPI apps.
- *Testing:* quality.
- *Answer:* `TestClient`/ASGI transport, dependency overrides, fakes, factories.
- *AdvancedRAG:* backend 218 tests passed in my run (4 skipped).
- *Follow-up:* Testing SSE?
- *Avoid:* hitting real services.

**Q120 · Advanced.** `def` vs `async def` endpoints.
- *Testing:* concurrency.
- *Answer:* `def` runs in a threadpool; `async def` on the loop; never block in `async def`.
- *AdvancedRAG:* blocking work wrapped in `to_thread`.
- *Follow-up:* Detect blocking?
- *Avoid:* sync DB calls in async code.

---

## 13. Testing and debugging

**Q121 · Basic.** Why mock external providers in tests?
- *Testing:* test hygiene.
- *Answer:* Determinism, speed, no cost/quota, failure injection.
- *AdvancedRAG:* mock OpenAI-compatible server, mock Pinecone.
- *Follow-up:* What do mocks fail to prove?
- *Avoid:* mocking your own logic.

**Q122 · Basic.** Unit, integration, contract, end-to-end.
- *Testing:* test pyramid.
- *Answer:* Scope increases; contract tests check service boundaries.
- *AdvancedRAG:* contract suite passed (52).
- *Follow-up:* Which catches schema drift?
- *Avoid:* only e2e.

**Q123 · Intermediate.** What do skipped tests tell you?
- *Testing:* honest reporting.
- *Answer:* Coverage gaps depend on environment; know which.
- *AdvancedRAG:* PostgreSQL-only concurrency/FTS tests skip on SQLite; Redis/S3 tests skip without env.
- *Follow-up:* How would you close it in CI?
- *Avoid:* reporting "all passed".

**Q124 · Intermediate.** Testing retry logic.
- *Testing:* time handling.
- *Answer:* Inject sleep, script outcomes, assert call counts and delays.
- *AdvancedRAG:* provider tests.
- *Follow-up:* Jitter?
- *Avoid:* real sleeps.

**Q125 · Intermediate.** Property-based testing use cases.
- *Testing:* advanced testing.
- *Answer:* Chunking is lossless; RRF permutation-invariant; normalisation idempotent.
- *AdvancedRAG:* chunk invariants, e.g., `"".join(parts) == text` for sentence split.
- *Follow-up:* Hypothesis strategies?
- *Avoid:* examples only.

**Q126 · Intermediate.** Use request ids for debugging.
- *Testing:* observability.
- *Answer:* One id across hops lets you follow one user action.
- *AdvancedRAG:* `X-Request-ID` propagated.
- *Follow-up:* Tracing vs request id?
- *Avoid:* logging PII to correlate.

**Q127 · Advanced.** Debug: a document stays `processing`.
- *Testing:* method.
- *Answer:* Job status/attempts/progress, RAG attempt status/stage/heartbeat, logs, stale sweep, RAG health.
- *AdvancedRAG:* `background_jobs.progress` shows `rag:<stage>`.
- *Follow-up:* When is it a crash vs slow?
- *Avoid:* restarting first.

**Q128 · Advanced.** Debug: vectors missing for a ready document.
- *Testing:* data integrity.
- *Answer:* Compare chunk count vs Pinecone namespace count; fetch ids; check verification; check delete jobs; eventual consistency.
- *AdvancedRAG:* verification samples ≤ 54 ids; ready means sample readable, not every vector.
- *Follow-up:* Full audit?
- *Avoid:* assuming ready = all verified.

**Q129 · Advanced.** Reproduce a production-only bug.
- *Testing:* engineering practice.
- *Answer:* Capture request id, inputs (sanitised), versions/config; reproduce locally with the same config; add a regression test.
- *AdvancedRAG:* production vs e2e modes.
- *Follow-up:* When data can't be shared?
- *Avoid:* debugging in prod blindly.

**Q130 · Advanced.** Chaos/failure testing plan.
- *Testing:* resilience verification.
- *Answer:* Kill RAG mid-ingest, throttle Pinecone, return 429s, drop Redis, restart during streams; verify documented recovery.
- *AdvancedRAG:* reliability tests with mocks; real chaos missing.
- *Follow-up:* Success criteria?
- *Avoid:* happy path only.

---

## 14. System design

**Q131 · Intermediate.** Design a RAG assistant for 1,000 users.
- *Testing:* end-to-end design.
- *Answer:* Auth, object storage, queue + workers, chunk/vector stores, hybrid retrieval + rerank, grounded generation with citations, evals, limits, monitoring.
- *AdvancedRAG:* it is that design at small scale.
- *Follow-up:* What changes at 100k?
- *Avoid:* skipping evaluation.

**Q132 · Intermediate.** Multi-tenant isolation design.
- *Testing:* security architecture.
- *Answer:* Namespaces/filters, signed tenant context, per-tenant quotas/keys, tests.
- *AdvancedRAG:* four layers + shared-token weakness.
- *Follow-up:* Tenant-specific encryption?
- *Avoid:* relying on prompts.

**Q133 · Intermediate.** Real-time document updates.
- *Testing:* consistency.
- *Answer:* Versioned ingestion with atomic switch, tombstones, eventual consistency windows.
- *AdvancedRAG:* versioning exists now: replace uploads a new version sharing a `logical_id`, the old version keeps serving until an atomic switch marks it `SUPERSEDED`, restore and a retention purge exist (`test_durability_versions.py`).
- *Follow-up:* Avoid half-updated reads.
- *Avoid:* mutating in place without visibility control.

**Q134 · Advanced.** Add per-sentence citations.
- *Testing:* grounding design.
- *Answer:* Claim-level attribution with quote spans, verifier per claim, UI linking.
- *AdvancedRAG:* currently chunk-level `[n]` with snippet.
- *Follow-up:* Evaluation?
- *Avoid:* unverifiable citations.

**Q135 · Advanced.** Migrate embeddings safely.
- *Testing:* data migration.
- *Answer:* New index, backfill from stored text, shadow eval, cutover, rollback plan.
- *AdvancedRAG:* chunk text in PostgreSQL makes backfill possible.
- *Follow-up:* Cost estimate?
- *Avoid:* in-place overwrite.

**Q136 · Advanced.** Cost control design.
- *Testing:* FinOps.
- *Answer:* Budgets per user, caching, cheaper models for checks, skip checks adaptively, token accounting.
- *AdvancedRAG:* token usage partial (stream usage off by default).
- *Follow-up:* How to attribute cost?
- *Avoid:* unmeasured assumptions.

**Q137 · Advanced.** Observability design.
- *Testing:* operations.
- *Answer:* Metrics, structured logs, traces, SLOs, alerts, dashboards.
- *AdvancedRAG:* metrics, JSON logs, request ids, W3C `traceparent` propagation, queue-depth/age/stuck gauges and job metrics exist (`test_observability.py`); tracing spans/exporter, dashboards and alerts are still missing.
- *Follow-up:* Top five alerts?
- *Avoid:* log everything.

**Q138 · Advanced.** On-prem/air-gapped variant.
- *Testing:* constraints.
- *Answer:* Local embeddings (already), local LLM, pgvector, S3-compatible storage, no external APIs.
- *AdvancedRAG:* embeddings local; Pinecone/rerank/gateway are external.
- *Follow-up:* Reranker replacement?
- *Avoid:* forgetting rerank dependency.

**Q139 · Advanced.** Design an evaluation service.
- *Testing:* ML ops.
- *Answer:* Versioned datasets, runners, metrics with CIs, regression gates, dashboards, human review queue.
- *AdvancedRAG:* a script exists; no service.
- *Follow-up:* Online feedback loop?
- *Avoid:* eval on the training set.

**Q140 · Advanced.** Design for resumable large-document ingestion.
- *Testing:* reliability.
- *Answer:* Slice checkpoints, durable partial vectors hidden until finalisation, resumable attempts, verification at the end.
- *AdvancedRAG:* named by docs as the largest remaining win.
- *Follow-up:* Visibility control?
- *Avoid:* breaking the "no partial state" guarantee.

---

## 15. Practical incidents

**Q141 · Intermediate.** Users see "No text could be extracted".
- *Testing:* triage.
- *Answer:* Scanned/image PDF; confirm via text layer; offer OCR or message.
- *AdvancedRAG:* `NO_EXTRACTABLE_TEXT` permanent.
- *Follow-up:* Partial scans?
- *Avoid:* retrying.

**Q142 · Intermediate.** Answers abstain more after a deploy.
- *Testing:* regression analysis.
- *Answer:* Check evidence grade distribution, grader JSON failures, model route change, retrieval strategy (`keyword_only_degraded`, `reranker: skipped`), prompt changes.
- *AdvancedRAG:* metadata fields.
- *Follow-up:* Rollback criteria?
- *Avoid:* blaming users.

**Q143 · Intermediate.** `/ready` shows `DIMENSION_MISMATCH`.
- *Testing:* config incident.
- *Answer:* Index dimension ≠ model; create correct index or fix config; re-ingest if model changed.
- *AdvancedRAG:* documented error text.
- *Follow-up:* Prevent?
- *Avoid:* resizing in place.

**Q144 · Advanced.** Spike of `LLM_RATE_LIMITED`.
- *Testing:* overload.
- *Answer:* Check call amplification (4–8/turn), concurrency, gateway quotas; add limiter/queue/fallback.
- *AdvancedRAG:* backoff ≤ 2 retries; rewrite fails fast.
- *Follow-up:* Communicate to users.
- *Avoid:* more retries.

**Q145 · Advanced.** RAG Core killed by OOM during ingestion.
- *Testing:* resource incident.
- *Answer:* Identify file; whole-file reads; batch sizes; parse bombs; limit; subprocess; more RAM.
- *AdvancedRAG:* stale heartbeat → retry loop risk.
- *Follow-up:* Poison document handling.
- *Avoid:* infinite retries.

**Q146 · Advanced.** Deleted documents still appear in answers.
- *Testing:* data lifecycle.
- *Answer:* Check status, vectors/chunks deletion, caches, stale message citations; delete job state.
- *AdvancedRAG:* deletion is soft then job; reconciler; old message citations remain.
- *Follow-up:* SLA for deletion.
- *Avoid:* assuming instant.

**Q147 · Advanced.** First message after idle takes 40 s.
- *Testing:* free-tier behaviour.
- *Answer:* Cold start; wake relay; model load; paid plan or keep-alive.
- *AdvancedRAG:* `agent.waking` events, wake budgets.
- *Follow-up:* Measure.
- *Avoid:* optimising the model.

**Q148 · Advanced.** Neon storage nearly full.
- *Testing:* capacity.
- *Answer:* Find big tables: job payloads (legacy base64), `stored_objects` (db backend), chunk text, CSV tables, messages; purge/retention; S3.
- *AdvancedRAG:* the base64 payload is gone for new uploads; with `STORAGE_BACKEND=db` the file bytes live in `stored_objects`, which is the table to watch.
- *Follow-up:* Immediate relief.
- *Avoid:* dropping data blindly.

**Q149 · Advanced.** One user's bad question seems to degrade others.
- *Testing:* noisy neighbour.
- *Answer:* Per-user stream cap, global limiter, queue fairness.
- *AdvancedRAG:* missing caps; `INGEST_CONCURRENCY=1` head-of-line blocking.
- *Follow-up:* Fairness policy.
- *Avoid:* only scaling up.

**Q150 · Advanced.** Citations point to the wrong page.
- *Testing:* data lineage.
- *Answer:* Page from chunk metadata; check parsing page mapping, empty-page skipping, chunk page assignment, duplicate docs.
- *AdvancedRAG:* chunk `page` from PDF page number; chunks never cross pages.
- *Follow-up:* DOCX has no pages.
- *Avoid:* trusting model-written page numbers (they come from metadata).

---

## 16. Behavioural

> Use **only true** experiences. The AdvancedRAG facts below are honest talking points about the project; adapt to what *you* did. Do not claim outcomes (users, scale, revenue) that did not happen.

**Q151 · Basic.** Tell me about a project you're proud of.
- *Testing:* ownership, clarity.
- *Answer (structure):* Problem → your role → key decisions → measured result → what you'd change. Keep numbers you can source.
- *AdvancedRAG:* e.g., "a hybrid RAG system with verified ingestion and fail-safe answers; documented 20-question retrieval baseline."
- *Follow-up:* What did you measure?
- *Avoid:* inflating.

**Q152 · Intermediate.** Describe a hard bug you solved.
- *Testing:* debugging method.
- *Answer:* Symptom, hypothesis, evidence, fix, regression test, lesson.
- *AdvancedRAG:* candidates from the repo's history: silent embedding truncation (token guard), DOCX table order, overlap cutting rows, dimension mismatch handling (documented fixes).
- *Follow-up:* How did you confirm the fix?
- *Avoid:* heroics without evidence.

**Q153 · Intermediate.** A technical trade-off you made.
- *Testing:* judgement.
- *Answer:* Options, criteria, decision, cost accepted.
- *AdvancedRAG:* no score threshold; hybrid kept despite weaker numbers; Postgres queue instead of a broker.
- *Follow-up:* Would you decide differently now?
- *Avoid:* pretending no downsides.

**Q154 · Intermediate.** What are the weaknesses of your project?
- *Testing:* honesty.
- *Answer:* Be specific: shared internal token, parser limits, payload retention, no load test, small benchmark.
- *AdvancedRAG:* Module 6 register.
- *Follow-up:* Which would you fix first and why?
- *Avoid:* "none".

**Q155 · Intermediate.** How do you prioritise when everything is urgent?
- *Testing:* prioritisation.
- *Answer:* Risk × effort; security and data loss first; measurable user impact.
- *AdvancedRAG:* storage durability and limits before scaling.
- *Follow-up:* Example.
- *Avoid:* "everything".

**Q156 · Intermediate.** How do you handle disagreement on a design?
- *Testing:* collaboration.
- *Answer:* Clarify goals, propose experiments, document decisions (ADR), commit once decided.
- *AdvancedRAG:* the "frozen baseline" decision process is an example of recorded decisions.
- *Follow-up:* When you were wrong.
- *Avoid:* blaming.

**Q157 · Advanced.** How do you know a change improved the system?
- *Testing:* data-driven mindset.
- *Answer:* Pre-registered metrics, held-out test, intervals, regression gate, monitoring after release.
- *AdvancedRAG:* benchmark gate in docs; limits acknowledged.
- *Follow-up:* What if metrics conflict?
- *Avoid:* anecdotes.

**Q158 · Advanced.** Tell me about a mistake and what you changed.
- *Testing:* learning.
- *Answer:* Facts, impact, root cause, process fix.
- *AdvancedRAG:* e.g., a stale benchmark gate (regex vs abstain message) shows why gates need re-running.
- *Follow-up:* How do you prevent recurrence?
- *Avoid:* minimising.

**Q159 · Advanced.** How do you learn new technology quickly?
- *Testing:* growth.
- *Answer:* Official docs, small experiments with tests, write-ups.
- *AdvancedRAG:* verified claims against official docs (bge card, Pinecone, pgvector, PostgreSQL).
- *Follow-up:* Example.
- *Avoid:* tutorial hopping.

**Q160 · Advanced.** Where do you want to grow as an AI engineer?
- *Testing:* self-awareness.
- *Answer:* Evaluation rigor, production reliability, cost/latency engineering, security of LLM systems.
- *AdvancedRAG:* the open items (load testing, adversarial evaluation, resumable ingestion) are natural next steps.
- *Follow-up:* First 90 days.
- *Avoid:* buzzwords.

---

## 17. Implementation pass: what was built and verified

These questions cover the work done after the first version of this course. Every claim is backed by a test or recorded run listed in [FEATURE_MATRIX](docs/verification/FEATURE_MATRIX.md); all evidence is hermetic (no real Pinecone, gateway or deployment), and nothing is committed.

**Q161 · Intermediate.** Users reported "The uploaded file could not be found". How did you find and fix the root cause?
- *Testing:* debugging method, distinguishing symptom from cause.
- *Answer:* Reproduce first: on Render's ephemeral disk a restart deletes locally stored files; the original ingest carried the bytes as base64 in its job, but `reprocess` and the reconciler enqueued jobs with no bytes, so they failed. Fix: record the storage backend per document, make jobs reference durable objects (`s3` or atomic `db`), verify a sha256 on read, return distinct errors (`FILE_MISSING` permanent with diagnostics vs `STORAGE_UNAVAILABLE` retryable), keep legacy jobs working.
- *AdvancedRAG:* `document_service.py`, `workers/document_jobs.py`, `integrations/storage.py`, `test_durability_versions.py`. Production config still selects `local`; the operator must switch.
- *Follow-up:* Why make the `db` backend commit the blob in the same transaction as the job?
- *Avoid:* saying the production bug is fixed; only the code path and its tests are.

**Q162 · Intermediate.** How does OCR work here and what do you know about its accuracy?
- *Testing:* honest limits of an ML component.
- *Answer:* Pages with too little native text are rendered (pypdfium2) and recognised by a self-hosted ONNX engine (RapidOCR) under page/document time and pixel budgets; mixed PDFs keep native pages; engine confidence and failed pages go into an extraction report. Accuracy is not measured: tests show clean synthetic scans are read and failures degrade honestly; there is no character-error-rate benchmark.
- *AdvancedRAG:* `rag_core/ocr.py`, `parsing.py`, `test_ocr_extraction.py`; off by default (`OCR_ENABLED`).
- *Follow-up:* How would you measure OCR quality?
- *Avoid:* quoting an accuracy number.

**Q163 · Intermediate.** A 40-page PDF has 12 scanned pages. What should the user see?
- *Testing:* surfacing partial failure instead of hiding it.
- *Answer:* With OCR on, those pages are read and tagged `ocr` with confidence; failed/low-confidence pages become warnings. With OCR off, the document is indexed from the readable pages and the report lists the unprocessed pages so the user knows they are not searchable.
- *AdvancedRAG:* `ExtractionReport` stored on the document and returned by the status API.
- *Follow-up:* Where is it shown in the UI?
- *Avoid:* "ready" with no warning.

**Q164 · Advanced.** Compare recursive, heading-aware and semantic chunking using your evaluation. Which would you ship?
- *Testing:* reading results without overclaiming.
- *Answer:* On the synthetic corpus (114 answerable questions, one run) none clearly beat recursive for dense search; heading-aware plus BM25 fusion gave the best Hit@1 (0.833 vs 0.746); semantic chunking cost ~7x more embedding texts without a win. Intervals overlap, so keep recursive as the default and treat the others as options.
- *AdvancedRAG:* `docs/verification/RETRIEVAL_EVALUATION.md`.
- *Follow-up:* What experiment would settle it?
- *Avoid:* declaring a winner from one run.

**Q165 · Advanced.** Your hybrid retrieval scored worse than dense alone. What do you do?
- *Testing:* diagnosing a negative result.
- *Answer:* Report it. Check the keyword leg alone (Hit@1 0.298 vs BM25 0.746), form a hypothesis (`ts_rank_cd` has no IDF), test it (swap in a BM25 scorer, or weight RRF), and only then change production. Do not tune on the test questions.
- *AdvancedRAG:* hypothesis stated, not yet tested; Postgres FTS vs BM25 reference in the evaluation.
- *Follow-up:* How would you avoid overfitting the fix?
- *Avoid:* hiding the result or "fixing" it by editing labels.

**Q166 · Advanced.** How do you guarantee the index never mixes embedding models?
- *Testing:* integrity design.
- *Answer:* Record model and dimension on each document, refuse or exclude mismatches at retrieval (`EMBEDDING_MODEL_MISMATCH`), check the Pinecone dimension and metric at startup/ready, and repair with an idempotent reindex that overwrites the same deterministic vector ids from stored chunks.
- *AdvancedRAG:* `pipeline.retrieve`, `reindex_document`, `vectorstore.py`; tests in `test_integrity_and_spool.py`. A real Pinecone index was not queried.
- *Follow-up:* What if the old model's vectors exist for ids you never overwrite?
- *Avoid:* claiming model identity is checked inside Pinecone.

**Q167 · Intermediate.** Explain document replacement and what "atomic" means here.
- *Testing:* consistency under concurrency.
- *Answer:* A new version is uploaded and indexed while the current one keeps answering; when indexing finishes, one transaction marks the new version current and the old `SUPERSEDED`; restore reverses it; superseded versions are purged after a retention window. Identical content is a no-op.
- *AdvancedRAG:* `document_service.replace/restore`, `cleanup_jobs.purge_superseded_versions`.
- *Follow-up:* What if indexing the new version fails?
- *Avoid:* forgetting that vectors of the old version must survive until the switch.

**Q168 · Intermediate.** What did you measure about large files, and what can't you claim?
- *Testing:* separating measurement from extrapolation.
- *Answer:* With fake embeddings: 1 MB of text indexed (1,456 chunks, 0.8 s, 125 MB peak); 5, 10 and 25 MB were refused at the 5,000-chunk limit in under 3 s (peak 221 MB). Real embedding speed was 34.8 chunks/s in a 256-chunk sample, so a full 5,000-chunk document is roughly 2.5 minutes of embedding on that laptop (extrapolation; my first figure of 3.7 chunks/s was a mistake: the timing window included a model download, which I found by re-measuring). Bounded load tests then ran 6-12 documents and 8-24 chat users concurrently on an isolated stack with zero errors, after they exposed a real Windows path bug. No Pinecone/Neon latency, soak test or capacity claim.
- *AdvancedRAG:* `scripts/perf_ingest.py`, `PERFORMANCE_AND_BACKUP.md`.
- *Follow-up:* Is a 25 MB upload "supported"?
- *Avoid:* quoting capacity numbers.

**Q169 · Intermediate.** How would you detect a stalled ingestion queue without reading logs?
- *Testing:* observability design.
- *Answer:* Export queue depth, age of the oldest pending job and stuck (stale-heartbeat) jobs as gauges; job outcome, duration and error-code counters; propagate a trace id across services. Alert on oldest-pending-age and stuck > 0.
- *AdvancedRAG:* `job_service.queue_stats`, `/metrics`, `test_observability.py`. No dashboards or alerts are configured.
- *Follow-up:* Which alerts first?
- *Avoid:* confusing metrics being exported with monitoring being in place.

**Q170 · Advanced.** What did the dependency audit find and what did you change?
- *Testing:* security hygiene with evidence.
- *Answer:* `pip-audit` on the dev environment reported advisories in Pillow 12.3.0, pypdf 6.19.0 and certifi; both libraries parse untrusted uploads. I upgraded (Pillow 13, pypdf 6.21.1), re-audited clean, re-ran the rag-core suite (201 passed on PostgreSQL), and raised the floors in `requirements.txt`. Caveats: requirements are unpinned ranges and the audit covered the venv, not the production images.
- *AdvancedRAG:* `docs/verification/SECURITY_AUDIT.md`.
- *Follow-up:* How would you make audits reproducible?
- *Avoid:* "no known vulnerabilities" without saying what was scanned.

**Q171 · Advanced.** Which hostile uploads are handled, and which are not?
- *Testing:* threat modelling a parser.
- *Answer:* Handled and tested: zip bomb by size and by ratio, fake DOCX, pixel-flood image, truncated/garbage images, content sniffing by magic bytes, attachment downloads with `nosniff`. Not handled: sandboxed parsing, a pypdf timeout, specific polyglot tests, antivirus.
- *AdvancedRAG:* `test_hostile_inputs.py`, `test_security_primitives.py`.
- *Follow-up:* How would you sandbox the parser?
- *Avoid:* calling the upload path "secure".

**Q172 · Advanced.** Which test results in this repository would you refuse to quote as production evidence, and why?
- *Testing:* epistemic honesty.
- *Answer:* All of them: they use a local vector store or a Pinecone emulator, a mock gateway, a throw-away PostgreSQL and synthetic data; one backend test failed once on SQLite and could not be reproduced. They show the code does what its tests state, not production quality, security or capacity.
- *AdvancedRAG:* `FEATURE_MATRIX.md` states this beside every status.
- *Follow-up:* What would production evidence look like?
- *Avoid:* "all tests pass, so it is production-ready".

---

## How to use this bank

1. Cover the answer, speak yours aloud for 60–90 seconds, then compare.
2. Mark each question: *solid*, *shaky*, *missed*. Re-read the linked module for shaky ones.
3. For project questions, practise saying the evidence level: "the code does X (file), the test passed in my run / I haven't verified Y".
4. Coding questions live in [ADVANCED_RAG_CODING_PRACTICE.md](ADVANCED_RAG_CODING_PRACTICE.md); last-minute notes in [ADVANCED_RAG_QUICK_REVISION.md](ADVANCED_RAG_QUICK_REVISION.md).
