# Module 2: Embeddings and Retrieval

> Part of [ADVANCED_RAG_2026_MASTER_COURSE.md](../../ADVANCED_RAG_2026_MASTER_COURSE.md). Read [Module 1](MODULE_1_SYSTEM_ARCHITECTURE.md) first (you need the map).
> Evidence conventions are explained in the master course ("How to read the evidence labels"). Short version: **[Code-read]** = I read the source, **[Ran]** = I executed it in this session, **[Documented]** = a repo doc says so, I did not reproduce it, **[Missing]** = not implemented, **[Unknown]** = I could not determine it.

> **Update after the implementation pass (2026-10-23, working tree on top of `31bad7e`, uncommitted).** Statements below were first written against `31bad7e`; where the code changed they are corrected in place or listed here. All new evidence is hermetic (local vector store, mock gateway, throw-away PostgreSQL); nothing ran against Pinecone, the real LLM gateway or a deployment. Traceability: [FEATURE_MATRIX](../verification/FEATURE_MATRIX.md). Old line numbers have shifted; search by name.
>
> **What changed for this module:** retrieval now refuses documents indexed with a different embedding model/dimension (`EMBEDDING_MODEL_MISMATCH`, 409) and the vector store checks the index metric; `rag_core/evaluation.py` computes Hit@K, Precision@K, Recall@K, MRR, nDCG, Wilson and bootstrap intervals, AUC and a BM25 reference; a recorded comparison of dense, Postgres full-text, RRF, BM25 and local rerank is in `docs/verification/RETRIEVAL_EVALUATION.md`. Still true: the production keyword leg is Postgres full-text (`ts_rank_cd`), not BM25; pgvector is not used.

---

## 1. Learning objectives

After this module you can:

1. Explain what an embedding is, why similar meanings land near each other, and why that is *not* the same as keyword matching.
2. Compute cosine similarity, dot product and Euclidean distance by hand, and say when they give the same ranking.
3. Explain why the embedding model, the vector dimension and the index must match, and list the three places AdvancedRAG checks this, plus the one failure those checks cannot catch.
4. Describe dense, sparse (BM25) and hybrid retrieval, Reciprocal Rank Fusion, and cross-encoder reranking, and compute an RRF fusion by hand.
5. Compute Precision@K, Recall@K, Hit@K, MRR and nDCG by hand.
6. Trace the real query path in `rag-core/rag_core/pipeline.py` line by line.
7. Debug "the answer is in the document but retrieval did not find it".

## 2. Prerequisites

- Module 1 (architecture, where RAG Core sits).
- Dot product of two lists of numbers; square root; logarithm base 2 (all re-explained below).
- Python lists and dictionaries.

---

## 3. Fundamentals

### 3.1 What is an embedding?

A computer cannot compare the *meaning* of "How many days can I work from home?" and "Employees may work remotely up to three days per week" directly. Strings share almost no words ("home" vs "remotely").

An **embedding model** is a neural network that reads a piece of text and outputs a fixed-length list of numbers (a **vector**). It is trained (with contrastive learning on huge numbers of text pairs) so that texts with similar meaning produce vectors that point in similar directions, and unrelated texts produce vectors that point elsewhere.

In AdvancedRAG the vector has **384 numbers**, produced by the model `BAAI/bge-small-en-v1.5` running locally inside RAG Core. [Code-read: `rag-core/rag_core/config.py::Settings`, `embeddings.py`]. The Hugging Face model card confirms 384 dimensions and a 512-token input limit (source 1).

Think of it as GPS coordinates for meaning. Two nearby addresses are similar places. The difference: the "map" has 384 dimensions, and no single dimension has a human-readable meaning.

### 3.2 Lexical vs semantic similarity

| | Lexical (keyword) | Semantic (embedding) |
|---|---|---|
| Matches on | the same words (after stemming) | similar meaning |
| "work from my apartment" vs "work remotely" | weak or no match | strong match |
| Exact identifier `ZRy-SL18` | strong match | often weak (rare tokens carry little meaning) |
| Explainable? | yes: these words matched | no: a number came out |
| Needs training data? | no | yes (the model was pre-trained) |

The benchmark questions in `scripts/rag_benchmark.py` are deliberately built on this split: "Can I do my job from my apartment most of the week?" has no keyword overlap with "work remotely up to three days per week" (semantic wins), while "Which product has the SKU ZRy-SL18?" is an exact-code lookup (keyword wins). That is the entire motivation for **hybrid** search. [Code-read]

### 3.3 Similarity measures, with numbers

Take two tiny 2-D "embeddings": **a = [3, 4]** and **b = [4, 3]**.

| Measure | Formula | Value |
|---|---|---|
| Dot product | a·b = Σ aᵢbᵢ | 3·4 + 4·3 = **24** |
| Length (norm) | ‖a‖ = √(Σ aᵢ²) | √25 = **5** (same for b) |
| Cosine similarity | a·b / (‖a‖‖b‖) | 24 / 25 = **0.96** |
| Euclidean distance | √Σ (aᵢ−bᵢ)² | √(1+1) = **1.414** |

*Cosine* measures the **angle** between vectors and ignores length: range −1 (opposite) to 1 (same direction). *Dot product* mixes angle and length. *Euclidean distance* is straight-line distance.

**Normalization.** Divide each vector by its length so every vector has length 1 ("unit vector"): a′ = [0.6, 0.8], b′ = [0.8, 0.6]. Now:

- dot(a′, b′) = 0.48 + 0.48 = **0.96** = cosine. (For unit vectors, dot product *is* cosine.)
- ‖a′−b′‖² = 0.04 + 0.04 = **0.08** = 2 − 2·0.96. In general, for unit vectors **‖a−b‖² = 2 − 2·cos**. So Euclidean distance and cosine give the **same ranking**: smaller distance ⇔ larger cosine.

I verified these numbers by running Python in this session. [Ran]

AdvancedRAG L2-normalizes every vector it produces (`_normalize` in `embeddings.py::_normalize`), so on a cosine Pinecone index "cosine", "dot" and "Euclidean" would all rank identically. [Code-read] The Pinecone index metric (cosine) is *configured in Pinecone*, and the code never checks the metric, only the dimension. [Code-read; the docs say "dense, cosine" ([docs/rag-architecture.md §2](../rag-architecture.md)) [Documented]]

### 3.4 Interpreting similarity scores

A score is **not a probability** and has no universal threshold.

- The bge model card says that the similarity distribution of the BGE models "is about in the interval [0.6, 1]" and that a score above 0.5 does not by itself mean two sentences are similar (source 1). So absolute scores cluster high. Use scores to **rank**, not to decide "relevant or not".
- AdvancedRAG made this lesson concrete: with the reranker, a *correct* paraphrased answer scored **0.007** while an *on-topic but unanswerable* question scored **0.38**, so no threshold works; they chose **no score cut-off** and let the grounded prompt and grading decide. [Documented: `docs/rag-architecture.md §4`; the comment in `pipeline.py::retrieve` [Code-read]]

### 3.5 Dimension, embedding space and compatibility

- **Dimension** = how many numbers per vector (384 here). It is a property of the model.
- **Embedding space** = the "map" a particular model learned. Vectors from two different models are in *different spaces* even if both have 384 numbers. Comparing them is like comparing latitude from one map with a pixel position from another.
- Therefore **documents and queries must be embedded by the same model**, and the index must have the same dimension.

AdvancedRAG enforces dimension in **three places** [Code-read]:

| # | Check | Where |
|---|---|---|
| 1 | Configured `EMBEDDING_DIMENSIONS` (if set) must equal the model catalog's dimension | `embeddings.py::LocalEmbeddings.__init__` |
| 2 | Every embedding batch must return vectors of exactly that dimension | `embeddings.py::LocalEmbeddings._run` |
| 3 | The Pinecone index's `dimension` must equal the model's, checked when the data client is first created, and in `/ready` | `vectorstore.py::PineconeStore._client`, `:212-220` |

**What these checks cannot catch** (a classic interview trap): if someone swaps `bge-small-en-v1.5` for another *384-dimensional* model (many exist), all three checks pass and retrieval quietly becomes garbage, because old document vectors and new query vectors live in different spaces. The code stores `embedding_model` and `embedding_dimensions` per document (`db.py`) but `retrieve()` never compares them with the current model. [Code-read: no such comparison in `pipeline.retrieve`] So protection against this relies on the procedure in `docs/rag-architecture.md §6` (new index + full re-ingest), not on code. **Classification: partially implemented.**

### 3.6 Query vs passage embeddings (asymmetric search)

BGE models are trained so that a short *question* and a longer *passage* that answers it end up close. The recommended way is to prefix queries (not passages) with an instruction: "Represent this sentence for searching relevant passages:"; the model card says the instruction is optional for v1.5 and passages never need it (source 1). AdvancedRAG calls fastembed's `passage_embed` for chunks and `query_embed` for queries (`embeddings.py::LocalEmbeddings._run`). [Code-read] Whether fastembed adds exactly that prefix inside `query_embed` I did not inspect in the library source. [Unknown]

---

## 4. Technical deep dive

### 4.1 Exact vs approximate nearest-neighbour (ANN) search

Brute force compares the query with all N vectors: O(N·d). Fine for thousands, slow for millions. **ANN** indexes (HNSW graphs, IVF clusters) trade a little recall for big speed.

- AdvancedRAG's **dev/test-only** `LocalVectorStore` is brute force in Python (`vectorstore.py::LocalVectorStore`). It is refused in production by config validation (`config.py`). [Code-read]
- Production uses **Pinecone serverless**; the ANN algorithm is Pinecone's internal choice. [Documented]
- **pgvector** (PostgreSQL extension) offers `<->` L2, `<#>` negative inner product, `<=>` cosine distance, with HNSW and IVFFlat indexes; vector indexes support up to 2,000 dimensions (`halfvec` 4,000) (source 3). **AdvancedRAG does not use pgvector.** I found no `CREATE EXTENSION vector` or vector column anywhere in the RAG Core migrations; the only embedding column (`rag_chunks.embedding`, JSONB) is used by the dev-only local store. [Code-read: `alembic/versions/0001_rag_schema.py`, `db.py`]. Your brief asked to explain pgvector "if and where it is used": it is not used. It is the main *alternative* to discuss (see Q in Section 8).

Why does this matter in an interview? You may be asked "why not pgvector?" A balanced answer: pgvector would make the chunk row and its vector one transactional unit (no separate "verify the vectors were written" step, no orphan cleanup across systems) and removes a vendor; Pinecone gives a managed ANN service, per-namespace isolation, and hosted reranking inference. For ≤ millions of vectors, pgvector in the existing Postgres is a perfectly reasonable choice; the project's actual constraint is that the RAG DB is on Neon free (≈0.5 GB per README), which a vector index would eat quickly. [Reasoning, not a repo fact]

### 4.2 HNSW intuition (one paragraph)

HNSW (Hierarchical Navigable Small World) builds a layered graph. Upper layers are sparse "highways"; the bottom layer connects every vector to its near neighbours. A query starts at the top, greedily walks toward the query, then drops down a layer, and repeats. Parameters (`M`, `efConstruction`, `efSearch`) trade memory, build time and recall. You do not tune this in Pinecone serverless; you do in pgvector.

### 4.3 Sparse retrieval and BM25

A **sparse** vector has one dimension per vocabulary word, almost all zeros. Classic keyword scoring ranks documents by how often the query's words appear, discounting common words.

**BM25** (the standard formula):

score(D, Q) = Σ over query terms *t* of IDF(t) · [ f(t,D)·(k₁+1) ] / [ f(t,D) + k₁·(1 − b + b·|D|/avgdl) ]

- f(t,D) = how often *t* appears in D; IDF(t) = rarer words count more; |D|/avgdl = length normalization; typical k₁ ≈ 1.2–2.0, b ≈ 0.75.
- Two ideas to remember: **term-frequency saturation** (the 10th occurrence adds little) and **IDF** (a collection-level statistic).

**Important correction for AdvancedRAG:** the keyword half is **PostgreSQL full-text search, not BM25.** The code ranks with `ts_rank_cd` (cover density) over `to_tsvector('english', text)` [Code-read: `pipeline.py::_keyword`]. PostgreSQL's documentation states its ranking functions "do not use any global information", so there is no IDF (source 4). The code comment says "OR semantics ... like BM25"; that is an analogy about OR matching, not the same scoring. **Measured consequence (synthetic corpus, one run):** fusing dense with this Postgres full-text leg scored *worse* than dense alone (Hit@1 0.605 vs 0.754), while a BM25 reference fused with dense scored better in two of three chunking setups; my hypothesis is the missing IDF, untested (`docs/verification/RETRIEVAL_EVALUATION.md`). If an interviewer says "so you use BM25", correct them politely: "full-text search with `ts_rank_cd`; similar role, different formula; BM25 would need an extension or a search engine."

How the query is built: `plainto_tsquery('english', q)` parses text safely but ANDs the terms (`'fat' & 'rat'`, source 4). The code casts it to text and replaces `&` with `|` to get **OR** semantics, then runs `to_tsquery` on that, so a chunk matching *any* query term is a candidate, ranked by `ts_rank_cd`. A GIN index on `to_tsvector('english', text)` exists (`0001_rag_schema.py::upgrade`) matching that expression. [Code-read] Whether the planner actually uses the index for each filtered query is [Unknown] (needs `EXPLAIN ANALYZE` on real data).

Consequence of OR semantics: with a long question, common-ish words ("what", "the" are stop words and removed; "company", "employee" may remain) make many chunks match, and ranking quality depends on `ts_rank_cd` alone with no IDF. That is one reason the system has a reranker after fusion.

### 4.4 Hybrid search and Reciprocal Rank Fusion

Semantic scores (cosine 0.6–0.9) and keyword scores (`ts_rank_cd` around 0.0–1.0, a different scale) cannot be added meaningfully. **RRF** ignores scores and fuses **ranks**:

RRF(d) = Σ over rankings *r* of 1 / (k + rank_r(d)), rank starting at 1, **k = 60** (Cormack, Clarke & Büttcher, SIGIR 2009; I could not parse the PDF fetch, so this formula and constant are from my knowledge of that paper, not re-verified today).

Code: `pipeline.rrf` (`pipeline.py::rrf`) uses `enumerate` (0-based) and adds `+1`, i.e. 1-based. [Code-read]

**Worked example** (computed and checked by running Python [Ran]). Semantic ranking: A, B, C, D. Keyword ranking: C, A, E, B.

| Doc | Semantic rank | Keyword rank | RRF score |
|---|---|---|---|
| A | 1 | 2 | 1/61 + 1/62 = 0.016393 + 0.016129 = **0.032522** |
| C | 3 | 1 | 1/63 + 1/61 = 0.015873 + 0.016393 = **0.032266** |
| B | 2 | 4 | 1/62 + 1/64 = 0.016129 + 0.015625 = **0.031754** |
| E | – | 3 | 1/63 = **0.015873** |
| D | 4 | – | 1/64 = **0.015625** |

Fused order: A, C, B, E, D. Notice: a document that appears in *both* lists beats one that is first in only one list. Why k = 60? It damps the advantage of rank 1 over rank 2 (1/61 vs 1/62 is tiny), so no single retriever dominates and agreement matters.

### 4.5 Bi-encoders vs cross-encoders (reranking)

- **Bi-encoder** (the embedding model): encodes query and passage **separately**, then compares vectors. Fast; passage vectors can be precomputed and indexed. Less precise because query and passage never "see" each other.
- **Cross-encoder** (a reranker): feeds the query and the passage **together** into one transformer that outputs a relevance score. More accurate, but must run once per (query, passage) pair, so it can only rescore a short list (here ≤ 20).

Pipeline pattern: *cheap, high-recall retrieval → expensive, high-precision rerank.* AdvancedRAG: top 20 semantic + top 20 keyword → RRF → top 20 → Pinecone-hosted `bge-reranker-v2-m3` → top 6. [Code-read: `pipeline.py::retrieve`, `rerank.py`]. Pinecone documents this model as accepting up to 1,024 tokens per query-document pair and 100 documents per request; `truncate` defaults to `NONE` (error on overflow) and the code passes `END` (truncate silently) (source 2; `rerank.py::PineconeReranker.rerank`). Since a chunk is ≤ ~1,400 characters, overflow is unlikely for normal prose. Dense numbers/tables tokenize to more tokens per character, which is why `fit_to_token_limit` exists for the *embedding* model (Module 3).

Failure policy: if the reranker raises, `retrieve` catches it, keeps the RRF order and records `reranker: "skipped"` (`pipeline.py::retrieve`). [Code-read]

### 4.6 Top-k and metadata filtering

- **Top-k**: how many results to return. AdvancedRAG: 20 candidates per retriever, then final **6** (`RETRIEVAL_CANDIDATES`, `RETRIEVAL_TOP_K`). Too small k loses recall; too large k wastes context and distracts the LLM. [Code-read: `config.py`]
- **Metadata filter**: restrict the search by attributes. The Pinecone query sends `filter: {"document_id": {"$in": [...]}}` and `namespace: user-<id>` [Code-read: `vectorstore.py::PineconeStore.query`]. The namespace gives hard per-user isolation; the filter narrows to the documents in scope (e.g., one named file).
- Metadata stored per vector: only `document_id`, `chunk_index`, `page`. **No text.** Query uses `includeMetadata: False`; the text is fetched from PostgreSQL by chunk id afterwards (`pipeline.py::retrieve`). [Code-read]
- Filtering can interact with ANN recall (pre- vs post-filtering): in some systems a very selective filter can return fewer than k results. Pinecone's exact behaviour here is [Unknown] from this repo; test with a selective filter if it matters.

### 4.7 Retrieval metrics (formulas)

Let the system return a ranked list. For each question, you have **relevance labels**: which chunks (or documents) are actually relevant.

| Metric | Definition | Question it answers |
|---|---|---|
| **Hit@K** | 1 if at least one relevant item is in the top K, else 0; average over questions | "Is the answer anywhere in the first K?" |
| **Precision@K** | (# relevant in top K) / K | "How much of what I showed is useful?" |
| **Recall@K** | (# relevant in top K) / (# relevant in total) | "How much of the useful material did I find?" |
| **MRR** | mean over questions of 1/rank of the first relevant item (0 if none) | "How early is the first good result?" |
| **nDCG@K** | DCG@K / IDCG@K, DCG = Σᵢ gain(relᵢ) / log₂(i+1), i from 1 | "Is the ranking good when relevance has grades?" |

**Worked example, Precision/Recall.** Three relevant chunks {a, b, c}; system returns [a, x, b, y, z] [Ran]:

| K | Relevant in top K | Precision@K | Recall@K |
|---|---|---|---|
| 1 | 1 | 1.00 | 0.333 |
| 3 | 2 | 0.667 | 0.667 |
| 5 | 2 | 0.40 | 0.667 |

Precision falls as K grows; recall never falls. That trade-off is why you choose K.

**Worked example, MRR.** Four questions, rank of first relevant result: 1, 3, none, 2. Reciprocal ranks: 1, 0.333, 0, 0.5 → MRR = (1 + 0.333 + 0 + 0.5)/4 = **0.458**. [Ran]

**Worked example, nDCG.** Graded relevance of the returned list: [3, 2, 0, 1]. DCG = 3/log₂2 + 2/log₂3 + 0/log₂4 + 1/log₂5 = 3 + 1.2619 + 0 + 0.4307 = **4.6925**. The ideal order [3, 2, 1, 0] gives IDCG = 3 + 1.2619 + 0.5 + 0 = **4.7619**. nDCG = 4.6925 / 4.7619 = **0.9854**. [Ran] (Some libraries use gain 2^rel − 1 instead of rel; that variant gives 0.9926 for this list [Ran]. Always state which one you use.)

**Why the log?** Positions further down matter less; dividing by log₂(i+1) discounts them smoothly.

**Connection to the project.** `scripts/rag_benchmark.py` computes Hit@1, Hit@6 and MRR per stage (semantic, keyword, RRF, final). It does **not** compute Precision@K, Recall@K or nDCG; the new `scripts/run_retrieval_eval.py` does (with confidence intervals), on a synthetic corpus. Module 4 explains what each does and does not prove.

### 4.8 Failure modes in retrieval

| Symptom | Likely cause | How to confirm |
|---|---|---|
| Answer exists in the document but is never retrieved | chunk boundary split the fact; wrong model for the language; query wording far from passage | print the chunk containing the fact; embed query and chunk; compare cosine |
| Exact codes (SKU, error code) not found | semantic-only retrieval | check that keyword results contain it |
| Irrelevant chunks outrank relevant ones | near-duplicate documents (Aurora X1 vs X2), generic query | look at top-20 before and after rerank |
| Everything scores 0.7–0.8 | model's compressed similarity range | compare relative ranks, not absolute scores |
| Zero results | document not `ready`, wrong namespace/user, filter excludes it | check `rag_documents`, Pinecone namespace count |
| Vector DB error "dimension mismatch" | index created with another dimension | `/ready` shows `DIMENSION_MISMATCH: index has X, embedding model produces 384` |
| Silent garbage after "upgrading" the model | same-dimension model swap (§3.5) | compare `rag_documents.embedding_model` with current model |
| Retrieval fine but answer wrong | generation/grading problem, not retrieval | check `evidence_grade`, `verification_status` in message metadata (Module 5) |

---

## 5. AdvancedRAG implementation: the actual query path

All in `rag-core/rag_core/pipeline.py`, function `retrieve` (lines 260-331). [Code-read]

```text
POST /api/v1/rag/retrieve  {user_id, query, document_ids, top_k?}      main.py
  1. indexed = documents of this user that exist in rag_documents       pipeline.py::retrieve
        none -> return empty chunks (no error)                           :271-273
  2. query_vector = embed([query], QUERY)[0]    (local ONNX, 384-d, unit length)      :275
  3. semantic = Pinecone.query(namespace user-<id>, top 20, filter doc ids)           :277
        retryable vector-store error -> semantic = [], strategy = keyword_only_degraded   :278-284
  4. keyword  = Postgres FTS top 20 (OR query, ts_rank_cd, scoped to user+docs)       :286
        both empty AND degraded -> raise vector_store_unavailable                      :287-288
  5. fused = rrf([semantic ids, keyword ids], k=60)[:20]                               :289
  6. rows = SELECT chunks WHERE id IN fused AND user_id = user                         :292
  7. ranked = reranker.rerank(query, [(id, text)...], top_n=6)                          :297
        exception -> RRF order, reranker="skipped"                                     :303-307
  8. build context: "[n] filename (page p)\n<text>" until 12,000 chars (first chunk always kept)   :311-319
  9. return {chunks[], context, metadata{strategy, latencies, counts}}                 :331
```

Notes worth knowing for an interview:

- **Ownership is re-checked at three layers**: step 1 (only the user's indexed documents), the Pinecone namespace (`user-<id>`), and the SQL `user_id = :user` filters in steps 4 and 6. [Code-read]
- Agent Core calls this from `Orchestrator._search` (`orchestrator.py::Orchestrator._search`) with `top_k = RETRIEVAL_TOP_K`.
- **Query rewriting** (follow-ups) and **evidence grading** happen in Agent Core, around this call. Rewrite: `resolve_search_query` asks a fast model to turn "And what about its operating temperature?" into a standalone query, then a deterministic guard (`rejection_reason`) rejects rewrites that introduce an entity not in the conversation, drop the subject, are too long, or look like "meta text"; on rejection it falls back to attaching the subject to the message (`query_resolution.py`). Grading is covered in Module 5. [Code-read]
- Settings that define retrieval (all frozen, guarded by `test_frozen_contract.py`): model `BAAI/bge-small-en-v1.5`, 384 d, `RETRIEVAL_CANDIDATES=20`, `RRF_K=60`, `RETRIEVAL_TOP_K=6`, `MAX_CONTEXT_CHARS=12000`, reranker `bge-reranker-v2-m3`. [Code-read: `config.py`; Documented: `docs/rag-architecture.md §2`]

### Implementation status table

| Capability | Status | Evidence |
|---|---|---|
| Local embeddings, 384-d, unit-normalized | Implemented; covered by tests | [Ran] the rag-core suite in this session: 87 passed, 4 skipped (which 4 were skipped and why I did not inspect). Tests include `test_local_model_dimension_and_query_passage_consistency` |
| Dimension guard (3 places) | Implemented | Code-read |
| Same-dimension model swap detection | **Implemented** (model name + dimension compared on read; mismatches excluded/refused; `/ready` reports it; reindex endpoint repairs) | `pipeline.retrieve`, `test_integrity_and_spool.py` [Ran] |
| Pinecone dense retrieval with namespace + document filter | Implemented; verified end to end only by the author's benchmark run | [Documented] |
| Pinecone metric = cosine | **Checked by code** at startup/ready (`vectorstore.py`); the check was tested against the Pinecone-emulating server, not a real index | `test_integrity_and_spool.py` |
| pgvector | **Not used** | no vector column/extension in migrations |
| BM25 | **Not used** (Postgres FTS `ts_rank_cd` instead) | `pipeline.py::_keyword` |
| Hybrid with RRF k=60 | Implemented | `pipeline.rrf`; `test_rrf_fuses_rankings` passed. **The PostgreSQL full-text branch (`_keyword` on Postgres) is covered only by a test marked `skipif(not IS_PG)`; my run used SQLite, so that test was skipped and the Postgres keyword path is not verified by me** (the SQLite fallback does term-overlap scoring instead) |
| Cross-encoder rerank (Pinecone hosted) with fallback | Implemented; hosted call not exercised by me | Code-read; mock Pinecone used in tests |
| Query rewriting with entity guard | Implemented | `query_resolution.py`; `agent-core/tests/test_follow_up_resolution.py` |
| Evidence grading + bounded correction | Implemented | `grading.py`, `orchestrator._retrieve_and_grade` |
| Precision@K / Recall@K / nDCG in repo | **Implemented** in `rag_core/evaluation.py` and `scripts/run_retrieval_eval.py` (the older `rag_benchmark.py` still reports only hit@1, hit@6, MRR) | `test_evaluation.py` [Ran] |
| Score threshold for "not relevant" | Deliberately **not** implemented | `pipeline.py::retrieve` |

---

## 6. Worked examples (runnable)

### 6.1 Similarity measures and the unit-vector identity

```python
# runnable
import math

def dot(a, b): return sum(x * y for x, y in zip(a, b))
def norm(a): return math.sqrt(dot(a, a))
def cosine(a, b): return dot(a, b) / (norm(a) * norm(b))
def euclid(a, b): return math.dist(a, b)
def unit(a): n = norm(a); return [x / n for x in a]

a, b = [3, 4], [4, 3]
assert dot(a, b) == 24
assert abs(cosine(a, b) - 0.96) < 1e-12
assert abs(euclid(a, b) - math.sqrt(2)) < 1e-12
ua, ub = unit(a), unit(b)
assert abs(dot(ua, ub) - 0.96) < 1e-12          # unit vectors: dot == cosine
assert abs(euclid(ua, ub) ** 2 - (2 - 2 * 0.96)) < 1e-12   # ||a-b||^2 = 2 - 2cos
print("ok")
```

### 6.2 RRF

```python
# runnable
def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for i, doc in enumerate(ranking, start=1):       # 1-based rank
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + i)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

fused = rrf([list("ABCD"), list("CAEB")])
assert [d for d, _ in fused] == list("ACBED")
assert abs(fused[0][1] - (1/61 + 1/62)) < 1e-12
print(fused)
```

### 6.3 Metrics

```python
# runnable
import math

def precision_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / k

def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / len(relevant) if relevant else 0.0

def hit_at_k(ranked, relevant, k):
    return 1.0 if set(ranked[:k]) & set(relevant) else 0.0

def reciprocal_rank(ranked, relevant):
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0

def dcg(gains):
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))

def ndcg_at_k(gains, k):
    ideal = sorted(gains, reverse=True)
    best = dcg(ideal[:k])
    return dcg(gains[:k]) / best if best else 0.0

ranked, rel = ["a", "x", "b", "y", "z"], {"a", "b", "c"}
assert precision_at_k(ranked, rel, 3) == 2 / 3
assert abs(recall_at_k(ranked, rel, 5) - 2 / 3) < 1e-12
assert hit_at_k(ranked, rel, 1) == 1.0
mrr = sum(reciprocal_rank(r, s) for r, s in [(["q", "a"], {"a"}), (["x"], {"a"})]) / 2
assert mrr == 0.25
assert abs(ndcg_at_k([3, 2, 0, 1], 4) - 0.98544) < 1e-4
print("ok")
```

### 6.4 A tiny end-to-end hybrid retriever (educational, not the production code)

```python
# runnable
import math, re
from collections import Counter

DOCS = {
    "d1": "Employees may work remotely up to three days per week",
    "d2": "Hotel stays are reimbursed up to 180 euros per night",
    "d3": "SKU ZRy-SL18 is the Zephyr Relay",
}
STOP = {"the", "a", "an", "of", "to", "is", "are", "up", "per", "can", "i", "my", "from", "do", "how"}

def tokens(text):
    return [t for t in re.findall(r"[a-z0-9\-]+", text.lower()) if t not in STOP]

# --- sparse: BM25 with IDF (the real formula; Postgres FTS does NOT do this)
def bm25_rank(query, docs, k1=1.5, b=0.75):
    toks = {d: tokens(t) for d, t in docs.items()}
    avgdl = sum(len(t) for t in toks.values()) / len(toks)
    df = Counter(w for t in toks.values() for w in set(t))
    N = len(docs)
    scores = {}
    for d, t in toks.items():
        tf = Counter(t)
        s = 0.0
        for q in tokens(query):
            if q in tf:
                idf = math.log(1 + (N - df[q] + 0.5) / (df[q] + 0.5))
                s += idf * tf[q] * (k1 + 1) / (tf[q] + k1 * (1 - b + b * len(t) / avgdl))
        scores[d] = s
    return [d for d, s in sorted(scores.items(), key=lambda kv: -kv[1]) if s > 0]

# --- "dense": a stand-in using hand-made concept vectors (a real model learns these)
CONCEPTS = {"remote": 0, "home": 0, "remotely": 0, "apartment": 0, "hotel": 1, "stay": 1, "stays": 1, "night": 1}
def embed(text):
    v = [0.0, 0.0, 0.0]
    for w in tokens(text):
        if w in CONCEPTS:
            v[CONCEPTS[w]] += 1
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]

def dense_rank(query, docs):
    q = embed(query)
    scored = {d: sum(a * b for a, b in zip(q, embed(t))) for d, t in docs.items()}
    return [d for d, s in sorted(scored.items(), key=lambda kv: -kv[1]) if s > 0]

def rrf(rankings, k=60):
    sc = {}
    for r in rankings:
        for i, d in enumerate(r, 1):
            sc[d] = sc.get(d, 0) + 1 / (k + i)
    return [d for d, _ in sorted(sc.items(), key=lambda kv: -kv[1])]

# paraphrase with no shared content words: dense finds it, sparse does not
Q = "do my job from my apartment"
assert bm25_rank(Q, DOCS) == []
assert dense_rank(Q, DOCS)[0] == "d1"
# exact code: sparse finds it, dense does not
assert bm25_rank("ZRy-SL18", DOCS)[0] == "d3"
assert dense_rank("ZRy-SL18", DOCS) == []
# hybrid gets both
assert rrf([dense_rank(Q, DOCS), bm25_rank(Q, DOCS)])[0] == "d1"
assert rrf([dense_rank("ZRy-SL18", DOCS), bm25_rank("ZRy-SL18", DOCS)])[0] == "d3"
print("ok")
```

---

## 7. Common mistakes

1. "Cosine similarity of 0.8 means 80% relevant." No: scores are model-specific rankings aids; bge scores compress into roughly 0.6–1.
2. Comparing vectors from different models, or embedding queries and documents with different models.
3. Believing a matching **dimension** proves a matching **model**.
4. Saying the system uses BM25 or pgvector. It uses PostgreSQL full-text (`ts_rank_cd`) and Pinecone.
5. Fusing by adding raw scores of different scales instead of ranks.
6. Treating Hit@K as quality of the *answer*; it only says the evidence was in the top K.
7. Reporting MRR from a handful of questions as if it were precise (Module 4).
8. Putting chunk text in vector metadata "for convenience" (bloats the index, complicates deletion; this project deliberately avoided it).
9. Forgetting that the reranker only reorders the candidate list: if the right chunk is not in the 20 candidates, no reranker can recover it.
10. Using a score threshold to detect unanswerable questions, which the project measured not to work.

---

## 8. Interview questions

### Beginner

**B1. What is an embedding?**
A fixed-length numeric vector produced by a neural network so that texts with similar meaning have vectors close together. Here: 384 numbers from `bge-small-en-v1.5`, computed locally.

**B2. Why embeddings instead of keyword search?**
They match meaning ("work from my apartment" ↔ "work remotely"). But they are weak on exact identifiers, so AdvancedRAG also runs keyword search and fuses the two.

**B3. What is cosine similarity?**
The cosine of the angle between two vectors: dot product divided by the product of lengths. 1 = same direction, 0 = unrelated (orthogonal), −1 = opposite. It ignores vector length.

**B4. What does "top-k" mean?**
Return the k best-ranked results. AdvancedRAG retrieves 20 candidates per retriever and keeps the final 6.

**B5. Why must query and document embeddings come from the same model?**
Each model defines its own space. Distances across spaces are meaningless.

### Intermediate

**I1. Dot product vs cosine vs Euclidean: when do they agree?**
When vectors are unit length: dot = cosine, and ‖a−b‖² = 2 − 2cos, so all three rank identically. AdvancedRAG normalizes all vectors (`_normalize`).

**I2. Why hybrid search? Give a concrete case from the project.**
Paraphrase questions ("job from my apartment") favour dense; exact codes ("SKU ZRy-SL18") favour keyword. The benchmark documents semantic-only 18/20 Hit@1 and keyword-only 15/20, with the final rerank at 19/20 [Documented].

**I3. How does RRF work, and why k = 60?**
Sum of 1/(k + rank) over rankings. Rank-based, so no score normalization; k damps the dominance of top ranks so documents ranked well by both retrievers win. 60 is the value from the original paper and a common default.

**I4. Bi-encoder vs cross-encoder?**
Bi-encoder embeds separately (fast, indexable). Cross-encoder scores the pair jointly (accurate, slow, only for a short list). Use bi-encoder for recall, cross-encoder for precision.

**I5. Is the project's keyword search BM25?**
No. PostgreSQL FTS with `ts_rank_cd` (cover density), OR-ed terms, GIN index. PostgreSQL ranking uses no collection-wide statistics, so no IDF. BM25 would need an extension or search engine.

**I6. Does the project use pgvector?**
No. Vectors are in Pinecone (namespace per user). Chunk text and the FTS index are in PostgreSQL. A local brute-force store exists for tests/dev only and is refused in production config.

**I7. What happens if Pinecone is down at query time?**
Retrieval degrades to keyword-only (`retrieval_strategy: keyword_only_degraded`); permanent errors are raised; if both are empty while degraded it raises `vector_store_unavailable`.

**I8. Why no relevance score threshold?**
The authors measured a correct answer with reranker score 0.007 and an unanswerable on-topic question at 0.38, so any threshold removes real answers. Abstention is decided by grading and the grounded prompt.

### Advanced

**A1. Someone changes `EMBEDDING_MODEL` to another 384-d model. What happens and how do you prevent it?**
No error (dimension still matches); new query vectors are in a different space from stored document vectors; retrieval degrades silently. Prevention: store `embedding_model` per document (already done) and add a check in `retrieve`/`/ready` that every indexed document's model equals the current one, fail loudly, and require a new index plus full re-ingest. Today it is procedural only.

**A2. Design an offline retrieval evaluation for this system.**
Labeled set of (question, relevant chunk/document ids) written independently of the system; metrics Hit@K, Recall@K, MRR, nDCG per stage; sample size with confidence intervals; separate answerable/unanswerable; frozen corpus; fixed seeds; report run metadata (model, index, commit). See Module 4.

**A3. The right chunk is rank 8 in dense, rank 15 in keyword. Will the final top 6 contain it?**
RRF scores: 1/68 + 1/75 = 0.01471 + 0.01333 = 0.02804. Its fused rank depends on others (a doc at rank 1 in one list only scores 0.0164, so this chunk, present in both, likely ranks in the top few). Then the reranker decides the final 6; it can still demote it. Hybrid agreement is a strong signal; rerank is the last judge.

**A4. When would you choose pgvector over Pinecone here?**
Corpus in the low millions of vectors, desire for transactional consistency between chunk rows and vectors, fewer vendors, SQL joins with metadata, and an existing Postgres with enough storage. Choose Pinecone for managed scale, isolation by namespace, hosted inference and no index tuning. Hybrid in one query (pgvector + FTS) is possible with SQL.

**A5. How would you detect retrieval regressions after changing chunk size?**
Frozen question set + labels keyed to *content* (phrases), not chunk ids (ids change when chunking changes); compare Hit@K/MRR per stage before/after; the repo's benchmark labels by "filename + required phrases" for exactly this reason [Code-read: `rag_benchmark.py::main.rank`].

---

## 9. Follow-up questions

- Why is `includeMetadata` false in the query? (Text lives in PostgreSQL; saves bandwidth.)
- What happens to ranking if the query is very long? (Embedding truncates at 512 tokens; the keyword OR query matches broadly.)
- Why does the code cap fused candidates at 20 before reranking? (Cost/latency; reranker accepts ≤ 100 docs per Pinecone docs.)
- Is cosine similarity symmetric? Is the retrieval task symmetric? (Cosine is; query→passage retrieval is asymmetric, hence the query instruction.)
- How would you add a metadata filter for "documents uploaded this month"? (Add a field to vector metadata at ingestion and to the filter; requires re-ingest or a Postgres pre-filter of document ids.)
- Why does a near-duplicate document (Aurora X1/X2) hurt top-1 but not hit@6?
- What does recall@20 of the candidate stage bound? (The ceiling for everything after it.)

## 10. Practical exercises

> Educational code. The production implementations are `pipeline.rrf`, `PineconeStore.query`, `_keyword` in `rag-core/rag_core/`.

### E2.1 Implement cosine top-k (brute force)

*Problem.* Given a query vector and a dict of id→vector, return the ids of the k most similar by cosine, best first.
*Expected.* `top_k([1,0], {"a":[1,0],"b":[0,1],"c":[1,1]}, 2) == ["a","c"]`.
*Hint.* Handle the zero vector.

```python
# runnable
import heapq, math

def cosine(a, b):
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)

def top_k(query, vectors, k):
    scored = ((cosine(query, v), vid) for vid, v in vectors.items())
    return [vid for _, vid in heapq.nlargest(k, scored)]

assert top_k([1, 0], {"a": [1, 0], "b": [0, 1], "c": [1, 1]}, 2) == ["a", "c"]
assert top_k([0, 0], {"a": [1, 0]}, 1) == ["a"]   # zero query: all scores 0, still returns something
print("ok")
```

*Complexity.* O(N·d) time for scoring plus O(N log k) for the heap; O(k) extra space. *Interview follow-up:* "How do you make this faster for 10 million vectors?" (ANN index such as HNSW; quantization; sharding.)

### E2.2 Implement RRF with ties and weights

Extend `rrf` with per-ranking weights and deterministic tie-breaking (by first appearance). Solution:

```python
# runnable
def weighted_rrf(rankings, weights=None, k=60):
    weights = weights or [1.0] * len(rankings)
    scores, first_seen = {}, {}
    for w, ranking in zip(weights, rankings):
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank)
            first_seen.setdefault(doc, len(first_seen))
    return sorted(scores, key=lambda d: (-scores[d], first_seen[d]))

assert weighted_rrf([["a", "b"], ["b", "a"]]) == ["a", "b"]            # tie -> first seen
assert weighted_rrf([["a", "b"], ["b", "a"]], weights=[1, 2]) == ["b", "a"]
print("ok")
```

### E2.3 Debug: dimension mismatch

*Problem.* A teammate sets `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` (384-d) and reports that startup is fine but answers are nonsense. Explain and write a guard.

*Answer.* All dimension checks pass; spaces differ. Guard: record the model name in each indexed document (already stored) and refuse to query if any indexed document's model differs:

```python
# runnable
def assert_same_model(indexed_models, current_model):
    stale = {m for m in indexed_models if m != current_model}
    if stale:
        raise RuntimeError(f"indexed with {sorted(stale)}, serving with {current_model}: re-ingest into a new index")

assert_same_model(["bge-small", "bge-small"], "bge-small")
try:
    assert_same_model(["bge-small"], "minilm")
except RuntimeError as e:
    assert "re-ingest" in str(e)
else:
    raise AssertionError("expected failure")
print("ok")
```

### E2.4 Troubleshooting scenarios (no code)

1. *"Retrieval returns nothing for a document marked `ready`."* Check, in order: `rag_documents` has the id for that user; `rag_chunks` count; Pinecone namespace `user-<id>` vector count; the document id is in the request's `document_ids`; `classify` narrowed scope to a named file; grading marked it `insufficient` (retrieval fine, answer abstained).
2. *"After deploying, `/ready` says `DIMENSION_MISMATCH: index has 768, embedding model produces 384`."* The Pinecone index belongs to an older model (the repo notes an older 1024-d index). Create a 384-d cosine index, update `PINECONE_INDEX`, re-ingest.
3. *"Exact error codes are never found."* Verify the keyword path returns them (`_keyword`); check OR query construction; check the hybrid strategy is not `semantic only`.
4. *"Top-1 is often the wrong sibling document."* Look at both siblings in top-6; consider metadata in the chunk header (filename in text), a named-document scope, or reranker tuning.

## 11. Revision notes

- Embedding = text → 384 floats (bge-small-en-v1.5, local ONNX, unit-normalized). Same model for query and passage; query vs passage entry points differ.
- cos = a·b/(‖a‖‖b‖). Unit vectors: dot = cos, ‖a−b‖² = 2−2cos.
- Three dimension checks (config, batch, index). **Cannot** catch same-dimension model swap.
- Dense: Pinecone (cosine, namespace `user-<id>`, filter `document_id $in`, top 20). Keyword: Postgres FTS `ts_rank_cd`, OR semantics, top 20 (not BM25). pgvector: not used.
- RRF: Σ 1/(60+rank). Rerank: Pinecone `bge-reranker-v2-m3`, top 6; failure → RRF order.
- Context ≤ 12,000 chars; chunk text from PostgreSQL; Pinecone holds vectors + 3 metadata fields.
- Metrics: Hit@K, P@K, R@K, MRR, nDCG formulas; the repo computes hit@1, hit@6, MRR only.
- No score threshold (0.007 vs 0.38 evidence).

## 12. Assessment (attempt before reading the answer key)

1. Compute cosine similarity, dot product and Euclidean distance for a = [1, 2, 2] and b = [2, 1, 2]. Then normalize both and verify ‖a′−b′‖² = 2 − 2cos.
2. Fuse these rankings with RRF (k = 60): semantic [P, Q, R], keyword [R, P, S]. Show the scores and final order.
3. Relevant set {1, 4, 7}. System returns [4, 2, 1, 9, 7]. Compute P@3, R@3, P@5, R@5, and the reciprocal rank.
4. Graded relevance of returned list = [0, 3, 1, 2]. Compute nDCG@4 (linear gain).
5. List the three dimension checks in AdvancedRAG and give one failure none of them catches.
6. Is AdvancedRAG's keyword search BM25? Justify with code and docs.
7. Does AdvancedRAG use pgvector? What stores which data (Pinecone vs PostgreSQL)?
8. Why does the project not use a reranker score threshold?
9. Explain what happens, step by step, in `retrieve` if Pinecone times out.
10. Design a test that detects a same-dimension model swap.
11. A user asks "What is the maximum nightly hotel rate?" and the correct chunk is rank 12 in dense and absent from keyword. Will the final context contain it? What decides?
12. Explain why bi-encoders cannot replace cross-encoders for the final ranking, and why cross-encoders cannot replace bi-encoders for first-stage retrieval.

---

## 13. ANSWER KEY (do not read until you have attempted the assessment)

**A1.** dot = 1·2 + 2·1 + 2·2 = 8. ‖a‖ = √9 = 3, ‖b‖ = √9 = 3. cos = 8/9 = 0.8889. Euclid = √(1 + 1 + 0) = √2 = 1.4142. Normalized: a′ = [1/3, 2/3, 2/3], b′ = [2/3, 1/3, 2/3]; ‖a′−b′‖² = (1/9 + 1/9 + 0) = 2/9 = 0.2222; 2 − 2·(8/9) = 2 − 16/9 = 2/9 ✓. *Misconception:* using unnormalized dot (8) as "similarity" and comparing across vectors of different lengths.

**A2.** P: semantic 1, keyword 2 → 1/61 + 1/62 = 0.032522. Q: semantic 2 → 1/62 = 0.016129. R: semantic 3, keyword 1 → 1/63 + 1/61 = 0.032266. S: keyword 3 → 1/63 = 0.015873. Order: P, R, Q, S. *Follow-up:* what if R were rank 1 in semantic? Then R = 1/61 + 1/61 = 0.032787 > P.

**A3.** Returned [4, 2, 1, 9, 7]. Top 3 = {4, 2, 1}: relevant 4 and 1 → P@3 = 2/3, R@3 = 2/3. Top 5 relevant: 4, 1, 7 → P@5 = 3/5 = 0.6, R@5 = 3/3 = 1.0. First relevant at rank 1 → RR = 1. *Misconception:* dividing recall by K instead of the number of relevant items.

**A4.** DCG = 0/log₂2 + 3/log₂3 + 1/log₂4 + 2/log₂5 = 0 + 1.8928 + 0.5 + 0.8614 = 3.2542. Ideal [3, 2, 1, 0]: 3 + 2/1.585 + 1/2 + 0 = 3 + 1.2619 + 0.5 = 4.7619. nDCG = 3.2542/4.7619 = 0.683. (Computed by hand from the formula; the same helper function in §6.3 reproduces the formula and I verified its 0.9854 example by running it.)

**A5.** (1) `EMBEDDING_DIMENSIONS` vs model catalog dimension; (2) per-batch output dimension; (3) Pinecone index dimension on first use and in `/ready`. None catches swapping to a different 384-d model (or changing the Pinecone *metric*, which is never read).

**A6.** No. `ts_rank_cd` over `to_tsvector('english', text)` with a GIN index (`pipeline.py::_keyword`, migration 0001). PostgreSQL ranking has no IDF/global statistics (PostgreSQL docs). The code comment "like BM25" refers to OR semantics only.

**A7.** No pgvector. Pinecone: one 384-d vector per chunk, id `<document_id>:<n>`, namespace `user-<id>`, metadata `document_id`, `chunk_index`, `page`. PostgreSQL (RAG DB): chunk text, FTS index, document records, ingestion lock, CSV tables. A JSONB `embedding` column exists but is only used by the dev/test local store.

**A8.** Measured: a correct paraphrase scored 0.007 while an unanswerable on-topic question scored 0.38 (docs/rag-architecture.md §4, [Documented]); any cut-off drops real answers. Abstention is decided by evidence grading and the prompt. *Follow-up:* one measurement is not proof for all corpora; it is a reason, not a law.

**A9.** `svc.vectors.query` raises a retryable `ServiceError`; `retrieve` catches it, logs `vector_search_failed_using_keyword_only`, sets `semantic=[]`, strategy `keyword_only_degraded`; keyword search runs; if both lists are empty it raises `vector_store_unavailable`; otherwise RRF over the single list, rerank, context. Note that Pinecone `query` is **not retried** (only upsert/fetch/delete are), per the docs.

**A10.** Unit test: create an indexed document row with `embedding_model="old"`, set current provider model to `"new"` with the same dimension, call `/ready` or `retrieve`, expect a loud error. This test now exists and passes (`test_documents_of_another_embedding_model_are_refused_loudly`). Also test the happy path where both match, and the repair path (`/reindex`).

**A11.** Dense rank 12 only: RRF = 1/72 = 0.01389. It enters the fused top 20 only if enough other chunks score lower; with a 20-item semantic list and 20-item keyword list, a chunk present in just one list at rank 12 will probably survive the cut to 20 (union ≤ 40, cut at 20) but sit mid-table. The reranker then scores the 20 and returns 6; because a cross-encoder reads query+text jointly, it can lift a good rank-12 chunk to the top 6. So: candidate generation must include it (recall), the reranker decides the final 6 (precision). *Misconception:* believing the reranker "searches" beyond the candidates.

**A12.** Bi-encoder alone ignores query–passage interaction (less precise), so use a cross-encoder to refine. A cross-encoder needs to run per pair: for N = 1 million chunks that is 1 million model runs per query, infeasible, so use bi-encoder vectors and an ANN index for the first stage.

---

## 14. Completion criteria (self-check; only you can mark this)

- [ ] You computed A1–A4 by hand before looking.
- [ ] You can explain the §5 pipeline on a whiteboard without notes.
- [ ] You can say, without hedging, that the project uses Postgres FTS, not BM25, and not pgvector.
- [ ] You can name the same-dimension model-swap gap and the fix.
- [ ] You ran the runnable blocks and they printed `ok`.

## Sources (accessed 2026-10-23)

1. BAAI/bge-small-en-v1.5 model card, Hugging Face: https://huggingface.co/BAAI/bge-small-en-v1.5
2. Pinecone, "Rerank results": https://docs.pinecone.io/guides/search/rerank-results
3. pgvector README: https://github.com/pgvector/pgvector
4. PostgreSQL documentation, "Controlling Text Search" (ranking): https://www.postgresql.org/docs/current/textsearch-controls.html
5. Cormack, Clarke, Büttcher, "Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods", SIGIR 2009 (formula and k = 60 recalled from the paper; the PDF could not be parsed in this session): https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
