# Retrieval evaluation — recorded runs and the decision they led to

Raw results: `evaluation/results/*.json`. The numbers below are from `20261023-212453-31bad7e-v12.json` (dataset **v1.2**, commit `31bad7e` plus the uncommitted working tree).
Reproduce: `python evaluation/build_dataset.py` then
`python scripts/run_retrieval_eval.py --db <disposable pg url containing "eval"> --strategies recursive,structure,semantic --experiments --rerank-local --ocr`.
Paired comparisons: `python scripts/compare_runs.py --a RUN --a-method rerank_local --b RUN --b-method rerank_local_idf [--types ...] [--split ...]`.
Everything uses an isolated development database and the in-database vector store; no production index, namespace or data was touched.

## What was measured, and how

- **Corpus (synthetic, regenerable byte for byte):** 15 content files (handbook, runbook, catalog, product pages as Markdown/DOCX, CSV price list, meeting notes, a text PDF with a paragraph split across a page break, a scanned PDF, and three **PDF table documents**: bordered, borderless and a two-page table) plus 60 distractor notes. Dense (bge-small-en-v1.5, 384-d, unchanged) + PostgreSQL full-text + RRF (k=60) exactly as in production code; reranker = a local cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`) as an **indicative stand-in** for the production reranker (`bge-reranker-v2-m3` via Pinecone, which was not available).
- **Questions: 263 (236 answerable, 27 unanswerable).** Types: direct, paraphrase, **keyword-style** ("hotel cost limit per night"), **misspelled** ("hotle reimbursment per nite"), exact identifier, table row (CSV and **62 PDF-table cells**), multi-passage, page boundary, scanned, injection-in-document, unanswerable. Labels are *content based* (file + required phrases), so the same questions score every chunking strategy. Dev/test split alternates by fact; nothing in the code was tuned on either split (the BM25 constants are the textbook k1=1.2, b=0.75, untuned).
- **Metrics:** Hit@K, Precision@K, Recall@K, MRR, nDCG@K with Wilson and bootstrap intervals (`rag_core/evaluation.py`); fair comparisons are **paired** on the same questions with a bootstrap interval on the per-question difference.
- **Dataset history (matters for reading older numbers):** v1.0 (141 questions) → v1.1 (+PDF tables, keyword and typo variants) → **v1.2**: the price-list CSV embedded each product SKU as a prefix of its article number, so correct identifier hits in the CSV were scored as misses (label named one file). Fixed. Results from v1.0/v1.1 files are not comparable and the v1.1 identifier numbers were wrong because of that flaw.

## Results (dataset v1.2, 236 answerable questions, K = 6 unless stated)

Recursive chunking (the default; 139 chunks):

| Method | Hit@1 | Hit@6 | P@6 | R@6 | R@20 | MRR | nDCG@6 |
|---|---|---|---|---|---|---|---|
| dense only | 0.737 | 0.932 | 0.167 | 0.932 | 0.958 | 0.815 | 0.831 |
| PG full-text, current (`ts_rank_cd`, OR) | 0.445 | 0.860 | 0.157 | 0.860 | 0.915 | 0.577 | 0.641 |
| **dense + current full-text, RRF (the hybrid until now)** | 0.598 | 0.924 | 0.167 | 0.924 | 0.962 | 0.734 | 0.773 |
| PG full-text, **idf** (BM25 over the OR candidates) | 0.886 | 0.975 | 0.179 | 0.975 | 0.987 | 0.920 | 0.931 |
| **dense + idf full-text, RRF** | 0.754 | 0.953 | 0.174 | 0.953 | **0.996** | 0.848 | 0.865 |
| PG `ts_rank` / AND-then-OR / RRF weight 0.5 (variants) | 0.886 / 0.538 / 0.691 | 0.949 / 0.890 / 0.915 | – | – | 0.966 / 0.941 / 0.958 | 0.908 / 0.649 / 0.786 | – |
| BM25 reference (in memory) / + dense RRF | 0.839 / 0.729 | 0.941 / 0.949 | – | – | 0.953 / 0.996 | 0.877 / 0.832 | – |
| current RRF + local cross-encoder | 0.864 | 0.958 | 0.175 | 0.958 | 0.962 | 0.908 | 0.919 |
| **idf RRF + local cross-encoder** | **0.894** | **0.992** | 0.184 | 0.992 | 0.996 | **0.939** | **0.950** |

The heading-aware (154 chunks) and semantic (149 chunks; 1,093 embedding texts) strategies show the same ordering: current RRF + cross-encoder Hit@6 0.966 / 0.962 → idf 0.996 / 0.996, MRR 0.924 / 0.919 → 0.953 / 0.949. Chunking strategy differences among themselves are within noise (e.g. dense Hit@1 0.737 / 0.767 / 0.767); **no chunking strategy is shown to be better, and the default (recursive) is unchanged.**

### Why the old full-text leg was weak (hypothesis tested, confirmed)

`ts_rank_cd` ignores how rare a term is, so the question's generic words ("product", "number", "article") outrank the one rare token. Evidence:

- **Exact identifiers** (12 questions such as "Which product has the article number AUX1-1390-S?"): the current keyword leg put **none** in its top 20 (Hit@6 0.00); the idf leg put **all 12** at rank 1 (Hit@1 1.00); after the cross-encoder, Hit@6 went 0.33 → 1.00.
- Overall, the current hybrid is **significantly worse than dense alone** before reranking: Hit@1 0.737 → 0.598 (paired difference −0.140, 95% CI [−0.195, −0.085]); MRR −0.081 [−0.114, −0.049].
- The idf fix, paired against the current hybrid: before reranking Hit@1 +0.157 [+0.110, +0.208], MRR +0.114 [+0.084, +0.146], candidate Recall@20 +0.034 [+0.013, +0.059]; **after the cross-encoder** MRR +0.031 [+0.013, +0.055], Hit@1 +0.030 [+0.009, +0.055], Hit@6 +0.034 [+0.013, +0.059] (recursive; the same size and sign for structure and semantic). On the dev split Hit@6 +0.042 [+0.008, +0.084]; on the test split +0.026 [0.000, +0.060].
- **Where the gain is:** all of it comes from exact-identifier questions (Hit@6 +0.667 [+0.417, +0.917]). For direct, paraphrase, keyword and typo questions the reranked results are identical (no question got worse in any of those groups).
- Cost: keyword query p50 20 ms → 91 ms locally (4 database round trips instead of 1); on a remote database (Neon) expect these round trips to dominate, i.e. a few hundred ms added per question. **Not measured against Neon.**

### Decision

`KEYWORD_RANKING` now defaults to `idf` in RAG Core. The original behaviour remains available (`KEYWORD_RANKING=cover_density`) for an immediate rollback, and it is tested that it still exhibits the weakness (so the test fails if the reason for the change ever disappears). **This changes retrieval behaviour on the next deployment; nothing was deployed.** The evidence is synthetic and uses a stand-in reranker, so it should be repeated on real documents and the production reranker before the change is trusted, and the setting can be pinned in the dashboard first (`KEYWORD_RANKING=cover_density`) to deploy without the behaviour change. Alternatives measured and rejected: `ts_rank` (good on this corpus but ignores length, worse on structure/semantic Hit@6), AND-then-OR (weak), lowering the keyword weight (helps a little, does not fix identifiers).

### PDF table extraction (`PDF_TABLES`, default `off`)

Compared on the 62 PDF-table questions (dataset v1.1, same corpus): bordered (`lines`) and borderless (`auto`) structuring changed retrieval metrics by amounts whose intervals include zero (reranked MRR +0.013 [−0.035, +0.065]; Hit@6 1.00 → 1.00). These questions only need the row key and the cell in the same chunk, which flat text also provides, so **this benchmark cannot show a benefit** of structure and shows no harm. The value of the feature is fidelity (each cell keeps its column name, multi-page tables keep their header, page numbers are kept), verified by `rag-core/tests/test_pdf_tables.py` (bordered, borderless, multi-page with repeated and with missing headers, empty cells, prose not mistaken for a table, malformed PDFs, time budget, scanned tables inferred from OCR positions and flagged). The default stays `off`; enable `lines` when documents contain ruled tables, `auto` only after trying it on your documents (borderless detection is a heuristic).

## Unanswerable questions

The top dense score separates answerable from unanswerable questions with AUC 0.892 (median 0.751 vs 0.612). This is a threshold signal, not a guarantee of correct abstention, and abstention itself is decided by the grounded prompt, not by retrieval.

## Limits

Synthetic corpus and questions, one run per configuration (the two earlier runs of the same configuration reproduced the same ordering), only 12 identifier questions, a stand-in reranker, English only, no real user documents, no Pinecone. These numbers support the engineering decision above; they do not establish a production quality level or that the system answers real questions correctly.
