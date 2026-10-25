# OCR accuracy — recorded measurement and how to enable OCR safely

Raw data: `evaluation/results/20261023-203232-ocr.json` (per sample). Reproduce: `python scripts/eval_ocr.py`.
Engine: `rapidocr-onnxruntime` 1.2.3 (ONNX Runtime, CPU, self-hosted, no network), the version the tests ran against.
Machine: one Windows 11 laptop, 12 logical CPUs.

## Method

- **Ground truth is exact.** Every sample is rendered from known text with Pillow, so there is no labelling error. The cost is realism: these are synthetic renders, not photographs or real scans.
- **Metric.** Character error rate (CER) = Levenshtein distance / reference length, after Unicode NFC and whitespace collapsing, case kept (case matters for identifiers). WER is recorded too.
- **English conditions:** 4 text blocks (3 lines each, with numbers, identifiers, punctuation, an e-mail address) × 11 conditions = 44 samples: Arial 34 px, Arial 22 px, Courier, Times, salt noise, Gaussian blur, down-sampling to 50% and 35%, rotation by 2° and 5°, JPEG quality 30.
- **Scripts:** one block per script rendered with a system font (German/French/Spanish accents, Russian, Greek, Chinese, Korean).
- **Page order:** a 4-page scanned PDF and a 4-page mixed PDF (pages 1 and 3 with a text layer, 2 and 4 scanned) through the real `parse_document`.

## Results

| English text | Samples | Mean CER | Worst sample |
|---|---|---|---|
| All conditions | 44 | **0.019** (micro 0.018) | 0.34 |
| Clean Arial 34 px | 4 | 0.085 | 0.34 (one whole line missing, see below) |
| Arial 22 px | 4 | 0.062 | 0.093 |
| Courier | 4 | 0.014 | 0.032 |
| Times | 4 | 0.000 | 0.000 |
| Salt noise | 4 | 0.022 | 0.026 |
| Blur σ=1.5 | 4 | 0.000 | 0.000 |
| Down-sampled 50% / 35% | 4 / 4 | 0.000 / 0.019 | 0.000 / 0.036 |
| Rotated 2° / 5° | 4 / 4 | 0.000 / 0.005 | 0.000 / 0.007 |
| JPEG q30 | 4 | 0.000 | 0.000 |

| Script (one block each) | CER | Reads? |
|---|---|---|
| Chinese (simplified) | 0.00 | yes |
| French / Spanish / German (accented Latin) | 0.05 / 0.07 / 0.10 | mostly; accents are often lost or changed |
| Russian (Cyrillic) | 0.86 | **no** |
| Greek | 0.78 | **no** |
| Korean | 0.87 | **no** |

Page order and numbers: the scanned PDF and the mixed PDF both came back in order with correct page numbers (1–4), native pages marked `native` and scanned pages marked `ocr`, no warnings.
Speed: about 1.7–2.1 s per three-line image on this machine (single-threaded engine, one OCR worker by design).

## What the numbers do not say

- **A line can be dropped without any signal.** In the one "clean" sample with the worst CER the engine silently omitted the entire line `Article number AUX1-4821-S ships from Rotterdam.` and added a stray character; the page's mean confidence stayed 0.9. Tuning the engine's thresholds did not fix it (lowering `text_score`/`box_thresh` or raising `unclip_ratio` made overall CER worse: 0.019 → 0.022–0.031), so the miss is in text detection and no change was made.
- **Confidence is not an error predictor.** Pearson correlation between mean page confidence and (1 − CER) over the 44 English samples is **0.11**. The `OCR_LOW_CONFIDENCE` warning therefore catches some bad pages but must not be read as proof that a page with a high score is complete.
- **Supported languages are what the shipped model reads.** Latin script (with accent errors) and Chinese worked here; Cyrillic, Greek and Korean did not. Those documents should not be enabled for OCR without a different model (not implemented, not tested). Arabic, Hebrew and Indic scripts were not tested.
- Not measured: handwriting, phone photos, skewed or curved pages, tables (their cell structure is inferred from positions and flagged), scans at other DPI, long documents (accuracy over many pages), and speed on the production hardware (Render free is 0.1 CPU; this laptop is far faster).
- Sample sizes are small (4 blocks per condition). Differences between conditions of a few CER points are not established.

## Enabling OCR safely (it is configurable and off by default)

1. Decide you need it: scans and photographed documents are otherwise rejected (`NO_EXTRACTABLE_TEXT`) or reported as unprocessed pages.
2. RAG Core image: the Dockerfile installs `rapidocr-onnxruntime`, `opencv-python-headless`, `pypdfium2` (pinned in `constraints/rag-core.txt`). Build it and run the test suite against that image before relying on it.
3. Set on **rag-core only**: `OCR_ENABLED=true`. Keep the bounds unless you have measured better ones: `OCR_MAX_PAGES_PER_DOCUMENT=200`, `OCR_PAGE_TIMEOUT_SECONDS=120`, `OCR_DOCUMENT_TIMEOUT_SECONDS=1800`, `OCR_RENDER_DPI=150`, `OCR_MAX_PIXELS=12000000`.
4. Capacity: OCR is CPU-bound and runs one page at a time; on a 0.1 CPU instance expect minutes per page. Do not enable it on the free instance without measuring there.
5. Images (PNG/JPEG) additionally need the backend allowlist: `ALLOWED_UPLOAD_EXTENSIONS=pdf,txt,md,csv,docx,png,jpg,jpeg`.
6. Tell users what to expect: the document shows a warning for pages that failed or were skipped; text from OCR may contain errors or missing lines. Do not use OCR output for anything that must be exact (amounts, identifiers) without checking the original.
7. Regression guard: `rag-core/tests/test_ocr_accuracy.py` fails if mean CER on a few clean and mildly degraded renders rises above loose floors (0.05 clean, 0.12 degraded). It is not an accuracy claim.
