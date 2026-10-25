"""OCR and extraction-report acceptance tests.

Real OCR (RapidOCR/ONNX) is used where the point is "the text is really recognised and then searchable";
a scripted engine is used where the point is a controlled failure, timeout, limit or confidence value.
All documents are generated at test time (tests/fixtures.py): the expected text is known exactly.
"""
import io
import json
import time
import uuid

import pytest

from conftest import IS_PG
from fixtures import blank_image, image_bytes, mixed_pdf, scanned_pdf, text_image, text_pdf
from rag_core import errors, ocr
from rag_core.config import settings
from rag_core.ocr import OcrEngine, OcrError, OcrResult
from rag_core.parsing import PDF, parse_document

PNG, JPEG = "image/png", "image/jpeg"
U1 = uuid.uuid4()

LINES_A = ["Quarterly revenue grew by eighteen percent in Europe.", "The warehouse in Rotterdam opened in March."]
LINES_B = ["Hotel stays are reimbursed up to 180 euros per night.", "Receipts must be submitted within thirty days."]


class ScriptedEngine(OcrEngine):
    """Returns scripted results per call; can fail or stall on chosen calls."""

    name = "scripted"

    def __init__(self, texts=None, confidence=0.9, fail_calls=(), sleep_calls=(), sleep_s=0.0):
        self.texts = list(texts or [])
        self.confidence, self.fail_calls, self.sleep_calls, self.sleep_s = confidence, set(fail_calls), set(sleep_calls), sleep_s
        self.calls = 0

    def recognize(self, image):
        n = self.calls
        self.calls += 1
        if n in self.sleep_calls:
            time.sleep(self.sleep_s)
        if n in self.fail_calls:
            raise OcrError("scripted failure")
        text = self.texts[n] if n < len(self.texts) else f"recognised text of call {n}"
        return OcrResult(text, self.confidence, 1)


@pytest.fixture
def ocr_on(monkeypatch):
    monkeypatch.setattr(settings, "OCR_ENABLED", True)
    yield
    ocr.set_engine(None)


@pytest.fixture(scope="module")
def real_engine():
    engine = ocr.RapidOcrEngine()
    engine.warm_up()
    return engine


def words(text):
    return {w.strip(".,").lower() for w in text.split()}


# ---------------------------------------------------------------- native / scanned / mixed

def test_text_pdf_does_not_use_ocr_even_when_enabled(ocr_on):
    engine = ScriptedEngine()
    ocr.set_engine(engine)
    doc = parse_document(text_pdf(["Alpha page with a normal text layer, long enough to count as text.", "Beta page also has real text on it."]), PDF)
    assert engine.calls == 0
    assert [p.method for p in doc.pages] == ["native", "native"]
    r = doc.report
    assert (r.native_pages, r.ocr_pages, r.method, r.partial) == (2, 0, "native", False)


def test_scanned_pdf_is_read_with_real_ocr_and_keeps_page_numbers(ocr_on, real_engine):
    ocr.set_engine(real_engine)
    pdf = scanned_pdf([text_image(LINES_A), text_image(LINES_B)])
    doc = parse_document(pdf, PDF)
    assert [p.number for p in doc.pages] == [1, 2]
    assert all(p.method == "ocr" and p.confidence is not None and 0 < p.confidence <= 1 for p in doc.pages)
    assert {"revenue", "rotterdam", "europe"} <= words(doc.pages[0].text)
    assert {"hotel", "reimbursed", "receipts"} <= words(doc.pages[1].text)
    r = doc.report
    assert (r.ocr_pages, r.native_pages, r.method, r.partial, r.total_pages) == (2, 0, "ocr", False, 2)
    assert r.ocr_engine == real_engine.name and r.ocr_confidence_mean is not None


def test_mixed_pdf_keeps_native_pages_and_ocrs_only_the_scanned_ones(ocr_on, real_engine):
    ocr.set_engine(real_engine)
    pdf = mixed_pdf(["Native page one has a genuine text layer with enough words.", text_image(LINES_B), "Native page three also has a genuine text layer here."])
    doc = parse_document(pdf, PDF)
    assert [(p.number, p.method) for p in doc.pages] == [(1, "native"), (2, "ocr"), (3, "native")]
    assert "Hotel" in doc.pages[1].text or "hotel" in doc.pages[1].text.lower()
    r = doc.report
    assert (r.native_pages, r.ocr_pages, r.method, r.partial) == (2, 1, "mixed", False)


def test_blank_pages_are_counted_and_never_sent_to_ocr(ocr_on):
    engine = ScriptedEngine()
    ocr.set_engine(engine)
    pdf = mixed_pdf(["Real text on the first page, long enough to be text.", blank_image(), blank_image(), "Real text on the last page, also long enough."])
    doc = parse_document(pdf, PDF)
    assert [p.number for p in doc.pages] == [1, 4]
    assert engine.calls == 0, "blank pages must be detected before OCR"
    assert doc.report.blank_pages == [2, 3] and doc.report.partial is False


def test_all_blank_document_has_no_extractable_text(ocr_on):
    ocr.set_engine(ScriptedEngine())
    with pytest.raises(errors.ServiceError) as exc:
        parse_document(scanned_pdf([blank_image(), blank_image()]), PDF)
    assert exc.value.code == errors.NO_EXTRACTABLE_TEXT


# ---------------------------------------------------------------- OCR disabled: nothing is silently lost

def test_scanned_pdf_without_ocr_fails_clearly(monkeypatch):
    monkeypatch.setattr(settings, "OCR_ENABLED", False)
    with pytest.raises(errors.ServiceError) as exc:
        parse_document(scanned_pdf([text_image(LINES_A)]), PDF)
    assert exc.value.code == errors.NO_EXTRACTABLE_TEXT and "OCR is not enabled" in exc.value.message and exc.value.retryable is False


def test_mixed_pdf_without_ocr_is_partial_and_says_which_pages(monkeypatch):
    monkeypatch.setattr(settings, "OCR_ENABLED", False)
    pdf = mixed_pdf(["Native page one has a genuine text layer with enough words.", text_image(LINES_B), blank_image()])
    doc = parse_document(pdf, PDF)
    r = doc.report
    assert [p.number for p in doc.pages] == [1]
    assert r.unprocessed_pages == [2] and r.partial is True
    assert r.warnings[0]["page"] == 2 and r.warnings[0]["code"] == "NO_TEXT_OCR_DISABLED"
    assert r.blank_pages == [3]  # a blank scanned page is recognised from a thumbnail and is not reported as lost content


# ---------------------------------------------------------------- controlled failures

def test_one_failed_page_does_not_invalidate_the_rest(ocr_on):
    ocr.set_engine(ScriptedEngine(texts=["First scanned page text", "", "Third scanned page text"], fail_calls={1}))
    pdf = scanned_pdf([text_image(LINES_A), text_image(LINES_B), text_image(LINES_A)])
    doc = parse_document(pdf, PDF)
    assert [p.number for p in doc.pages] == [1, 3]
    r = doc.report
    assert r.failed_pages == [2] and r.partial is True and r.ocr_pages == 2
    assert [(w["page"], w["code"]) for w in r.warnings] == [(2, "OCR_FAILED")]


def test_ocr_timeout_is_recorded_and_later_pages_still_work(ocr_on, monkeypatch):
    monkeypatch.setattr(settings, "OCR_PAGE_TIMEOUT_SECONDS", 0.3)
    ocr.set_engine(ScriptedEngine(texts=["", "Second page recovered"], sleep_calls={0}, sleep_s=1.0))
    pdf = scanned_pdf([text_image(LINES_A), text_image(LINES_B)])
    doc = parse_document(pdf, PDF)
    r = doc.report
    assert r.failed_pages == [1] and [w["code"] for w in r.warnings] == ["OCR_TIMEOUT"]
    assert [p.number for p in doc.pages] == [2] and doc.pages[0].text == "Second page recovered"
    assert r.partial is True


def test_page_limit_marks_the_rest_unprocessed(ocr_on, monkeypatch):
    monkeypatch.setattr(settings, "OCR_MAX_PAGES_PER_DOCUMENT", 1)
    engine = ScriptedEngine()
    ocr.set_engine(engine)
    doc = parse_document(scanned_pdf([text_image(LINES_A), text_image(LINES_B), text_image(LINES_A)]), PDF)
    assert engine.calls == 1 and [p.number for p in doc.pages] == [1]
    assert doc.report.unprocessed_pages == [2, 3] and {w["code"] for w in doc.report.warnings} == {"OCR_SKIPPED_PAGE_LIMIT"}
    assert doc.report.partial is True


def test_document_time_budget_stops_ocr(ocr_on, monkeypatch):
    monkeypatch.setattr(settings, "OCR_DOCUMENT_TIMEOUT_SECONDS", 0.2)
    ocr.set_engine(ScriptedEngine(sleep_calls={0}, sleep_s=0.5))
    doc = parse_document(scanned_pdf([text_image(LINES_A), text_image(LINES_B), text_image(LINES_A)]), PDF)
    assert [p.number for p in doc.pages] == [1] and doc.report.unprocessed_pages == [2, 3]
    assert {w["code"] for w in doc.report.warnings} == {"OCR_SKIPPED_TIME_BUDGET"}


def test_low_confidence_is_reported_but_text_is_kept(ocr_on, monkeypatch):
    monkeypatch.setattr(settings, "OCR_MIN_CONFIDENCE", 0.5)
    ocr.set_engine(ScriptedEngine(texts=["uncertain smudged words"], confidence=0.2))
    doc = parse_document(scanned_pdf([text_image(LINES_A)]), PDF)
    assert doc.pages[0].text == "uncertain smudged words" and doc.pages[0].confidence == 0.2
    assert [w["code"] for w in doc.report.warnings] == ["OCR_LOW_CONFIDENCE"] and doc.report.partial is False


def test_confidence_is_never_invented_when_the_engine_gives_none(ocr_on):
    ocr.set_engine(ScriptedEngine(texts=["text without a score"], confidence=None))
    doc = parse_document(scanned_pdf([text_image(LINES_A)]), PDF)
    assert doc.pages[0].confidence is None and doc.report.ocr_confidence_mean is None


def test_degraded_scan_never_crashes_and_reports_honestly(ocr_on, real_engine):
    """Blurred, noisy, downscaled page: accuracy is not asserted (it is genuinely degraded); the pipeline must
    either return text with an engine confidence or declare the page unreadable, never raise."""
    ocr.set_engine(real_engine)
    bad = text_image(LINES_A, noise=True, blur=1.6, scale=0.35)
    try:
        doc = parse_document(scanned_pdf([bad, text_image(LINES_B)]), PDF)
    except errors.ServiceError as exc:  # both pages unreadable would be NO_EXTRACTABLE_TEXT, still a clean outcome
        assert exc.code == errors.NO_EXTRACTABLE_TEXT
        return
    assert any(p.number == 2 for p in doc.pages)
    first = [p for p in doc.pages if p.number == 1]
    if first:
        assert first[0].method == "ocr"
    else:
        assert doc.report.blank_pages == [1] or doc.report.failed_pages == [1]


# ---------------------------------------------------------------- images

@pytest.mark.parametrize("fmt,mime", [("PNG", PNG), ("JPEG", JPEG)])
def test_png_and_jpeg_are_read_with_ocr(ocr_on, real_engine, fmt, mime):
    ocr.set_engine(real_engine)
    doc = parse_document(image_bytes(text_image(LINES_B), fmt), mime)
    assert len(doc.pages) == 1 and doc.pages[0].number is None and doc.pages[0].method == "ocr"
    assert {"hotel", "reimbursed", "receipts"} <= words(doc.pages[0].text)
    assert doc.report.source_type == "image" and doc.report.ocr_pages == 1


def test_images_are_unsupported_when_ocr_is_off(monkeypatch):
    monkeypatch.setattr(settings, "OCR_ENABLED", False)
    with pytest.raises(errors.ServiceError) as exc:
        parse_document(image_bytes(text_image(LINES_A)), PNG)
    assert exc.value.code == errors.UNSUPPORTED_CONTENT


def test_malformed_and_truncated_images_fail_cleanly(ocr_on):
    ocr.set_engine(ScriptedEngine())
    good = image_bytes(text_image(LINES_A))
    for bad in [b"not an image at all", good[: len(good) // 2], b"\x89PNG\r\n\x1a\n" + b"\x00" * 50]:
        with pytest.raises(errors.ServiceError) as exc:
            parse_document(bad, PNG)
        assert exc.value.code == errors.PARSE_FAILED and exc.value.retryable is False


def test_image_with_too_many_pixels_is_rejected(ocr_on, monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_MAX_PIXELS", 10_000)
    ocr.set_engine(ScriptedEngine())
    with pytest.raises(errors.ServiceError) as exc:
        parse_document(image_bytes(blank_image(400, 400)), PNG)
    assert exc.value.code in (errors.DOCUMENT_TOO_LARGE, errors.PARSE_FAILED)


def test_blank_image_has_no_text(ocr_on):
    engine = ScriptedEngine()
    ocr.set_engine(engine)
    with pytest.raises(errors.ServiceError) as exc:
        parse_document(image_bytes(blank_image()), PNG)
    assert exc.value.code == errors.NO_EXTRACTABLE_TEXT and engine.calls == 0


# ---------------------------------------------------------------- malformed / encrypted PDFs

def test_malformed_pdf_is_a_clean_non_retryable_failure():
    for bad in [b"%PDF-1.4 this is not really a pdf", b"%PDF-" + b"\x00" * 100, b""]:
        with pytest.raises(errors.ServiceError) as exc:
            parse_document(bad, PDF)
        assert exc.value.code in (errors.PARSE_FAILED, errors.NO_EXTRACTABLE_TEXT) and exc.value.retryable is False


def test_encrypted_pdf_is_rejected():
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for page in PdfReader(io.BytesIO(text_pdf(["Secret page with plenty of words in the text layer."]))).pages:
        writer.add_page(page)
    writer.encrypt("pw")
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(errors.ServiceError) as exc:
        parse_document(buf.getvalue(), PDF)
    assert exc.value.code == errors.PARSE_FAILED and "Password" in exc.value.message


def test_pdf_page_limit(monkeypatch):
    monkeypatch.setattr(settings, "MAX_PDF_PAGES", 2)
    with pytest.raises(errors.ServiceError) as exc:
        parse_document(text_pdf(["a long enough page of text number one", "a long enough page of text number two", "a long enough page of text number 3"]), PDF)
    assert exc.value.code == errors.DOCUMENT_TOO_LARGE


# ---------------------------------------------------------------- end to end: OCR text is searchable

async def _ingest(client, doc_id, data, mime, name):
    meta = {"document_id": str(doc_id), "user_id": str(U1), "filename": name, "mime_type": mime}
    return await client.post("/api/v1/rag/ingest", data={"metadata": json.dumps(meta)}, files={"file": (name, data, mime)})


async def test_ocr_text_is_indexed_and_retrievable_with_page_and_method(client, ocr_on, real_engine):
    ocr.set_engine(real_engine)
    pdf = mixed_pdf(["Native page about parking permits and visitor passes for the office.", text_image(LINES_B), text_image(LINES_A)])
    doc = uuid.uuid4()
    r = await _ingest(client, doc, pdf, PDF, "scan.pdf")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["chunks"] >= 3 and body["extraction"]["method"] == "mixed" and body["extraction"]["ocr_pages"] == 2
    q = await client.post("/api/v1/rag/retrieve", json={"user_id": str(U1), "query": "hotel reimbursed per night euros", "document_ids": [str(doc)]})
    top = q.json()["chunks"][0]
    assert top["page"] == 2 and top["extraction_method"] == "ocr" and "hotel" in top["text"].lower()
    q = await client.post("/api/v1/rag/retrieve", json={"user_id": str(U1), "query": "Rotterdam warehouse opened", "document_ids": [str(doc)]})
    assert q.json()["chunks"][0]["page"] == 3


@pytest.mark.skipif(not IS_PG, reason="background ingestion with concurrent sessions needs PostgreSQL (SQLite allows one writer); runs with TEST_DATABASE_URL")
async def test_partial_extraction_is_visible_through_the_status_api(client, ocr_on):
    ocr.set_engine(ScriptedEngine(texts=["Recovered text on page one"], fail_calls={1}))
    pdf = scanned_pdf([text_image(LINES_A), text_image(LINES_B)])
    doc = uuid.uuid4()
    meta = {"document_id": str(doc), "user_id": str(U1), "filename": "s.pdf", "mime_type": PDF}
    r = await client.post("/api/v1/rag/ingestions", data={"metadata": json.dumps(meta)}, files={"file": ("s.pdf", pdf, PDF)})
    assert r.status_code == 202
    for _ in range(200):
        st = (await client.get(f"/api/v1/rag/ingestions/{doc}", params={"user_id": str(U1)})).json()
        if st["status"] in ("succeeded", "failed"):
            break
        await __import__("asyncio").sleep(0.05)
    assert st["status"] == "succeeded", st
    ex = st["extraction"]
    assert ex["partial"] is True and ex["failed_pages"] == [2] and ex["ocr_pages"] == 1
    assert ex["warnings"][0]["code"] == "OCR_FAILED"
