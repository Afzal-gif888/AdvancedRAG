"""Hostile or malformed uploads must be refused with a typed, non-retryable error and no crash or runaway work."""
import io
import zipfile

import pytest
from PIL import Image

from rag_core import errors
from rag_core.config import settings
from rag_core.parsing import DOCX, parse_document

PNG = "image/png"


def _zip(entries: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_a_zip_bomb_is_refused_before_it_is_unpacked(monkeypatch):
    monkeypatch.setattr(settings, "MAX_ZIP_UNCOMPRESSED_BYTES", 50 * 1024 * 1024)
    bomb = _zip({"word/document.xml": b"\0" * (80 * 1024 * 1024)})  # tiny on disk, huge unpacked
    assert len(bomb) < 1024 * 1024
    with pytest.raises(errors.ServiceError) as exc:
        parse_document(bomb, DOCX, settings)
    assert exc.value.code == errors.PARSE_FAILED and not exc.value.retryable


def test_a_high_compression_ratio_is_refused_even_below_the_size_cap():
    bomb = _zip({"word/document.xml": b"A" * (20 * 1024 * 1024)})  # ratio far above MAX_ZIP_RATIO
    with pytest.raises(errors.ServiceError) as exc:
        parse_document(bomb, DOCX, settings)
    assert exc.value.code == errors.PARSE_FAILED


def test_a_file_claiming_to_be_docx_but_not_a_zip_is_a_parse_failure():
    with pytest.raises(errors.ServiceError) as exc:
        parse_document(b"%PDF-1.4 not a zip at all", DOCX, settings)
    assert exc.value.code == errors.PARSE_FAILED and not exc.value.retryable


def test_a_pixel_flood_image_is_refused(monkeypatch):
    monkeypatch.setattr(settings, "IMAGE_MAX_PIXELS", 1_000_000)
    buf = io.BytesIO()
    Image.new("L", (4000, 4000), 255).save(buf, "PNG")  # 16 MP of white compresses to a few KB
    assert len(buf.getvalue()) < 200_000
    with pytest.raises(errors.ServiceError) as exc:
        parse_document(buf.getvalue(), PNG, settings)
    assert not exc.value.retryable


def test_a_truncated_or_garbage_image_is_refused_as_unsupported():
    buf = io.BytesIO()
    Image.new("RGB", (200, 200), "white").save(buf, "PNG")
    for data in (buf.getvalue()[:60], b"\x89PNG\r\n\x1a\n" + b"garbage" * 20, b""):
        with pytest.raises(errors.ServiceError) as exc:
            parse_document(data, PNG, settings)
        assert exc.value.code == "UNSUPPORTED_CONTENT" and not exc.value.retryable


# ---------------------------------------------------------------- polyglots and resource budgets

def test_a_pdf_zip_polyglot_is_treated_as_exactly_what_it_is_declared_to_be():
    from fixtures import text_pdf
    pdf = text_pdf(["Only the PDF side is read.", "Second page."])
    zip_side = _zip({"word/document.xml": b"<w:document/>", "payload.txt": b"hidden second file"})
    polyglot = pdf + zip_side                 # valid PDF prefix, valid ZIP central directory at the end
    as_pdf = parse_document(polyglot, "application/pdf", settings)
    assert [p.number for p in as_pdf.pages] == [1, 2] and "hidden second file" not in "\n".join(p.text for p in as_pdf.pages)
    with pytest.raises(errors.ServiceError) as exc:                    # declared as DOCX: not a real document
        parse_document(polyglot, DOCX, settings)
    assert not exc.value.retryable


def test_a_text_file_with_html_and_script_content_is_just_text():
    html = b"<html><script>alert(document.cookie)</script><body>Quarterly figures: 1,200 units.</body></html>"
    doc = parse_document(html, "text/plain", settings)
    assert "Quarterly figures" in doc.pages[0].text         # indexed as text; never rendered or executed by the service


def test_the_pdf_parse_budget_reports_unread_pages_instead_of_running_on(monkeypatch):
    import types
    from rag_core import parsing
    from fixtures import text_pdf
    ticks = iter(range(10_000))
    monkeypatch.setattr(parsing, "time", types.SimpleNamespace(monotonic=lambda: next(ticks), perf_counter=parsing.time.perf_counter))
    monkeypatch.setattr(settings, "PDF_PARSE_TIMEOUT_SECONDS", 3)
    doc = parse_document(text_pdf([f"Page number {i} has some text." for i in range(1, 8)]), "application/pdf", settings)
    read = [p.number for p in doc.pages]
    assert read and read[0] == 1 and len(read) < 7, read
    assert doc.report.partial and set(doc.report.failed_pages) == set(range(1, 8)) - set(read)
    assert any(w["code"] == "PARSE_TIME_BUDGET" for w in doc.report.warnings)


def test_a_pdf_page_cap_is_enforced_before_any_page_is_read(monkeypatch):
    from fixtures import text_pdf
    monkeypatch.setattr(settings, "MAX_PDF_PAGES", 2)
    with pytest.raises(errors.ServiceError) as exc:
        parse_document(text_pdf(["a", "b", "c"]), "application/pdf", settings)
    assert exc.value.code == errors.DOCUMENT_TOO_LARGE and not exc.value.retryable
