"""Table structure in PDFs: bordered, borderless, multi-page, scanned (OCR layout) and malformed inputs.
Expectations are written from the generated content, never from what the extractor happens to return."""
import pytest
from PIL import Image, ImageDraw

from fixtures import _font, layout_pdf, scanned_pdf, text_image
from rag_core import errors, ocr
from rag_core.config import settings
from rag_core.parsing import PDF, parse_document

SALES = [["Product", "Region", "Units", "Revenue"],
         ["Aurora X1", "EU", "120", "45,000"],
         ["Aurora X2", "US", "80", "38,400"],
         ["Borealis", "APAC", "15", "9,750"],
         ["Cirrus", "EU", "300", "81,000"]]
COLS = [72, 200, 300, 400]
OPT = {"bordered": True, "col_x": COLS, "right": 500}


def page_with(table_rows, bordered, intro="Quarterly sales report for the regional offices.", outro="All figures are in euros unless stated otherwise."):
    return [("text", 740, intro), ("table", 700, table_rows, {**OPT, "bordered": bordered}), ("text", 540, outro)]


@pytest.fixture
def tables(monkeypatch):
    def setting(mode):
        monkeypatch.setattr(settings, "PDF_TABLES", mode)
    return setting


def parse(pdf):
    return parse_document(pdf, PDF, settings)


# ---------------------------------------------------------------- bordered

def test_bordered_table_keeps_headers_rows_and_page_number(tables):
    tables("lines")
    doc = parse(layout_pdf([page_with(SALES, bordered=True)]))
    assert [p.number for p in doc.pages] == [1]
    text = doc.pages[0].text
    for r in SALES[1:]:
        assert f"Product: {r[0]} | Region: {r[1]} | Units: {r[2]} | Revenue: {r[3]}" in text
    assert text.index("Quarterly sales report") < text.index("Product: Aurora X1") < text.index("All figures are in euros")
    assert doc.report.tables == [{"page": 1, "kind": "lines", "rows": 4, "columns": 4, "header": True, "continued": False}]


def test_off_is_the_unchanged_plain_text_default():
    assert settings.PDF_TABLES == "off"
    doc = parse(layout_pdf([page_with(SALES, bordered=True)]))
    assert "Product: Aurora X1" not in doc.pages[0].text and "Aurora X1" in doc.pages[0].text and doc.report.tables == []


def test_empty_cells_stay_empty_and_are_not_shifted(tables):
    tables("lines")
    rows = [["Item", "Note", "Qty"], ["Bolt", "", "40"], ["Nut", "zinc", ""]]
    doc = parse(layout_pdf([[("table", 700, rows, {"bordered": True, "col_x": [72, 200, 330], "right": 430})]]))
    text = doc.pages[0].text
    assert "Item: Bolt | Qty: 40" in text
    assert "Item: Nut | Note: zinc" in text and "Item: Nut | Note: zinc | Qty" not in text


def test_a_numeric_first_row_is_data_not_a_header(tables):
    tables("lines")
    rows = [["2021", "10", "7"], ["2022", "12", "9"], ["2023", "15", "11"]]
    doc = parse(layout_pdf([[("table", 700, rows, {"bordered": True, "col_x": [72, 172, 272], "right": 372})]]))
    assert doc.report.tables[0]["header"] is False
    assert "2021 | 10 | 7" in doc.pages[0].text and "column 1" not in doc.pages[0].text


# ---------------------------------------------------------------- borderless

def test_borderless_table_needs_auto_mode(tables):
    pdf = layout_pdf([page_with(SALES, bordered=False)])
    tables("lines")
    assert parse(pdf).report.tables == [], "lines mode never guesses at borderless tables"
    tables("auto")
    doc = parse(pdf)
    assert [t["kind"] for t in doc.report.tables] == ["text"]
    assert "Product: Borealis | Region: APAC | Units: 15 | Revenue: 9,750" in doc.pages[0].text


def test_prose_is_not_mistaken_for_a_table(tables):
    tables("auto")
    lines = [
        "The committee met on Tuesday to review the proposal and agreed that the budget",
        "should be reconsidered after the next quarter, once the regional figures have been",
        "published. Several members raised concerns about the timeline, noting that earlier",
        "projects of similar scope had run over by a considerable margin. The chair asked",
        "for a written summary before the following meeting and closed the session at noon.",
        "No further business was recorded, and the minutes were approved without comment.",
    ]
    doc = parse(layout_pdf([[("text", 740 - 18 * i, line) for i, line in enumerate(lines)]]))
    assert doc.report.tables == []
    assert "committee met on Tuesday" in doc.pages[0].text


# ---------------------------------------------------------------- multi-page

HEAD = ["Item", "Zone", "Qty", "Price"]


def long_rows(n):
    return [[f"Item {i:02d}", f"Zone {i % 3}", str(i * 7), f"{i * 13}.50"] for i in range(1, n + 1)]


def test_repeated_header_on_the_next_page_is_dropped_and_rows_keep_their_columns(tables):
    tables("lines")
    rows = long_rows(34)
    p1 = [("table", 760, [HEAD] + rows[:33], OPT)]
    p2 = [("table", 760, [HEAD] + rows[33:], OPT)]
    doc = parse(layout_pdf([p1, p2]))
    assert [t["continued"] for t in doc.report.tables] == [False, True]
    assert doc.pages[1].number == 2
    assert "Item | Zone | Qty | Price" not in doc.pages[1].text, "the repeated header row is not data and not repeated"
    assert "Item: Item 34 | Zone: Zone 1 | Qty: 238 | Price: 442.50" in doc.pages[1].text


def test_headerless_continuation_inherits_the_header_of_the_previous_page(tables):
    tables("lines")
    rows = long_rows(35)
    doc = parse(layout_pdf([[("table", 760, [HEAD] + rows[:33], OPT)], [("table", 760, rows[33:], OPT)]]))
    assert [t["continued"] for t in doc.report.tables] == [False, True]
    assert "Item: Item 34 | Zone: Zone 1 | Qty: 238 | Price: 442.50" in doc.pages[1].text
    assert "Item: Item 35" in doc.pages[1].text


def test_a_different_table_on_the_next_page_is_not_glued_to_the_previous_one(tables):
    tables("lines")
    other = [["Name", "Role", "Team", "Since"], ["Ana", "Lead", "Core", "2019"], ["Ben", "Dev", "Core", "2021"]]
    doc = parse(layout_pdf([[("table", 760, [HEAD] + long_rows(33), OPT)], [("table", 760, other, OPT)]]))
    assert [t["continued"] for t in doc.report.tables] == [False, False]
    assert "Name: Ana | Role: Lead | Team: Core | Since: 2019" in doc.pages[1].text


def test_columns_are_never_guessed_for_a_headerless_table(tables):
    tables("lines")
    rows = [["2021", "10", "7"], ["2022", "12", "9"]]
    doc = parse(layout_pdf([[("table", 760, rows, {"bordered": True, "col_x": [72, 172, 272], "right": 372})]]))
    assert "Qty" not in doc.pages[0].text and doc.report.tables[0]["header"] is False


# ---------------------------------------------------------------- failures

@pytest.mark.parametrize("make", [
    lambda good: good[: len(good) // 2],                       # truncated
    lambda good: good.replace(b"endobj", b"endobX", 3),        # damaged object markers
    lambda good: b"%PDF-1.4\n" + b"\x00" * 200,                # header only
])
def test_malformed_pdfs_never_crash_and_never_invent_tables(tables, make):
    tables("auto")
    good = layout_pdf([page_with(SALES, bordered=True)])
    try:
        doc = parse_document(make(good), PDF, settings)
    except errors.ServiceError as exc:
        assert not exc.retryable
    else:
        for t in doc.report.tables:   # whatever survived must be real: no more rows than the source had
            assert t["rows"] <= len(SALES)
        text = "\n".join(p.text for p in doc.pages)
        assert "Zenith" not in text


def test_table_time_budget_leaves_plain_text_and_warns(tables, monkeypatch):
    tables("lines")
    monkeypatch.setattr(settings, "PDF_TABLE_TIMEOUT_SECONDS", 1e-9)
    doc = parse(layout_pdf([page_with(SALES, bordered=True)]))
    assert doc.report.tables == [] and "Aurora X1" in doc.pages[0].text
    assert any(w["code"] == "TABLES_TIME_BUDGET" for w in doc.report.warnings)


def test_missing_pdfplumber_degrades_to_plain_text_with_a_warning(tables, monkeypatch):
    tables("lines")
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "pdfplumber":
            raise ImportError("no pdfplumber")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    doc = parse(layout_pdf([page_with(SALES, bordered=True)]))
    assert doc.report.tables == [] and any(w["code"] == "TABLES_UNAVAILABLE" for w in doc.report.warnings)


# ---------------------------------------------------------------- scanned (OCR layout)

def test_ocr_layout_table_blocks_need_aligned_rows():
    from rag_core.pdftables import ocr_table_blocks
    text = "Report\nProduct | Region | Units\nAurora | EU | 120\nBorealis | APAC | 15\nCirrus | EU | 300\nTotal for the quarter was good."
    (start, end, rows), = ocr_table_blocks(text)
    assert (start, end) == (1, 5) and rows[0] == ["Product", "Region", "Units"] and rows[-1] == ["Cirrus", "EU", "300"]
    # a row with a missing cell ends the run: no structure is inferred across the gap
    assert ocr_table_blocks("A | B | C\n1 | 2 | 3\n4 | 5\n6 | 7 | 8\n9 | 10 | 11") == []


@pytest.fixture
def real_engine():
    try:
        eng = ocr.RapidOcrEngine()
        eng.recognize(text_image(["warm up"], size=30))
    except Exception as exc:
        pytest.skip(f"OCR engine unavailable: {exc}")
    ocr.set_engine(eng)
    yield eng
    ocr.set_engine(None)


def test_scanned_table_is_structured_from_ocr_positions_and_flagged(tables, monkeypatch, real_engine):
    tables("lines")
    monkeypatch.setattr(settings, "OCR_ENABLED", True)
    rows = [["Product", "Region", "Units"], ["Aurora", "EU", "120"], ["Borealis", "APAC", "15"], ["Cirrus", "EU", "300"]]
    img = Image.new("L", (1400, 520), 255)
    d = ImageDraw.Draw(img)
    f = _font(40)
    for r, row in enumerate(rows):
        for x, cell in zip((80, 560, 1000), row):
            d.text((x, 60 + r * 100), cell, fill=0, font=f)
    doc = parse_document(scanned_pdf([img.convert("RGB")]), PDF, settings)
    text = doc.pages[0].text
    assert doc.pages[0].method == "ocr"
    assert "Product: Cirrus | Region: EU | Units: 300" in text, text
    assert [t["kind"] for t in doc.report.tables] == ["ocr_layout"]
    assert any(w["code"] == "TABLE_FROM_OCR_LAYOUT" for w in doc.report.warnings)
