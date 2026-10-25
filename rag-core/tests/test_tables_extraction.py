"""Table handling: row/column relationships survive extraction, repeated headers are not indexed as data,
empty cells and ragged rows do not shift values into the wrong column, identifiers stay exactly searchable."""
import json
import uuid

from fixtures import docx_bytes
from rag_core.parsing import CSV, DOCX, parse_document, parse_table

U1 = uuid.uuid4()


def csv_lines(text: str):
    return parse_document(text.encode(), CSV).pages[0].text.split("\n")


def test_each_row_is_self_describing_with_column_labels():
    lines = csv_lines("sku,product,country,price_eur\nZRy-SL18,Zephyr Relay,Germany,89\nHD-PT7,Halo Display,Portugal,186\n")
    assert lines == ["sku: ZRy-SL18 | product: Zephyr Relay | country: Germany | price_eur: 89",
                     "sku: HD-PT7 | product: Halo Display | country: Portugal | price_eur: 186"]


def test_empty_cells_do_not_shift_values_into_other_columns():
    lines = csv_lines("a,b,c\n1,,3\n,5,\n")
    assert lines == ["a: 1 | c: 3", "b: 5"]


def test_numeric_values_and_identifiers_are_kept_exactly():
    lines = csv_lines("id,amount,code\n007,1,234.50,x\n".replace("1,234.50", '"1,234.50"') + "")
    assert "id: 007" in lines[0] and "amount: 1,234.50" in lines[0]


def test_repeated_header_rows_are_not_indexed_as_data():
    """A table exported page by page repeats its header row; those repeats must not become rows."""
    lines = csv_lines("name,qty\nbolt,10\nname,qty\nnut,20\nNAME,QTY\nwasher,30\n")
    assert lines == ["name: bolt | qty: 10", "name: nut | qty: 20", "name: washer | qty: 30"]
    table = parse_table(b"name,qty\nbolt,10\nname,qty\nnut,20\n", 100, 10)
    assert table.rows == [["bolt", "10"], ["nut", "20"]]


def test_ragged_and_malformed_rows_are_tolerated_without_losing_data():
    lines = csv_lines("a,b,c\n1,2\n3,4,5,6\n\n,,\n7,8,9\n")
    assert lines[0] == "a: 1 | b: 2"
    assert lines[1].startswith("a: 3 | b: 4 | c: 5") and "column 4: 6" in lines[1]   # the extra cell is kept, labelled honestly
    assert lines[-1] == "a: 7 | b: 8 | c: 9" and len(lines) == 3                   # fully empty rows vanish


def test_semicolon_and_tab_delimiters_are_sniffed():
    assert csv_lines("a;b\n1;2\n") == ["a: 1 | b: 2"]
    assert csv_lines("a\tb\n1\t2\n") == ["a: 1 | b: 2"]


def test_numeric_first_row_is_data_not_a_header():
    assert csv_lines("1,2\n3,4\n") == ["1 | 2", "3 | 4"]


def test_docx_table_keeps_header_semantics_across_page_repeats_and_merged_cells():
    def build(d):
        d.add_heading("Specs", 1)
        t = d.add_table(rows=1, cols=2)
        t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Property", "Value"
        for k, v in [("Battery life", "14 months"), ("Weight", "85 grams")]:
            row = t.add_row()
            row.cells[0].text, row.cells[1].text = k, v
        rep = t.add_row()                       # header repeated at a page break
        rep.cells[0].text, rep.cells[1].text = "Property", "Value"
        row = t.add_row()
        row.cells[0].text, row.cells[1].text = "Ingress protection", "IP65"
        merged = t.add_row()
        merged.cells[0].merge(merged.cells[1])  # horizontally merged cell
        merged.cells[0].text = "Note: values at 20 C"
        d.add_paragraph("After the table.")

    text = parse_document(docx_bytes(build), DOCX).pages[0].text
    lines = text.split("\n")
    assert "Property: Battery life | Value: 14 months" in lines and "Property: Ingress protection | Value: IP65" in lines
    assert not any(ln.startswith("Property: Property") for ln in lines), "repeated header must not become a data row"
    assert text.count("Note: values at 20 C") == 1                      # a merged cell is read once
    assert lines.index("Specs") < lines.index("Property: Battery life | Value: 14 months") < lines.index("After the table.")


async def test_exact_identifier_lookup_survives_the_whole_pipeline(client):
    rows = ["sku,product,region,price"] + [f"SKU-{i:04d}-X,Product {i},Region {i % 7},{100 + i}" for i in range(300)]
    doc = uuid.uuid4()
    meta = {"document_id": str(doc), "user_id": str(U1), "filename": "price.csv", "mime_type": CSV}
    r = await client.post("/api/v1/rag/ingest", data={"metadata": json.dumps(meta)}, files={"file": ("price.csv", "\n".join(rows).encode(), CSV)})
    assert r.status_code == 200
    q = await client.post("/api/v1/rag/retrieve", json={"user_id": str(U1), "query": "SKU-0217-X", "document_ids": [str(doc)]})
    top_text = q.json()["chunks"][0]["text"]
    assert "sku: SKU-0217-X | product: Product 217 | region: Region 0 | price: 317" in top_text
