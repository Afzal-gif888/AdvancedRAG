"""Metric implementations checked against hand-computed values, and integrity checks of the evaluation dataset."""
import json
import math
from pathlib import Path

import pytest

from rag_core import evaluation as ev
from rag_core.chunking import chunk_pages
from rag_core.parsing import parse_document

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "evaluation" / "dataset"


# ---------------------------------------------------------------- metrics, by hand

def test_precision_recall_hit_for_a_known_ranking():
    # relevant = {a, b, c}; ranking [a, x, b, y, z]  ->  rels 1,0,1,0,0
    rels = [1, 0, 1, 0, 0]
    assert ev.hit_at_k(rels, 1) == 1.0 and ev.hit_at_k([0, 0, 1], 2) == 0.0
    assert ev.precision_at_k(rels, 3) == pytest.approx(2 / 3) and ev.precision_at_k(rels, 5) == pytest.approx(0.4)
    evidence = [{"file": "f", "phrases": ["a"]}, {"file": "f", "phrases": ["b"]}, {"file": "f", "phrases": ["c"]}]
    chunks = [("f", "a"), ("f", "x"), ("f", "b"), ("f", "y"), ("f", "z")]
    assert ev.recall_at_k(chunks, evidence, 3) == pytest.approx(2 / 3)
    assert ev.recall_at_k(chunks, evidence, 1) == pytest.approx(1 / 3)


def test_reciprocal_rank_and_mrr():
    assert [ev.reciprocal_rank(r) for r in ([1, 0], [0, 0, 1], [0, 0])] == [1.0, pytest.approx(1 / 3), 0.0]
    mrr = sum(ev.reciprocal_rank(r) for r in ([0, 1], [1], [0, 0], [0, 0, 0, 1])) / 4
    assert mrr == pytest.approx((0.5 + 1 + 0 + 0.25) / 4)


def test_ndcg_matches_the_worked_example():
    # Docs/Module 4: gains [3,2,0,1] -> 0.9854 with graded gains; here binary rels:
    assert ev.dcg([3, 2, 0, 1]) == pytest.approx(4.692536, abs=1e-5)
    assert ev.dcg([3, 2, 1, 0]) == pytest.approx(4.761860, abs=1e-5)
    assert ev.dcg([3, 2, 0, 1]) / ev.dcg([3, 2, 1, 0]) == pytest.approx(0.98544, abs=1e-5)
    # binary: relevant at ranks 2 and 3 of 3, two relevant exist -> DCG = 1/log2(3) + 1/log2(4); ideal = 1 + 1/log2(3)
    assert ev.ndcg_at_k([0, 1, 1], 3) == pytest.approx((1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3)))
    assert ev.ndcg_at_k([1, 1, 0], 3) == pytest.approx(1.0) and ev.ndcg_at_k([0, 0, 0], 3) == 0.0


def test_ndcg_uses_the_available_relevant_count_for_the_ideal_ranking():
    assert ev.ndcg_at_k([1, 0, 0], 3, total_relevant=3) < 1.0       # two relevant chunks were never retrieved
    assert ev.ndcg_at_k([1, 0, 0], 3) == 1.0                         # relative to what was retrieved


def test_score_question_end_to_end():
    ranked = [("f.md", "irrelevant"), ("f.md", "the answer is 180 euros"), ("g.md", "180 euros but wrong file")]
    s = ev.score_question(ranked, [{"file": "f.md", "phrases": ["180 euros"]}], ks=(1, 3))
    assert s["rr"] == 0.5 and s["hit@1"] == 0.0 and s["hit@3"] == 1.0 and s["recall@3"] == 1.0
    assert s["precision@3"] == pytest.approx(1 / 3)


def test_all_phrases_of_an_item_must_match_in_the_same_chunk():
    item = {"file": "p.csv", "phrases": ["Aurora X2", "Germany", "249"]}
    assert ev.is_relevant("p.csv", "product: Aurora X2 | country: Germany | price: 249", [item])
    assert not ev.is_relevant("p.csv", "product: Aurora X2 | country: France | price: 249", [item])
    assert not ev.is_relevant("other.csv", "product: Aurora X2 | country: Germany | price: 249", [item])


# ---------------------------------------------------------------- statistics

def test_wilson_interval_values():
    lo, hi = ev.wilson(19, 20)
    assert (round(lo, 3), round(hi, 3)) == (0.764, 0.991)
    assert round(ev.wilson(20, 20)[0], 3) == 0.839 and ev.wilson(0, 0) == (0.0, 1.0)


def test_bootstrap_interval_is_reproducible_and_brackets_the_mean():
    values = [1.0] * 17 + [0.0] * 3
    a, b = ev.bootstrap_ci(values, seed=3), ev.bootstrap_ci(values, seed=3)
    assert a == b and a[0] < 0.85 < a[1]


def test_percentile_and_auc():
    assert ev.percentile(list(range(1, 101)), 95) == 95 and ev.percentile([5], 99) == 5
    assert ev.auc([0.9, 0.8], [0.1, 0.2]) == 1.0 and ev.auc([0.5], [0.5]) == 0.5 and ev.auc([0.2], [0.8]) == 0.0


# ---------------------------------------------------------------- BM25

def test_bm25_prefers_rare_terms_and_saturates_term_frequency():
    docs = {"d1": "hotel hotel hotel hotel hotel policy", "d2": "the sku zry-sl18 is the zephyr relay", "d3": "policy about travel and hotel receipts"}
    idx = ev.BM25Index(docs)
    assert idx.search("zry-sl18")[0][0] == "d2"
    assert idx.idf("zry-sl18") > idx.idf("hotel")                      # rarer term, larger IDF
    # saturation: 5 occurrences do not score 5x one occurrence
    one = ev.BM25Index({"a": "hotel x", "b": "other y z"}).search("hotel")[0][1]
    five = ev.BM25Index({"a": "hotel hotel hotel hotel hotel", "b": "other y z"}).search("hotel")[0][1]
    assert five < 5 * one
    assert idx.search("nonexistentterm") == []


def test_bm25_idf_formula():
    idx = ev.BM25Index({"a": "x y", "b": "x z", "c": "w"})
    assert idx.idf("x") == pytest.approx(math.log(1 + (3 - 2 + 0.5) / (2 + 0.5)))


# ---------------------------------------------------------------- the dataset itself

pytestmark_dataset = pytest.mark.skipif(not (DATASET / "questions.jsonl").exists(), reason="run evaluation/build_dataset.py")


@pytestmark_dataset
def test_dataset_is_internally_consistent():
    questions = ev.load_questions(DATASET / "questions.jsonl")
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf8"))
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)) == manifest["questions"]
    assert {q["split"] for q in questions} == {"dev", "test"}
    for q in questions:
        assert q["answerable"] == bool(q["relevant"]), q["id"]
        assert not q["answerable"] or all(item["file"] in manifest["files"] and item["phrases"] for item in q["relevant"]), q["id"]
    assert manifest["answerable"] >= 100 and manifest["unanswerable"] >= 20
    assert {"direct", "paraphrase", "table", "exact_identifier", "multi_passage", "ambiguous", "page_boundary", "scanned", "unanswerable", "injection_doc"} <= set(manifest["types"])


@pytestmark_dataset
def test_every_label_is_satisfiable_and_specific_under_the_baseline_chunking():
    """An answerable question whose evidence appears in no chunk could never be scored correctly; one whose
    evidence appears in dozens of chunks would make Hit@K meaningless. Scanned files are checked in the OCR run."""
    questions = ev.load_questions(DATASET / "questions.jsonl")
    mime = {".md": "text/markdown", ".txt": "text/plain", ".csv": "text/csv", ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    chunks = {}
    for path in sorted(p for p in (DATASET / "corpus").rglob("*") if p.is_file()):
        name = path.relative_to(DATASET / "corpus").as_posix()
        if name == "scan.pdf":
            continue
        doc = parse_document(path.read_bytes(), mime[path.suffix])
        chunks[name] = [c.text for c in chunk_pages(doc.pages, 1200, 200)]
    for q in questions:
        if not q["answerable"] or q.get("requires") == "ocr":
            continue
        for item in q["relevant"]:
            hits = sum(all(p in t for p in item["phrases"]) for t in chunks[item["file"]])
            assert hits >= 1, f"{q['id']}: no chunk satisfies {item}"
            if q["type"] not in ("ambiguous", "page_boundary"):
                assert hits <= 3, f"{q['id']}: evidence is not specific ({hits} chunks match {item})"


# ---------------------------------------------------------------- text accuracy

def test_edit_distance_and_cer_wer_known_values():
    from rag_core.evaluation import cer, edit_distance, wer
    assert edit_distance("kitten", "sitting") == 3 and edit_distance("", "abc") == 3 and edit_distance("abc", "abc") == 0
    assert edit_distance(["a", "b"], ["a", "c", "b"]) == 1
    assert cer("hello world", "hello world") == 0.0
    assert abs(cer("hello world", "hallo wurld") - 2 / 11) < 1e-9
    assert cer("a  b\n c", "a b c") == 0.0, "whitespace is not an error"
    assert cer("Hello", "hello") > 0 and cer("Hello", "hello", fold=True) == 0.0
    assert cer("abc", "") == 1.0 and cer("", "") == 0.0 and cer("", "x") == 1.0
    assert cer("ab", "abcdef") == 2.0, "insertions can push CER above 1"
    assert abs(wer("the quick brown fox", "the quack brown") - 2 / 4) < 1e-9
