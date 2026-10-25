"""A regression guard for OCR accuracy with the real engine: character error rate on synthetic samples whose text is known.

This is NOT an accuracy claim. It protects against a dependency or configuration change making OCR much worse, using a few
clean and mildly degraded renders in one script (Latin). The wider measurement (conditions, scripts, confidence) lives in
scripts/eval_ocr.py and docs/verification/OCR_ACCURACY.md; the floors here are deliberately looser than what was measured there.
"""
import pytest

from fixtures import text_image
from rag_core import ocr
from rag_core.evaluation import cer

LINES = [
    ["Invoice 88231 is due in 45 days, payable by bank transfer.", "Contract number 77412 expires on 30 November."],
    ["Hotel stays are reimbursed up to 180 euros per night.", "Receipts must be submitted within 30 days."],
    ["Quarterly revenue grew by eighteen percent in Europe.", "The warehouse in Rotterdam opened in March 2019."],
]


@pytest.fixture(scope="module")
def engine():
    try:
        eng = ocr.RapidOcrEngine()
        eng.recognize(text_image(["warm up"], size=30))
    except Exception as exc:
        pytest.skip(f"OCR engine unavailable: {exc}")
    return eng


def _mean_cer(engine, **kw):
    scores = []
    for lines in LINES:
        got = engine.recognize(text_image(lines, size=34, **kw)).text
        scores.append(cer("\n".join(lines), got))
    return sum(scores) / len(scores)


def test_clean_text_is_read_almost_perfectly(engine):
    assert _mean_cer(engine) <= 0.05


@pytest.mark.parametrize("kw", [{"noise": True}, {"blur": 1.2}, {"scale": 0.5}])
def test_mild_degradation_stays_within_a_loose_floor(engine, kw):
    assert _mean_cer(engine, **kw) <= 0.12, kw


def test_confidence_is_reported_but_is_not_an_error_signal(engine):
    res = engine.recognize(text_image(LINES[0], size=34))
    assert res.confidence is not None and 0 < res.confidence <= 1
    # Documented limit: a line the detector misses entirely does not lower this number (see OCR_ACCURACY.md), so warnings and
    # per-page confidence are hints for a human, never proof that a page is complete.
