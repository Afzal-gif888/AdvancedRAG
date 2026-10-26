"""The worked examples in docs/verification/METRIC_DEFINITIONS.md, executed. If a formula in rag_core.evaluation changes,
these fail and the documentation must change with them."""
import math
import random

import pytest

from rag_core import evaluation as ev


def test_hit_at_5_eight_of_ten_queries():
    hits = [1] * 8 + [0] * 2  # per query: any relevant chunk in the top 5?
    assert sum(hits) / len(hits) == pytest.approx(0.80)


def test_hit_precision_recall_example():
    rels = [1, 0, 1, 0, 0, 0]
    assert ev.hit_at_k(rels, 1) == 1.0 and ev.hit_at_k([0, 0, 1], 2) == 0.0 and ev.hit_at_k([0, 0, 1], 3) == 1.0
    assert ev.precision_at_k(rels, 5) == pytest.approx(0.4)
    evidence = [{"file": "a", "phrases": ["x"]}, {"file": "b", "phrases": ["y"]}]
    chunks = [("a", "has x"), ("c", "noise"), ("b", "has y")]
    assert ev.recall_at_k(chunks, evidence, 1) == 0.5
    assert ev.recall_at_k(chunks, evidence, 3) == 1.0


def test_mrr_example():
    assert ev.reciprocal_rank([0, 0, 1]) == pytest.approx(1 / 3)
    per_query = [ev.reciprocal_rank(r) for r in ([1, 0], [0, 0, 1], [0, 0, 0])]
    assert sum(per_query) / 3 == pytest.approx((1 + 1 / 3 + 0) / 3)  # 0.4444


def test_ndcg_example():
    rels = [0, 1, 0, 1, 0]
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3)
    assert ev.ndcg_at_k(rels, 5, total_relevant=2) == pytest.approx(dcg / idcg)
    assert dcg / idcg == pytest.approx(0.6509, abs=1e-4)
    assert ev.ndcg_at_k([1, 1, 0], 3, total_relevant=2) == pytest.approx(1.0)  # ideal order


def test_nearest_rank_percentiles():
    xs = list(range(1, 101))
    assert ev.percentile(xs, 50) == 50 and ev.percentile(xs, 95) == 95 and ev.percentile(xs, 99) == 99
    ten = [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000]
    assert ev.percentile(ten, 50) == 50 and ev.percentile(ten, 90) == 90
    assert ev.percentile(ten, 99) == 1000  # with 10 samples p99 is just the maximum: not a real estimate
    assert sum(ten) / len(ten) == 145  # one outlier drags the mean far above the median


def test_cer_example():
    assert ev.edit_distance("kitten", "sitting") == 3
    assert ev.cer("kitten", "sitting") == pytest.approx(0.5)
    assert ev.cer("abcd", "abcd") == 0.0
    assert ev.cer("ab", "abxxxx") == pytest.approx(2.0)  # can exceed 1


def test_wilson_interval_for_8_of_10():
    lo, hi = ev.wilson(8, 10)
    assert (round(lo, 3), round(hi, 3)) == (0.490, 0.943)


def test_paired_difference_interval_is_computed_on_per_question_differences():
    a = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
    b = [1, 0, 0, 1, 0, 0, 1, 0, 1, 0]
    diffs = [x - y for x, y in zip(a, b)]
    assert sum(diffs) / len(diffs) == pytest.approx(0.3)
    lo, hi = ev.bootstrap_ci(diffs, seed=1)
    assert lo > 0 and hi < 0.7 and lo <= 0.3 <= hi


def test_bootstrap_is_deterministic_for_a_seed():
    vals = [random.Random(3).random() for _ in range(20)]
    assert ev.bootstrap_ci(vals, seed=5) == ev.bootstrap_ci(vals, seed=5)


def test_throughput_is_not_concurrency():
    # 100 requests finished in 20 s wall at 4 in flight: throughput 5 req/s; mean latency by Little's law L = lambda * W -> W = 4 / 5 = 0.8 s
    throughput, in_flight = 100 / 20, 4
    assert in_flight / throughput == pytest.approx(0.8)
