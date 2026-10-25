"""Retrieval evaluation: relevance labels, metrics, confidence intervals, and a reference BM25.

Pure functions, no I/O, no model: everything here is unit tested with hand-computed values
(tests/test_evaluation.py). The runner that uses them against the real pipeline is scripts/run_retrieval_eval.py.

Relevance model. A question has a list of *evidence items*; each item is {file, phrases}. A retrieved chunk is
relevant if it comes from that file and contains every phrase of at least one item. Content labels (not chunk
ids) are used so one question set can score any chunking strategy.

Metric definitions (K = cutoff, rel_i = 1 if the chunk at rank i is relevant):
  Hit@K        1 if any of the top K is relevant
  Precision@K  (sum of rel_i, i <= K) / K
  Recall@K     (number of evidence items covered by the top K) / (number of evidence items)
  RR           1 / rank of the first relevant chunk (0 if none); MRR is the mean over questions
  nDCG@K       DCG@K / IDCG@K with DCG = sum rel_i / log2(i + 1) and the ideal ranking putting
               min(K, number of relevant chunks in the candidate list) relevant chunks first (binary gain)
"""
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Evidence = Dict[str, Any]  # {"file": str, "phrases": [str, ...]}


# ---------------------------------------------------------------- relevance

def matches_item(filename: str, text: str, item: Evidence) -> bool:
    return filename == item["file"] and all(p in text for p in item["phrases"])


def is_relevant(filename: str, text: str, evidence: Sequence[Evidence]) -> bool:
    return any(matches_item(filename, text, item) for item in evidence)


def covered_items(chunks: Sequence[Tuple[str, str]], evidence: Sequence[Evidence]) -> int:
    """How many evidence items are matched by at least one of the given (filename, text) chunks."""
    return sum(1 for item in evidence if any(matches_item(f, t, item) for f, t in chunks))


# ---------------------------------------------------------------- ranking metrics

def hit_at_k(rels: Sequence[int], k: int) -> float:
    return 1.0 if any(rels[:k]) else 0.0


def precision_at_k(rels: Sequence[int], k: int) -> float:
    return sum(rels[:k]) / k


def recall_at_k(chunks: Sequence[Tuple[str, str]], evidence: Sequence[Evidence], k: int) -> float:
    return covered_items(chunks[:k], evidence) / len(evidence) if evidence else 0.0


def reciprocal_rank(rels: Sequence[int]) -> float:
    for i, r in enumerate(rels, start=1):
        if r:
            return 1.0 / i
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(rels: Sequence[float], k: int, total_relevant: Optional[int] = None) -> float:
    """nDCG with the ideal ranking built from the relevant items available (``total_relevant`` if given, else the
    number of relevant entries in ``rels``)."""
    available = total_relevant if total_relevant is not None else sum(1 for r in rels if r)
    ideal = dcg([1.0] * min(k, available))
    return dcg(list(rels[:k])) / ideal if ideal else 0.0


@dataclass
class QuestionResult:
    qid: str
    qtype: str
    rels: List[int]
    covered_at: Dict[int, float]  # k -> recall


def score_question(ranked: Sequence[Tuple[str, str]], evidence: Sequence[Evidence], ks: Iterable[int] = (1, 3, 6, 20)) -> Dict[str, float]:
    rels = [1 if is_relevant(f, t, evidence) else 0 for f, t in ranked]
    out: Dict[str, float] = {"rr": reciprocal_rank(rels)}
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(rels, k)
        out[f"precision@{k}"] = precision_at_k(rels, k)
        out[f"recall@{k}"] = recall_at_k(ranked, evidence, k)
        out[f"ndcg@{k}"] = ndcg_at_k(rels, k)
    return out


def aggregate(per_question: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = per_question[0].keys() if per_question else []
    return {k: sum(q[k] for q in per_question) / len(per_question) for k in keys}


# ---------------------------------------------------------------- statistics

def wilson(successes: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]


def percentile(values: Sequence[float], p: float) -> float:
    s = sorted(values)
    return s[max(math.ceil(p / 100 * len(s)), 1) - 1]


def auc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """Probability that a random positive scores higher than a random negative (ties count half); 0.5 = no
    separation. Used to ask: can the top retrieval score tell answerable from unanswerable questions?"""
    if not positives or not negatives:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


# ---------------------------------------------------------------- reference BM25 (for comparison only)

_STOP = {"the", "a", "an", "of", "to", "is", "are", "in", "on", "for", "and", "or", "at", "by", "with", "be", "as", "it", "that", "this",
         "what", "which", "how", "do", "does", "i", "my", "can", "will", "from", "was", "were", "has", "have"}
_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-_.]*[a-z0-9]|[a-z0-9]")


def tokens(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25Index:
    """Okapi BM25: score = sum_t IDF(t) * f*(k1+1) / (f + k1*(1 - b + b*|d|/avgdl)),
    IDF(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5)). PostgreSQL full-text ranking has no IDF; this is the
    alternative the evaluation compares it with."""

    def __init__(self, docs: Dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = {i: Counter(tokens(t)) for i, t in docs.items()}
        self.len = {i: sum(c.values()) for i, c in self.tf.items()}
        self.avg = (sum(self.len.values()) / len(self.len)) if self.len else 0.0
        self.df: Counter = Counter()
        for c in self.tf.values():
            self.df.update(c.keys())
        self.n = len(docs)

    def idf(self, term: str) -> float:
        n_t = self.df.get(term, 0)
        return math.log(1 + (self.n - n_t + 0.5) / (n_t + 0.5))

    def search(self, query: str, limit: int = 20) -> List[Tuple[str, float]]:
        q = tokens(query)
        scored = []
        for doc_id, tf in self.tf.items():
            s = 0.0
            for term in q:
                f = tf.get(term, 0)
                if f:
                    s += self.idf(term) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[doc_id] / (self.avg or 1)))
            if s > 0:
                scored.append((doc_id, s))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:limit]


# ---------------------------------------------------------------- dataset

def load_questions(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


# ---------------------------------------------------------------- text accuracy (OCR)

def edit_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Levenshtein distance (insertions, deletions, substitutions) between two sequences."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _norm(text: str, fold: bool) -> str:
    import unicodedata

    text = " ".join(unicodedata.normalize("NFC", text).split())
    return text.casefold() if fold else text


def cer(reference: str, hypothesis: str, fold: bool = False) -> float:
    """Character error rate: edit distance over the reference length (whitespace collapsed; case kept unless `fold`).
    Can exceed 1.0 when the hypothesis is much longer than the reference."""
    ref, hyp = _norm(reference, fold), _norm(hypothesis, fold)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str, fold: bool = False) -> float:
    ref, hyp = _norm(reference, fold).split(), _norm(hypothesis, fold).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)
