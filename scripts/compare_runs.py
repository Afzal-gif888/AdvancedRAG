"""Paired comparison of two retrieval methods on the same questions, from the JSON files written by run_retrieval_eval.py.

    python scripts/compare_runs.py --a RUN.json --a-method rrf --b RUN2.json --b-method rrf_idf [--strategy recursive]
                                   [--types pdf_table,keyword] [--split test]

Reports, for MRR, Hit@1, Hit@6 and Recall@20: both means, the mean paired difference (b - a) with a 95% bootstrap interval,
and the number of questions where b is better / worse / equal. A difference whose interval contains 0 is not evidence of an
improvement. The questions must be the same in both files (same dataset version); only questions present in both are used.
"""
import argparse
import json
import random
import sys

NAMES = ["mrr", "hit@1", "hit@6", "recall@20"]


def load(path, strategy, method):
    d = json.load(open(path, encoding="utf8"))
    meta = d["meta"].get("dataset_version")
    try:
        return meta, d["strategies"][strategy]["by_method"][method]["per_question"]
    except KeyError:
        sys.exit(f"{path}: no per-question data for strategy={strategy} method={method} (re-run with the current script)")


def boot(diffs, n=4000, seed=1):
    rnd = random.Random(seed)
    m = len(diffs)
    means = sorted(sum(diffs[rnd.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-method", required=True)
    ap.add_argument("--b-method", required=True)
    ap.add_argument("--strategy", default="recursive")
    ap.add_argument("--types", default="")
    ap.add_argument("--split", default="")
    args = ap.parse_args()
    va, pa = load(args.a, args.strategy, args.a_method)
    vb, pb = load(args.b, args.strategy, args.b_method)
    if va != vb:
        sys.exit(f"dataset versions differ: {va} vs {vb}")
    types = {t for t in args.types.split(",") if t}
    qids = [q for q in pa if q in pb and (not types or pa[q][4] in types) and (not args.split or pa[q][5] == args.split)]
    out = {"n": len(qids), "a": f"{args.a_method}", "b": f"{args.b_method}", "strategy": args.strategy, "types": sorted(types), "split": args.split or "all"}
    for i, name in enumerate(NAMES):
        xs, ys = [pa[q][i] for q in qids], [pb[q][i] for q in qids]
        diffs = [y - x for x, y in zip(xs, ys)]
        lo, hi = boot(diffs) if diffs else (0, 0)
        out[name] = {"a": round(sum(xs) / len(xs), 4), "b": round(sum(ys) / len(ys), 4), "diff": round(sum(diffs) / len(diffs), 4), "ci95": [round(lo, 4), round(hi, 4)],
                     "better": sum(d > 1e-9 for d in diffs), "worse": sum(d < -1e-9 for d in diffs), "equal": sum(abs(d) <= 1e-9 for d in diffs)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
