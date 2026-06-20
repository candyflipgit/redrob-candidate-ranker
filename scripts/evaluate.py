#!/usr/bin/env python3
"""Grade a submission.csv against our offline labels.

    python scripts/evaluate.py [--submission submission.csv] [--full]

Labels = hand-read gold holdout (eval/holdout_labels.csv) where available, else
the silver fit-tier model. Prints the official composite + components, the
honeypot rate in the top-100, and the tier distribution.  --full also labels the
whole pool to report the relevance landscape.
"""
from __future__ import annotations
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from redrob import evalmetrics as M  # noqa: E402
from redrob import silver  # noqa: E402
from redrob.jobspec import JobSpec  # noqa: E402


def load_submission(path):
    ranked = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0:
                continue
            if row and row[0].strip():
                ranked.append(row[0].strip())
    return ranked


def load_gold(path):
    gold = {}
    if not Path(path).exists():
        return gold
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].startswith("CAND_"):
                try:
                    gold[row[0].strip()] = int(row[1])
                except ValueError:
                    pass
    return gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="submission.csv")
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--spec", default="job_spec.yaml")
    ap.add_argument("--gold", default="eval/holdout_labels.csv")
    ap.add_argument("--full", action="store_true",
                    help="also label the whole pool for the relevance landscape")
    args = ap.parse_args()

    spec = JobSpec.from_yaml(ROOT / args.spec if not Path(args.spec).is_absolute() else args.spec)
    ranked = load_submission(ROOT / args.submission if not Path(args.submission).exists() else args.submission)
    gold = load_gold(ROOT / args.gold)

    cand_path = args.candidates if Path(args.candidates).exists() else ROOT / args.candidates

    if args.full:
        rel, honeypots = silver.label_pool(cand_path, spec)
        pool_dist = Counter(rel.values())
        pool_relevant = sum(1 for t in rel.values() if t >= M.REL_THRESHOLD)
        print(f"pool: {len(rel):,} labelled | relevant (tier>={M.REL_THRESHOLD}): "
              f"{pool_relevant:,} | honeypots: {len(honeypots):,}")
        print(f"pool tier dist: "
              f"{dict(sorted(pool_dist.items()))}")
    else:
        rel, honeypots = silver.label_pool(cand_path, spec, only=set(ranked))

    # gold overrides silver where we have hand-read labels
    n_gold = 0
    for cid in ranked:
        if cid in gold:
            rel[cid] = gold[cid]
            n_gold += 1

    scores = M.composite(ranked, rel)
    hp_rate = M.honeypot_rate(ranked, honeypots)
    top_dist = Counter(rel.get(c, 0) for c in ranked)

    print(f"\nsubmission: {args.submission}  ({len(ranked)} ranked)")
    print(f"labels    : {n_gold} gold (hand-read) + {len(ranked) - n_gold} silver\n")
    print(f"  NDCG@10   : {scores['ndcg@10']:.4f}")
    print(f"  NDCG@50   : {scores['ndcg@50']:.4f}")
    print(f"  MAP       : {scores['map']:.4f}")
    print(f"  P@10      : {scores['p@10']:.4f}")
    print(f"  --------")
    print(f"  COMPOSITE : {scores['composite']:.4f}   "
          f"(0.5*N10 + 0.3*N50 + 0.15*MAP + 0.05*P10)")
    print(f"\n  honeypot rate (top-{len(ranked)}): {hp_rate:.1%}  "
          f"{'OK' if hp_rate <= 0.10 else 'FAIL (>10% = DQ)'}")
    print(f"  top-{len(ranked)} tier dist: {dict(sorted(top_dist.items(), reverse=True))}")


if __name__ == "__main__":
    main()
