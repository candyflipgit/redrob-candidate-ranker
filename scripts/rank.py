#!/usr/bin/env python3
"""Produce the ranked submission CSV.

    python scripts/rank.py --candidates candidates.jsonl --out submission.csv

CPU-only, no network. Currently the text-only baseline ranker; the embedding +
tiered-scoring pipeline will slot in behind the same interface.
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from redrob import baseline  # noqa: E402
from redrob.jobspec import JobSpec  # noqa: E402


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() or p.exists() else (ROOT / p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--spec", default="job_spec.yaml")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--artifacts", default="artifacts",
                    help="dir with precomputed embeddings; '' to force lexical-only")
    args = ap.parse_args()

    spec = JobSpec.from_yaml(_resolve(args.spec))
    artifacts = _resolve(args.artifacts) if args.artifacts else None
    t0 = time.time()
    top, ref, used_emb = baseline.rank_pool(_resolve(args.candidates), spec,
                                            args.top, artifacts_dir=artifacts)

    out = Path(args.out)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, s in enumerate(top, 1):
            w.writerow([s.cid, i, f"{s.score:.6f}", baseline.reasoning(s)])

    dt = time.time() - t0
    mode = "hybrid (lexical+semantic)" if used_emb else "lexical-only (no artifacts)"
    print(f"wrote {len(top)} rows -> {out}  in {dt:.1f}s  ({mode}, ref {ref})")
    print("top 5:")
    for i, s in enumerate(top[:5], 1):
        print(f"  {i:>3}. {s.cid}  score={s.score:.4f}  [{s.title}, "
              f"{s.yoe}y]  must={s.must_hits} coh={s.coherence}")


if __name__ == "__main__":
    main()
