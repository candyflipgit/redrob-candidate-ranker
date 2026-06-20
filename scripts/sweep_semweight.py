#!/usr/bin/env python3
"""Tune the semantic-blend weight against the eval harness.

final(w) = (0.55*((1-w)*lexical + w*semantic) + 0.45*coherence) * M
         = A + w*B          (linear in w)
so we cache A, B, silver-tier and honeypot per candidate in ONE pass, then sweep
many weights instantly. Gold holdout overrides silver where available.

Run:  python scripts/sweep_semweight.py
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from redrob import io as cio, baseline as B, silver, evalmetrics as M  # noqa: E402
from redrob.jobspec import JobSpec  # noqa: E402


def load_gold():
    g = {}
    p = ROOT / "eval/holdout_labels.csv"
    if p.exists():
        for row in csv.reader(open(p, encoding="utf-8")):
            if row and row[0].startswith("CAND_"):
                g[row[0]] = int(row[1])
    return g


def main():
    spec = JobSpec.from_yaml(ROOT / "job_spec.yaml")
    sem_pct = B.load_semantic_pct(ROOT / "artifacts")
    if sem_pct is None:
        print("no embeddings in artifacts/ — run precompute first")
        return
    ref = B.reference_date(ROOT / "candidates.jsonl")
    gold = load_gold()

    recs = []           # (cid, A, B_coef)
    rel, honeypots = {}, set()
    for c in cio.iter_candidates(ROOT / "candidates.jsonl"):
        cid = c["candidate_id"]
        prof, hist, sig = c.get("profile", {}), cio.career_history(c), cio.signals(c)
        narrative_l = cio.narrative_text(c).lower()
        lex, _, _ = B.alignment(narrative_l, spec)
        coh = B.coherence(prof, hist, spec)
        avail, _ = B.availability(sig, ref)
        bf = B.band_fit(prof.get("years_of_experience"), spec)
        loc = B.location_fit(prof, sig, spec)
        dmult, _ = B.disqualifier_mult(hist, narrative_l, spec)
        hp = B.is_honeypot(c)
        sem = sem_pct.get(cid, 0.0)
        Mmult = bf * (0.85 + 0.15 * loc) * dmult * avail * (0.02 if hp else 1.0)
        A = (0.55 * lex + 0.45 * coh) * Mmult
        Bc = 0.55 * (sem - lex) * Mmult
        recs.append((cid, A, Bc))
        if hp:
            honeypots.add(cid)
        rel[cid] = silver.silver_tier(c, spec)
    rel.update(gold)

    print(f"cached {len(recs):,} candidates | gold labels: {len(gold)}\n")
    print(f"{'w':>5} {'NDCG@10':>8} {'NDCG@50':>8} {'MAP':>7} {'P@10':>6} "
          f"{'COMPOSITE':>10} {'hp%':>5} {'tier>=4 in top100':>18}")
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        scored = sorted(((A + w * Bc, cid) for cid, A, Bc in recs),
                        key=lambda x: (-x[0], x[1]))
        ranked = [cid for _, cid in scored[:100]]
        s = M.composite(ranked, rel)
        hpr = M.honeypot_rate(ranked, honeypots)
        hi = sum(1 for c in ranked if rel.get(c, 0) >= 4)
        print(f"{w:>5.2f} {s['ndcg@10']:>8.4f} {s['ndcg@50']:>8.4f} "
              f"{s['map']:>7.4f} {s['p@10']:>6.2f} {s['composite']:>10.4f} "
              f"{hpr:>5.0%} {hi:>18}")


if __name__ == "__main__":
    main()
