#!/usr/bin/env python3
"""Weight-sensitivity check: is the ranking robust to its weights, or fragile?

Caches the raw score components once, then recomputes the top-100 under perturbed
weight configs (+/-~20% around defaults) and reports, vs the default config:
  - top-10 Jaccard overlap   (1.0 = identical top-10)
  - composite + NDCG@10 on the gold-anchored harness

If the top-10 stays stable, the current weighted+multiplicative scoring is robust
and a lexicographic/tiered refactor is unnecessary.

Run:  python scripts/sensitivity.py
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from redrob import io as cio, baseline as B, silver, trust, evalmetrics as M  # noqa: E402
from redrob.jobspec import JobSpec  # noqa: E402


def final_score(comp, a_w, c_w, sw, ff_lo, ff_hi):
    lex, coh, sem, focus, bf, loc, pen, avail, hpf = comp
    align = (1 - sw) * lex + sw * sem
    ff = ff_lo + (ff_hi - ff_lo) * focus
    return (a_w * align + c_w * coh) * ff * bf * (0.85 + 0.15 * loc) * pen * avail * hpf


def top_ids(cache, cfg, n=100):
    scored = sorted(((final_score(c, *cfg), cid) for cid, c in cache),
                    key=lambda x: (-x[0], x[1]))
    return [cid for _, cid in scored[:n]]


def main():
    spec = JobSpec.from_yaml(ROOT / "job_spec.yaml")
    sem_pct = B.load_semantic_pct(ROOT / "artifacts")
    ref = B.reference_date(ROOT / "candidates.jsonl")
    gold = {r[0]: int(r[1]) for r in csv.reader(open(ROOT / "eval/holdout_labels.csv", encoding="utf-8")) if r and r[0].startswith("CAND_")}

    cache, rel, hps = [], {}, set()
    for c in cio.iter_candidates(ROOT / "candidates.jsonl"):
        cid = c["candidate_id"]
        prof, hist, sig = c.get("profile", {}), cio.career_history(c), cio.signals(c)
        nl = cio.narrative_text(c).lower()
        lex, _, _ = B.alignment(nl, spec)
        coh = B.coherence(prof, hist, spec)
        avail, _ = B.availability(sig, ref)
        bf = B.band_fit(prof.get("years_of_experience"), spec)
        loc = B.location_fit(prof, sig, spec)
        ts = trust.signals(c, spec)
        hp = B.is_honeypot(c)
        cache.append((cid, (lex, coh, sem_pct.get(cid, 0.0), ts["focus"], bf, loc,
                            ts["penalty"], avail, 0.02 if hp else 1.0)))
        rel[cid] = silver.silver_tier(c, spec)
        if hp:
            hps.add(cid)
    rel.update(gold)

    default = (0.55, 0.45, 0.20, 0.85, 1.15)
    configs = {
        "default            ": default,
        "align-heavy 0.65/35": (0.65, 0.35, 0.20, 0.85, 1.15),
        "coh-heavy   0.45/55": (0.45, 0.55, 0.20, 0.85, 1.15),
        "sem 0.10           ": (0.55, 0.45, 0.10, 0.85, 1.15),
        "sem 0.35           ": (0.55, 0.45, 0.35, 0.85, 1.15),
        "focus-strong .75/25": (0.55, 0.45, 0.20, 0.75, 1.25),
        "focus-weak  .92/08 ": (0.55, 0.45, 0.20, 0.92, 1.08),
    }
    base_top = top_ids(cache, default)
    base10 = set(base_top[:10])
    print(f"cached {len(cache):,} | gold {len(gold)}\n")
    print(f"{'config':22} {'top10 Jaccard':>14} {'NDCG@10':>8} {'composite':>10}")
    for name, cfg in configs.items():
        ids = top_ids(cache, cfg)
        j = len(base10 & set(ids[:10])) / len(base10 | set(ids[:10]))
        s = M.composite(ids, rel)
        print(f"{name:22} {j:>14.2f} {s['ndcg@10']:>8.4f} {s['composite']:>10.4f}")


if __name__ == "__main__":
    main()
