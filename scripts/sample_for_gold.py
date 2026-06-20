#!/usr/bin/env python3
"""Select a stratified sample for the hand-read GOLD holdout and dump compact,
truth-focused profiles (raw summary + career descriptions) so they can be judged
independently of the ranker's own features.

Strata: our submission top-12, detected honeypots, sample_submission's top
(confirmed stuffers), and boundary cases at silver tiers 2/3/4.

Run:  python scripts/sample_for_gold.py  ->  writes _gold_sample.txt + the id list
"""
from __future__ import annotations
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from redrob import io as cio, silver  # noqa: E402
from redrob.jobspec import JobSpec  # noqa: E402

random.seed(7)


def top_ids(path, n):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0 or not row:
                continue
            out.append(row[0].strip())
            if len(out) >= n:
                break
    return out


def main():
    spec = JobSpec.from_yaml(ROOT / "job_spec.yaml")
    sub_top = top_ids(ROOT / "submission.csv", 12)
    stuffers = top_ids(ROOT / "sample_submission.csv", 10)

    want = set(sub_top) | set(stuffers)
    buckets = {2: [], 3: [], 4: [], "hp": []}
    cache = {}

    for c in cio.iter_candidates(ROOT / "candidates.jsonl"):
        cid = c["candidate_id"]
        if cid in want:
            cache[cid] = c
        t = silver.silver_tier(c, spec)
        if silver.is_honeypot(c) and len(buckets["hp"]) < 6:
            buckets["hp"].append(cid); cache[cid] = c
        elif t in (2, 3, 4):
            if len(buckets[t]) < 5 and random.random() < 0.02:
                buckets[t].append(cid); cache[cid] = c

    selected = list(dict.fromkeys(
        sub_top + stuffers + buckets[4] + buckets[3] + buckets[2] + buckets["hp"]))

    lines = []
    for cid in selected:
        c = cache.get(cid)
        if not c:
            continue
        p = c.get("profile", {})
        sig = cio.signals(c)
        tier = silver.silver_tier(c, spec)
        hp = silver.is_honeypot(c)
        tag = ("SUBTOP" if cid in sub_top else
               "STUFFER?" if cid in stuffers else
               "HONEYPOT?" if hp else f"silver{tier}")
        lines.append(f"\n[{tag}] {cid} | {p.get('current_title')} | "
                     f"{p.get('years_of_experience')}y | {p.get('location')},"
                     f"{p.get('country')} | silver={tier} hp={hp}")
        lines.append(f"  SUM: {(p.get('summary') or '')[:160]}")
        for h in cio.career_history(c):
            lines.append(f"  JOB: {h.get('title')} @ {h.get('company')} "
                         f"({h.get('duration_months')}mo,{h.get('industry')}): "
                         f"{(h.get('description') or '')[:120]}")
        lines.append(f"  SIG: resp={sig.get('recruiter_response_rate')} "
                     f"active={sig.get('last_active_date')} open={sig.get('open_to_work_flag')} "
                     f"relocate={sig.get('willing_to_relocate')}")

    (ROOT / "_gold_sample.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"selected {len(selected)} candidates -> _gold_sample.txt")
    print("ids:", ",".join(selected))


if __name__ == "__main__":
    main()
