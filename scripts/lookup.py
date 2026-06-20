#!/usr/bin/env python3
"""Dump key fields for specific candidate_ids, to calibrate detectors against
real examples. Usage: python scripts/lookup.py CAND_0004989 CAND_0003114 ..."""
from __future__ import annotations
import json, sys

WANT = set(sys.argv[1:])
out = []
with open("candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        # cheap pre-filter before json.loads
        if not any(w in line for w in WANT):
            continue
        c = json.loads(line)
        if c["candidate_id"] not in WANT:
            continue
        p = c.get("profile", {})
        sig = c.get("redrob_signals", {}) or {}
        out.append(f"\n{'='*72}")
        out.append(f"{c['candidate_id']}  |  {p.get('current_title')}  |  "
                   f"{p.get('years_of_experience')}y  |  {p.get('location')}, {p.get('country')}")
        out.append(f"headline: {p.get('headline')}")
        out.append(f"summary : {(p.get('summary') or '')[:300]}")
        out.append("career_history (title @ company, dur):")
        for h in c.get("career_history", []) or []:
            out.append(f"   - {h.get('title')} @ {h.get('company')} "
                       f"({h.get('duration_months')}mo, {h.get('industry')}, {h.get('company_size')})")
            out.append(f"       {(h.get('description') or '')[:160]}")
        out.append("skills (name | prof | dur_mo | endorse):")
        for s in c.get("skills", []) or []:
            out.append(f"   - {s.get('name'):28} | {s.get('proficiency'):12} | "
                       f"{str(s.get('duration_months')):>4} | {s.get('endorsements')}")
        assess = sig.get("skill_assessment_scores", {}) or {}
        out.append(f"assessments: {assess}")
        out.append("signals: "
                   f"resp={sig.get('recruiter_response_rate')} "
                   f"last_active={sig.get('last_active_date')} "
                   f"open={sig.get('open_to_work_flag')} relocate={sig.get('willing_to_relocate')} "
                   f"gh={sig.get('github_activity_score')} notice={sig.get('notice_period_days')} "
                   f"complete={sig.get('profile_completeness_score')}")
        out.append(f"education: " + "; ".join(
            f"{e.get('degree')} {e.get('field_of_study')} ({e.get('start_year')}-{e.get('end_year')}, {e.get('tier')})"
            for e in c.get("education", []) or []))

with open("_lookup_out.txt", "w", encoding="utf-8") as fo:
    fo.write("\n".join(out))
print(f"wrote {len(WANT)} requested, found details -> _lookup_out.txt")
