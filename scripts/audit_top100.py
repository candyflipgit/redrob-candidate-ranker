#!/usr/bin/env python3
"""DQ-insurance: aggressively audit the submitted top-100 for honeypots our
shipped (high-precision) detector might have missed — the organizers score
against THEIR honeypot set, and >10% in the top-100 is a disqualification.

We can afford broader checks here because it's only 100 candidates and every
flag is reviewed by hand. Includes a data-driven company-age estimate (earliest
start-date seen across the whole pool) to catch "N years at an M-year-old
company" with M < N.
"""
from __future__ import annotations
import csv
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from redrob import io as cio  # noqa: E402


def main():
    sub = ROOT / "submission.csv"
    top = {r[0]: int(r[1]) for i, r in enumerate(csv.reader(open(sub, encoding="utf-8"))) if i and r}
    pool = ROOT / "candidates.jsonl"

    # pass 1: company -> earliest start date seen anywhere in the pool; pool max date
    company_first: dict[str, date] = {}
    ref = date(1970, 1, 1)
    for c in cio.iter_candidates(pool):
        for h in cio.career_history(c):
            sd = cio.parse_date(h.get("start_date"))
            comp = (h.get("company") or "").strip()
            if sd and comp:
                if comp not in company_first or sd < company_first[comp]:
                    company_first[comp] = sd
        la = cio.parse_date(cio.signals(c).get("last_active_date"))
        if la and la > ref:
            ref = la

    # pass 2: audit the top-100
    flags = []
    for c in cio.iter_candidates(pool):
        cid = c["candidate_id"]
        if cid not in top:
            continue
        prof = c.get("profile", {})
        hist = cio.career_history(c)
        skills = cio.skills(c)
        yoe = prof.get("years_of_experience")
        reasons = []

        total = sum(int(h.get("duration_months") or 0) for h in hist)
        if isinstance(yoe, (int, float)) and total > (yoe + 2) * 12 + 6:
            reasons.append(f"tenure_sum {total}mo >> YOE {yoe}y")
        for h in hist:
            dm = int(h.get("duration_months") or 0)
            sd, ed = cio.parse_date(h.get("start_date")), cio.parse_date(h.get("end_date"))
            comp = (h.get("company") or "").strip()
            if dm > 150:
                reasons.append(f"{dm}mo single tenure @{comp}")
            if sd and ed and ((ed.year - sd.year) * 12 + ed.month - sd.month) < -1:
                reasons.append(f"end<start @{comp}")
            if h.get("is_current") and h.get("end_date"):
                reasons.append(f"is_current+end_date @{comp}")
            # company-age: tenure exceeds how long the company has plausibly existed
            if comp in company_first and dm > 0:
                age_mo = (ref.year - company_first[comp].year) * 12 + (ref.month - company_first[comp].month)
                if dm > age_mo + 18:
                    reasons.append(f"tenure {dm}mo > company-age ~{age_mo}mo @{comp}")
        ez = sum(1 for s in skills if s.get("proficiency") == "expert" and int(s.get("duration_months") or 0) == 0)
        if ez >= 4:
            reasons.append(f"{ez} expert skills @0 months")

        if reasons:
            flags.append((top[cid], cid, prof.get("current_title"), reasons))

    print(f"audited top-{len(top)} | companies tracked: {len(company_first):,} | pool ref {ref}\n")
    if not flags:
        print("CLEAN: no top-100 candidate trips any impossibility check. Honeypot DQ risk: very low.")
    else:
        print(f"{len(flags)} candidate(s) to review by hand "
              f"({len(flags)/len(top):.0%} of top-100):\n")
        for rank, cid, title, reasons in sorted(flags):
            print(f"  rank {rank:>3}  {cid} [{title}]")
            for r in reasons:
                print(f"            - {r}")


if __name__ == "__main__":
    main()
