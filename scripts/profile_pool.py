#!/usr/bin/env python3
"""
General candidate-pool profiler.

Design principle: NOTHING here is hardcoded to a specific candidate or to this
particular JD. Every detector is a universal rule (internal logical consistency,
claim-vs-evidence mismatch, distributional outlier) that would run unchanged on
any Redrob-schema pool. The goal is to (a) understand the pool's structure and
(b) prove the general trap detectors fire correctly on real data.

Run:  python scripts/profile_pool.py [--path candidates.jsonl] [--limit N]
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from datetime import date
import numpy as np

# --- A small seed lexicon used ONLY for profiling/illustration. In the real
# ranker these terms come from the parsed JobSpec, not from a constant here.
AI_SKILL_SEED = {
    "machine learning", "deep learning", "nlp", "natural language processing",
    "embeddings", "retrieval", "ranking", "information retrieval", "rag",
    "transformers", "llm", "pytorch", "tensorflow", "vector search", "faiss",
    "pinecone", "weaviate", "qdrant", "elasticsearch", "recommendation",
    "fine-tuning", "lora", "learning to rank", "semantic search", "bert",
}
# Role tokens that indicate an actual engineering/ML/data practitioner.
TECH_ROLE_TOKENS = {
    "engineer", "developer", "scientist", "ml", "ai", "data", "research",
    "software", "backend", "machine learning", "nlp", "architect", "analyst",
}


def parse_date(s):
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def iter_candidates(path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if limit and i >= limit:
                break
            yield json.loads(line)


# ---------------------------------------------------------------------------
# Universal trap detectors (return a list of reason strings; empty == clean)
# ---------------------------------------------------------------------------
def honeypot_reasons(c, ref_year):
    """Internal-consistency violations that are logically impossible regardless
    of role or dataset. A non-empty list means 'subtly impossible profile'."""
    r = []
    prof = c.get("profile", {})
    yoe = prof.get("years_of_experience")
    hist = c.get("career_history", []) or []
    edu = c.get("education", []) or []
    skills = c.get("skills", []) or []

    # 1. Sum of role tenures wildly exceeds stated years of experience.
    total_months = sum(int(h.get("duration_months") or 0) for h in hist)
    if yoe is not None and total_months > (yoe + 2) * 12 + 6:
        r.append(f"tenure_sum {total_months}mo >> YOE {yoe}y")

    # 2. duration_months inconsistent with start/end dates.
    for h in hist:
        sd, ed = parse_date(h.get("start_date")), parse_date(h.get("end_date"))
        if sd and ed:
            span = (ed.year - sd.year) * 12 + (ed.month - sd.month)
            dm = int(h.get("duration_months") or 0)
            if span < -1 or abs(span - dm) > 18:
                r.append(f"date/duration mismatch @{h.get('company')}")
        if h.get("is_current") and h.get("end_date"):
            r.append(f"is_current but has end_date @{h.get('company')}")
        if sd and sd.year > ref_year:
            r.append(f"future start_date @{h.get('company')}")

    # 3. "Expert" in many skills with ~0 months of actual use.
    expert_zero = [s for s in skills
                   if s.get("proficiency") == "expert"
                   and int(s.get("duration_months") or 0) == 0]
    if len(expert_zero) >= 5:
        r.append(f"{len(expert_zero)} expert skills with 0 months used")

    # 4. Started working before finishing/starting education implausibly early.
    first_start = min((parse_date(h.get("start_date")) for h in hist
                       if parse_date(h.get("start_date"))), default=None)
    edu_start = min((e.get("start_year") for e in edu
                     if isinstance(e.get("start_year"), int)), default=None)
    if first_start and edu_start and first_start.year < edu_start - 1:
        r.append(f"work started {first_start.year} before edu {edu_start}")

    # 5. Career history implies more calendar span than a human age allows for YOE.
    if first_start and yoe is not None:
        span_years = ref_year - first_start.year
        if span_years > yoe + 6:
            r.append(f"career spans {span_years}y but YOE only {yoe}y")
    return r


def is_tech_role(title):
    t = (title or "").lower()
    return any(tok in t for tok in TECH_ROLE_TOKENS)


def keyword_stuffer_score(c):
    """Claim-vs-evidence mismatch: many on-topic skills claimed, but the role
    and corroborating evidence don't support them. Higher == more suspicious."""
    prof = c.get("profile", {})
    skills = c.get("skills", []) or []
    sig = c.get("redrob_signals", {}) or {}
    assess = sig.get("skill_assessment_scores", {}) or {}

    ai_skills = [s for s in skills
                 if any(k in (s.get("name", "").lower()) for k in AI_SKILL_SEED)]
    if len(ai_skills) < 4:
        return 0.0, len(ai_skills)

    # Evidence the claims are real: corroborating role, endorsements, assessment,
    # and actual months of use.
    role_ok = is_tech_role(prof.get("current_title")) or any(
        is_tech_role(h.get("title")) for h in c.get("career_history", []) or [])
    corroborated = 0
    for s in ai_skills:
        ev = (int(s.get("endorsements") or 0) > 0
              or int(s.get("duration_months") or 0) >= 6
              or float(assess.get(s.get("name", ""), 0)) >= 50)
        corroborated += int(ev)
    frac_uncorroborated = 1 - corroborated / max(1, len(ai_skills))
    score = frac_uncorroborated * (0.4 + 0.6 * (not role_ok))
    return round(score, 3), len(ai_skills)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="candidates.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    n = 0
    titles = Counter(); countries = Counter(); sizes = Counter()
    yoe = []; resp = []; complete = []; gh = []; notice = []
    last_active = []; offer_acc = []
    open_to_work = 0; relocate = 0; vmail = 0; vphone = 0; li = 0
    honeypots = []; stuffers = []
    max_active = date(1970, 1, 1)

    # first pass needs a reference year; use a fixed generous ceiling, then refine
    REF_YEAR = date.today().year

    for c in iter_candidates(args.path, args.limit):
        n += 1
        prof = c.get("profile", {})
        sig = c.get("redrob_signals", {}) or {}
        titles[prof.get("current_title", "?")] += 1
        countries[prof.get("country", "?")] += 1
        sizes[prof.get("current_company_size", "?")] += 1
        if isinstance(prof.get("years_of_experience"), (int, float)):
            yoe.append(prof["years_of_experience"])
        if isinstance(sig.get("recruiter_response_rate"), (int, float)):
            resp.append(sig["recruiter_response_rate"])
        if isinstance(sig.get("profile_completeness_score"), (int, float)):
            complete.append(sig["profile_completeness_score"])
        if isinstance(sig.get("github_activity_score"), (int, float)):
            gh.append(sig["github_activity_score"])
        if isinstance(sig.get("notice_period_days"), (int, float)):
            notice.append(sig["notice_period_days"])
        if isinstance(sig.get("offer_acceptance_rate"), (int, float)):
            offer_acc.append(sig["offer_acceptance_rate"])
        open_to_work += int(bool(sig.get("open_to_work_flag")))
        relocate += int(bool(sig.get("willing_to_relocate")))
        vmail += int(bool(sig.get("verified_email")))
        vphone += int(bool(sig.get("verified_phone")))
        li += int(bool(sig.get("linkedin_connected")))
        la = parse_date(sig.get("last_active_date"))
        if la:
            last_active.append(la); max_active = max(max_active, la)

        hr = honeypot_reasons(c, REF_YEAR)
        if hr:
            honeypots.append((c["candidate_id"], prof.get("current_title"), hr))
        ss, n_ai = keyword_stuffer_score(c)
        if ss >= 0.6 and n_ai >= 6:
            stuffers.append((c["candidate_id"], prof.get("current_title"), ss, n_ai))

    def pctl(a, ps=(0, 25, 50, 75, 90, 100)):
        if not a:
            return "n/a"
        a = np.array(a, float)
        return "  ".join(f"p{p}={np.percentile(a, p):.2f}" for p in ps)

    recency = [(max_active - d).days for d in last_active] if last_active else []

    print(f"\n{'='*70}\nPOOL PROFILE  (n={n:,})\n{'='*70}")
    print("\n-- Top current_title archetypes --")
    for t, ct in titles.most_common(20):
        print(f"   {ct:6,}  {t}")
    print(f"\n   distinct titles: {len(titles)}")

    print("\n-- Geography (top countries) --")
    for c_, ct in countries.most_common(8):
        print(f"   {ct:6,}  {c_}")

    print("\n-- Company size --")
    for s, ct in sizes.most_common():
        print(f"   {ct:6,}  {s}")

    print("\n-- Numeric distributions --")
    print(f"   years_of_experience : {pctl(yoe)}")
    print(f"   recruiter_resp_rate : {pctl(resp)}")
    print(f"   completeness_score  : {pctl(complete)}")
    print(f"   github_activity     : {pctl(gh)}   (== -1 means no github)")
    print(f"   notice_period_days  : {pctl(notice)}")
    print(f"   offer_accept_rate   : {pctl(offer_acc)}   (== -1 means no history)")
    print(f"   last_active recency : {pctl(recency)}   (days before pool max {max_active})")

    print("\n-- Boolean signal rates --")
    print(f"   open_to_work     : {open_to_work/n:.1%}")
    print(f"   willing_relocate : {relocate/n:.1%}")
    print(f"   verified_email   : {vmail/n:.1%}")
    print(f"   verified_phone   : {vphone/n:.1%}")
    print(f"   linkedin_conn    : {li/n:.1%}")
    print(f"   github == -1     : {sum(1 for x in gh if x==-1)/max(1,len(gh)):.1%}")
    print(f"   offer_acc == -1  : {sum(1 for x in offer_acc if x==-1)/max(1,len(offer_acc)):.1%}")

    print(f"\n{'='*70}\nTRAP DETECTORS (universal rules)\n{'='*70}")
    print(f"\n-- Honeypot candidates (impossible profiles): {len(honeypots):,} "
          f"({len(honeypots)/n:.2%} of pool) --")
    by_title = Counter(t for _, t, _ in honeypots)
    for t, ct in by_title.most_common(10):
        print(f"   {ct:5}  {t}")
    print("   examples:")
    for cid, t, rs in honeypots[:6]:
        print(f"     {cid} [{t}] -> {rs[:2]}")

    print(f"\n-- Keyword-stuffer suspects (claim>>evidence): {len(stuffers):,} "
          f"({len(stuffers)/n:.2%} of pool) --")
    st_title = Counter(t for _, t, _, _ in stuffers)
    for t, ct in st_title.most_common(12):
        print(f"   {ct:5}  {t}")
    print("   examples:")
    for cid, t, ss, na in sorted(stuffers, key=lambda x: -x[2])[:6]:
        print(f"     {cid} [{t}] stuffer_score={ss} ai_skills={na}")
    print()


if __name__ == "__main__":
    main()
