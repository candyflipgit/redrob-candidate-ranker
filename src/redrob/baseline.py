"""Text-only baseline ranker (walking skeleton).

Scores every candidate against a JobSpec using only lexical/structural signals —
no embeddings yet. Its job is to prove the end-to-end path (stream -> score ->
top-100 -> valid CSV) and give us an always-working fallback submission that the
embedding + tiered-scoring versions must then beat on the offline eval harness.

The four pillars (see approach memory): JD-alignment and internal-coherence are
the additive core; availability is a multiplicative modifier; the honeypot/trust
checks are a near-zero sink. All logic is general (driven by the JobSpec + the
pool), nothing hardcoded to a specific candidate.
"""
from __future__ import annotations

import math
import re
from collections import namedtuple
from datetime import date

from . import io as cio
from . import trust
from .jobspec import JobSpec

# Stuffer "tell": summaries stitched from a different role than the profile.
_SCRAMBLE_TELL = re.compile(
    r"background is in|i've spent my career in|my professional background|"
    r"i'm a .{0,40}? with substantial experience", re.I)
# Title actually denotes an ML/AI/IR practitioner.
_ML_ROLE = re.compile(
    r"\b(ml|ai|machine learning|nlp|data scientist|applied scientist|"
    r"research engineer|search engineer|recommendation)\b", re.I)

Scored = namedtuple(
    "Scored",
    "score cid title yoe must_hits nice_hits top_terms coherence response "
    "recency_days band_fit honeypot location disq semantic")


def _present(terms, text_l):
    return [t for t in terms if t in text_l]


def alignment(narrative_l: str, spec: JobSpec):
    must = _present(spec.must_have_terms, narrative_l)
    nice = _present(spec.nice_to_have_terms, narrative_l)
    weighted = len(must) + 0.5 * len(nice)
    return 1.0 - math.exp(-weighted / 6.0), must, nice


def coherence(profile: dict, hist: list, spec: JobSpec) -> float:
    summary_l = (profile.get("summary") or "").lower()
    desc_l = " ".join((h.get("description") or "") for h in hist).lower()
    title_l = (profile.get("current_title") or "").lower()
    pos = spec.positive_query()
    summary_jd = any(t in summary_l for t in pos)
    desc_jd = any(t in desc_l for t in pos)
    title_role = bool(_ML_ROLE.search(title_l))
    scramble = 1.0 if _SCRAMBLE_TELL.search(summary_l) else 0.0
    base = 0.40 * summary_jd + 0.40 * desc_jd + 0.20 * title_role
    return base * (1.0 - 0.7 * scramble)


def availability(sig: dict, ref_date: date):
    r = sig.get("recruiter_response_rate")
    r = r if isinstance(r, (int, float)) else 0.3
    la = cio.parse_date(sig.get("last_active_date"))
    recency_days = (ref_date - la).days if (la and ref_date) else 180
    recency_factor = max(0.0, min(1.0, 1.0 - recency_days / 180.0))
    open_w = 1.0 if sig.get("open_to_work_flag") else 0.0
    verified = 0.5 * (bool(sig.get("verified_email")) + bool(sig.get("verified_phone")))
    base = 0.45 * r + 0.35 * recency_factor + 0.15 * open_w + 0.05 * verified
    return 0.40 + 0.70 * base, recency_days  # multiplier in ~[0.40, 1.10]


def band_fit(yoe, spec: JobSpec) -> float:
    if not isinstance(yoe, (int, float)):
        return 0.8
    over = max(0.0, spec.min_years - yoe) + max(0.0, yoe - spec.max_years)
    return 1.0 / (1.0 + 0.25 * over)


def location_fit(profile: dict, sig: dict, spec: JobSpec) -> float:
    loc = f"{profile.get('location','')} {profile.get('country','')}".lower()
    if any(c in loc for c in spec.locations):
        return 1.0
    if spec.relocate_ok and sig.get("willing_to_relocate"):
        return 0.8
    return 0.45


def is_honeypot(c: dict) -> bool:
    """Universal logical-impossibility checks (the refined keepers; the
    work-before-education rule was dropped as a false-positive generator)."""
    prof = c.get("profile", {})
    hist = cio.career_history(c)
    skills = cio.skills(c)
    yoe = prof.get("years_of_experience")

    total = sum(int(h.get("duration_months") or 0) for h in hist)
    if isinstance(yoe, (int, float)) and total > (yoe + 2) * 12 + 6:
        return True
    for h in hist:
        sd, ed = cio.parse_date(h.get("start_date")), cio.parse_date(h.get("end_date"))
        if sd and ed:
            span = (ed.year - sd.year) * 12 + (ed.month - sd.month)
            if span < -1 or abs(span - int(h.get("duration_months") or 0)) > 18:
                return True
        if h.get("is_current") and h.get("end_date"):
            return True
    expert_zero = sum(1 for s in skills
                      if s.get("proficiency") == "expert"
                      and int(s.get("duration_months") or 0) == 0)
    return expert_zero >= 5


SEM_WEIGHT = 0.2   # semantic share of alignment (tuned jointly with the trust layer)

def score_candidate(c: dict, spec: JobSpec, ref_date: date,
                    semantic_pct: float | None = None,
                    sem_weight: float = SEM_WEIGHT) -> Scored:
    prof = c.get("profile", {})
    hist = cio.career_history(c)
    sig = cio.signals(c)
    narrative_l = cio.narrative_text(c).lower()

    align_lex, must, nice = alignment(narrative_l, spec)
    # Blend lexical term-signal with semantic percentile. Earlier, semantic
    # promoted keyword-bearing off-career generalists (CV/services with IR
    # skills) and hurt precision; once the trust layer demotes them, semantic is
    # net-positive again. Joint gold-anchored tuning puts the optimum at ~0.2.
    align = ((1 - sem_weight) * align_lex + sem_weight * semantic_pct
             if semantic_pct is not None else align_lex)
    coh = coherence(prof, hist, spec)
    avail, recency_days = availability(sig, ref_date)
    bf = band_fit(prof.get("years_of_experience"), spec)
    loc = location_fit(prof, sig, spec)
    ts = trust.signals(c, spec)        # graded wrong-domain / services / IR-focus
    hp = is_honeypot(c)

    # focus_factor boosts genuine-IR careers; penalty demotes cv/services
    # generalists who carry IR keywords but not IR careers.
    fit = (0.55 * align + 0.45 * coh) * ts["focus_factor"]
    base = fit * bf * (0.85 + 0.15 * loc) * ts["penalty"]
    final = base * avail * (0.02 if hp else 1.0)

    return Scored(
        score=round(final, 6), cid=c["candidate_id"],
        title=prof.get("current_title") or "?",
        yoe=prof.get("years_of_experience"),
        must_hits=len(must), nice_hits=len(nice), top_terms=must[:4],
        coherence=round(coh, 3), response=sig.get("recruiter_response_rate"),
        recency_days=recency_days, band_fit=round(bf, 3), honeypot=hp,
        location=loc, disq=ts["flags"],
        semantic=round(semantic_pct, 3) if semantic_pct is not None else None)


def reference_date(path) -> date:
    """Pool-relative 'today' = latest last_active_date (self-calibrating)."""
    ref = date(1970, 1, 1)
    for c in cio.iter_candidates(path):
        la = cio.parse_date(cio.signals(c).get("last_active_date"))
        if la and la > ref:
            ref = la
    return ref


def load_semantic_pct(artifacts_dir) -> dict | None:
    """Load precomputed embeddings (numpy only — no torch at rank time) and
    return {candidate_id: semantic percentile in [0,1]} vs the JD query, or None
    if artifacts are absent (lexical-only fallback)."""
    import json
    from pathlib import Path
    d = Path(artifacts_dir)
    ef, idf, qf = d / "cand_embeddings.npy", d / "cand_ids.json", d / "jd_query_emb.npy"
    if not (ef.exists() and idf.exists() and qf.exists()):
        return None
    import numpy as np
    from scipy.stats import rankdata
    emb = np.load(ef)
    q = np.load(qf)[0]
    ids = json.loads(idf.read_text(encoding="utf-8"))
    sims = emb @ q                       # cosine (vectors are L2-normalized)
    pct = rankdata(sims, method="average") / len(sims)
    return {cid: float(p) for cid, p in zip(ids, pct)}


def rank_pool(path, spec: JobSpec, top_n: int = 100, artifacts_dir="artifacts",
              sem_weight: float = SEM_WEIGHT):
    ref = reference_date(path)
    sem_pct = load_semantic_pct(artifacts_dir) if artifacts_dir else None
    scored = []
    for c in cio.iter_candidates(path):
        sp = sem_pct.get(c["candidate_id"]) if sem_pct else None
        scored.append(score_candidate(c, spec, ref, semantic_pct=sp, sem_weight=sem_weight))
    scored.sort(key=lambda s: (-s.score, s.cid))  # ties -> candidate_id ascending
    return scored[:top_n], ref, (sem_pct is not None)


def reasoning(s: Scored) -> str:
    head = f"{s.title}, {s.yoe:.1f}y" if isinstance(s.yoe, (int, float)) else s.title
    parts = [head]
    if s.top_terms:
        parts.append("narrative cites " + ", ".join(s.top_terms[:3]))
    elif s.semantic is not None and s.semantic >= 0.9:
        parts.append("strong semantic match to the role")
    parts.append(f"coherence {s.coherence:.2f}")
    if isinstance(s.response, (int, float)):
        parts.append(f"response {s.response:.2f}, active {s.recency_days}d ago")
    concerns = []
    if s.band_fit < 0.85:
        concerns.append("experience outside 5-9y band")
    if s.disq:
        concerns.append("/".join(s.disq))
    if s.honeypot:
        concerns.append("inconsistent profile (flagged)")
    text = "; ".join(parts)
    if concerns:
        text += ". Concerns: " + ", ".join(concerns)
    return text[:240]
