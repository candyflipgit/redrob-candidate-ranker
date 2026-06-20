"""Silver relevance labeller — a *fit-based* tiering model, deliberately built
differently from the ranker so the eval isn't purely circular.

Two principles:
  1. Relevance = FIT (can this person do the job), NOT availability. The hidden
     ground-truth tiers are about quality/fit; availability is a ranking nudge.
     So this labeller ignores response-rate / recency / open-to-work entirely.
  2. It scores the ARCHETYPE structurally (role in title + career, retrieval/
     ranking focus, coherence, band, product-vs-services) and forces honeypots
     to tier 0, matching the spec.

Silver is a weak, large-coverage signal. The trusted metric is the hand-read
gold holdout (eval/holdout_labels.csv); silver is used for coverage and for
the pool-wide honeypot/tier picture. Where they're both available, gold wins.
"""
from __future__ import annotations

from . import io as cio
from .baseline import coherence, is_honeypot, _ML_ROLE
from .jobspec import JobSpec

# The retrieval/ranking heart of this kind of role.
_IR_FOCUS = ["retrieval", "ranking", "re-rank", "rerank", "recommendation",
             "recommender", "recsys", "semantic search", "vector search",
             "information retrieval", "learning to rank", "embedding"]


def silver_tier(c: dict, spec: JobSpec) -> int:
    """Graded fit tier 0..5."""
    if is_honeypot(c):
        return 0

    prof = c.get("profile", {})
    hist = cio.career_history(c)
    narrative_l = cio.narrative_text(c).lower()
    title_l = (prof.get("current_title") or "").lower()
    yoe = prof.get("years_of_experience")

    coh = coherence(prof, hist, spec)
    must_hits = sum(1 for t in spec.must_have_terms if t in narrative_l)
    ir_focus = any(t in narrative_l for t in _IR_FOCUS)
    role_in_title = bool(_ML_ROLE.search(title_l))
    role_in_career = sum(1 for h in hist if _ML_ROLE.search((h.get("title") or "")))
    scramble_gap = coh < 0.3  # coherence already encodes the scramble penalty

    fit = 0.0
    fit += 2.0 if coh >= 0.8 else (1.0 if coh >= 0.5 else 0.0)
    fit += 1.0 if ir_focus else 0.0
    fit += 1.0 if must_hits >= 8 else (0.5 if must_hits >= 4 else 0.0)
    fit += 1.0 if role_in_title else (0.5 if role_in_career >= 1 else 0.0)
    if isinstance(yoe, (int, float)):
        fit += 0.5 if 5 <= yoe <= 9 else (0.25 if 4 <= yoe <= 10 else 0.0)

    # fit penalties (services-only / wrong-domain / scrambled)
    svc = spec.disqualifiers.get("services_only")
    if svc:
        comps = [(h.get("company") or "").lower() for h in hist]
        if comps and all(any(s in cmp for s in svc) for cmp in comps):
            fit -= 0.5
    wd = spec.disqualifiers.get("wrong_domain")
    if wd and any(t in narrative_l for t in wd) and not ir_focus:
        fit -= 1.0
    if scramble_gap:
        fit -= 1.0

    return max(0, min(5, round(fit)))


def label_pool(path, spec: JobSpec, only: set | None = None):
    """Return {candidate_id: tier} and a set of honeypot ids. If `only` is given,
    label just those ids (fast path for grading a submission)."""
    rel, honeypots = {}, set()
    for c in cio.iter_candidates(path):
        cid = c["candidate_id"]
        if only is not None and cid not in only:
            continue
        if is_honeypot(c):
            honeypots.add(cid)
        rel[cid] = silver_tier(c, spec)
        if only is not None and len(rel) == len(only):
            break
    return rel, honeypots
