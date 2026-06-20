"""Ranking metrics — the same yardsticks the challenge scores us on, so we can
grade our own rankings offline (there is no live leaderboard).

All functions take a `ranked_ids` list (best-first) and a `rel` dict
candidate_id -> graded relevance tier (0..5). Relevance is graded for NDCG and
thresholded (tier >= REL_THRESHOLD) for the precision-style metrics, mirroring
the spec ("P@10 = fraction of top-10 that are tier 3+").
"""
from __future__ import annotations

import math

REL_THRESHOLD = 3  # tier >= this counts as "relevant" for MAP / P@k

# Official composite weights (submission_spec section 4).
W_NDCG10, W_NDCG50, W_MAP, W_P10 = 0.50, 0.30, 0.15, 0.05


def _dcg(gains) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids, rel: dict, k: int) -> float:
    """Graded NDCG@k with linear gain (sklearn-style). Ideal ordering is taken
    over the labelled items in the ranked list."""
    gains = [rel.get(c, 0) for c in ranked_ids[:k]]
    ideal = sorted((rel.get(c, 0) for c in ranked_ids), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def average_precision(ranked_ids, rel: dict, threshold: int = REL_THRESHOLD) -> float:
    relevant = sum(1 for c in ranked_ids if rel.get(c, 0) >= threshold)
    if relevant == 0:
        return 0.0
    hits, acc = 0, 0.0
    for i, c in enumerate(ranked_ids, 1):
        if rel.get(c, 0) >= threshold:
            hits += 1
            acc += hits / i
    return acc / relevant


def precision_at_k(ranked_ids, rel: dict, k: int, threshold: int = REL_THRESHOLD) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for c in ranked_ids[:k] if rel.get(c, 0) >= threshold) / k


def composite(ranked_ids, rel: dict) -> dict:
    n10 = ndcg_at_k(ranked_ids, rel, 10)
    n50 = ndcg_at_k(ranked_ids, rel, 50)
    ap = average_precision(ranked_ids, rel)
    p10 = precision_at_k(ranked_ids, rel, 10)
    comp = W_NDCG10 * n10 + W_NDCG50 * n50 + W_MAP * ap + W_P10 * p10
    return {"ndcg@10": n10, "ndcg@50": n50, "map": ap, "p@10": p10,
            "composite": comp}


def honeypot_rate(top_ids, honeypot_ids) -> float:
    if not top_ids:
        return 0.0
    hp = set(honeypot_ids)
    return sum(1 for c in top_ids if c in hp) / len(top_ids)
