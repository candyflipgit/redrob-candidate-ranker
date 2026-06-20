#!/usr/bin/env python3
"""Hunt for internally-COHERENT, JD-aligned candidates (the true positives) and
measure cross-field incoherence (the trap signature). General heuristics only.

Hypothesis under test: real fits score high on JD-intent signal *in their free
text* (summary + every career description) AND are internally coherent (title /
summary / descriptions agree). Traps have keywords in the skills list but
contradict themselves across narrative fields.
"""
from __future__ import annotations
import json, re
from collections import Counter

# JD-intent terms (the senior-AI-engineer archetype). Multi-word aware.
JD_TERMS = [
    "retrieval", "re-rank", "rerank", "ranking", "recommendation", "recommender",
    "embedding", "vector search", "vector database", "vector db", "semantic search",
    "information retrieval", "learning to rank", "ndcg", "mrr", "bm25", "faiss",
    "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch",
    "rag", "llm", "fine-tun", "sentence-transformer", "sentence transformer",
    "search relevance", "candidate matching", "matching system", "recsys",
    "transformer", "bert", "nlp", "natural language",
]
# Summary "tell" that the profile is a scrambled non-fit.
SCRAMBLE_TELL = re.compile(
    r"background is in|i've spent my career in|my professional background", re.I)
TECH_TITLE = re.compile(
    r"engineer|developer|scientist|\bml\b|\bai\b|machine learning|nlp|architect|data",
    re.I)


def jd_signal(text):
    t = text.lower()
    return sum(t.count(term) for term in JD_TERMS)


def main():
    titles = Counter()
    rows = []
    with open("candidates.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            p = c.get("profile", {})
            titles[p.get("current_title", "?")] += 1
            summary = p.get("summary", "") or ""
            descs = " ".join((h.get("description") or "")
                             for h in c.get("career_history", []) or [])
            headline = p.get("headline", "") or ""
            narrative = f"{summary} {descs} {headline}"

            sig_summary = jd_signal(summary)
            sig_descs = jd_signal(descs)
            # coherence: JD signal must appear in BOTH summary and descriptions,
            # not just one field; and title should be technical; and summary must
            # not contain the scramble tell.
            coherent = (sig_summary > 0 and sig_descs > 0
                        and TECH_TITLE.search(p.get("current_title", ""))
                        and not SCRAMBLE_TELL.search(summary))
            total = sig_summary + sig_descs
            rows.append((total, sig_summary, sig_descs, coherent,
                         c["candidate_id"], p.get("current_title"),
                         p.get("years_of_experience"), summary[:140]))

    print(f"\n-- ALL {len(titles)} titles by count --")
    for t, ct in titles.most_common():
        print(f"   {ct:6,}  {t}")

    coherent_rows = [r for r in rows if r[3]]
    print(f"\n-- Coherent + JD-aligned candidates: {len(coherent_rows):,} "
          f"of {len(rows):,} --")
    coherent_rows.sort(key=lambda r: -r[0])
    print("\nTop 25 coherent candidates by JD free-text signal:")
    for total, ss, sd, _, cid, title, yoe, summ in coherent_rows[:25]:
        print(f"   {cid} [{title}, {yoe}y] sig={total} (sum={ss},desc={sd})")
        print(f"       {summ}")

    # Distribution of total JD signal across whole pool
    import numpy as np
    arr = np.array([r[0] for r in rows])
    print(f"\n-- JD free-text signal distribution (whole pool) --")
    for p_ in (50, 75, 90, 95, 99, 99.9, 100):
        print(f"   p{p_}: {np.percentile(arr, p_):.1f}")
    print(f"   candidates with signal >= 5 : {(arr>=5).sum():,}")
    print(f"   candidates with signal >= 8 : {(arr>=8).sum():,}")


if __name__ == "__main__":
    main()
