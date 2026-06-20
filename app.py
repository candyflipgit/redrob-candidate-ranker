"""Gradio sandbox for the Redrob candidate ranker (HuggingFace Spaces demo).

Embeds a small candidate sample LIVE (MiniLM-L6) and runs the full ranking
pipeline end-to-end on CPU — the Stage-3 small-sample reproducibility check.
Uses the bundled 50-candidate sample by default, or an uploaded file (<=100).
"""
from __future__ import annotations

import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import gradio as gr
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from redrob import io as cio, baseline, reasoning, embed
from redrob.jobspec import JobSpec

SPEC = JobSpec.from_yaml(os.path.join(ROOT, "job_spec.yaml"))
DEFAULT_SAMPLE = os.path.join(ROOT, "sample_candidates.json")


def _reference_date(cands):
    ref = date(1970, 1, 1)
    for c in cands:
        la = cio.parse_date(cio.signals(c).get("last_active_date"))
        if la and la > ref:
            ref = la
    return ref if ref.year > 1970 else date.today()


def _rank(cands, top_n):
    ref = _reference_date(cands)
    narratives = [cio.narrative_text(c) for c in cands]
    q = embed.embed([embed.jd_query_text(SPEC)], is_query=True)[0]
    mat = embed.embed(narratives)
    sims = mat @ q
    pct = (rankdata(sims) / len(sims)) if len(sims) > 1 else np.array([1.0])
    sem = {c["candidate_id"]: float(p) for c, p in zip(cands, pct)}

    scored = [baseline.score_candidate(c, SPEC, ref, semantic_pct=sem[c["candidate_id"]])
              for c in cands]
    scored.sort(key=lambda s: (-s.score, s.cid))
    by_id = {c["candidate_id"]: c for c in cands}

    rows = []
    for i, s in enumerate(scored[:int(top_n)], 1):
        rows.append({
            "rank": i, "candidate_id": s.cid, "title": s.title,
            "years": s.yoe, "score": round(s.score, 4),
            "reasoning": reasoning.generate(by_id[s.cid], s, SPEC, i),
        })
    return pd.DataFrame(rows)


def run(file, top_n):
    path = DEFAULT_SAMPLE
    if file is not None:
        path = file if isinstance(file, str) else getattr(file, "name", DEFAULT_SAMPLE)
    cands = cio.load_candidates(path)[:100]
    return _rank(cands, top_n)


with gr.Blocks(title="Redrob Candidate Ranker") as demo:
    gr.Markdown(
        "# Redrob — Intelligent Candidate Discovery\n"
        "Ranks candidates against the **Senior AI Engineer** JD using internal "
        "coherence + JD-alignment + a trust layer (career ≠ keywords), not keyword "
        "counting. Upload a small candidates file (≤100; JSON array or JSONL) or "
        "use the bundled 50-candidate sample.")
    with gr.Row():
        file_in = gr.File(label="candidates (.json / .jsonl) — optional",
                          file_types=[".json", ".jsonl"])
        top_n = gr.Slider(5, 100, value=20, step=5, label="top N")
    btn = gr.Button("Rank", variant="primary")
    out = gr.Dataframe(label="Ranked shortlist", wrap=True)
    btn.click(run, inputs=[file_in, top_n], outputs=out)
    demo.load(run, inputs=[file_in, top_n], outputs=out)


if __name__ == "__main__":
    # ssr_mode=False: Gradio 5/6 SSR can hang at APP_STARTING on HF Spaces
    demo.launch(ssr_mode=False)
