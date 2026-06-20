#!/usr/bin/env python3
"""Generate redrob_colab_demo.ipynb — a self-running Colab sandbox.

Clones the PUBLIC HuggingFace Space repo (code + sample, no token needed),
installs sentence-transformers, and runs the full ranking pipeline end-to-end on
CPU. A valid sandbox per submission_spec 10.5.
"""
import json
from pathlib import Path

setup = r'''# Clone the public code + sample and install the embedding library
!git clone -q https://huggingface.co/spaces/candyfliphf/redrob-ranker
!pip install -q sentence-transformers
import os, sys
os.chdir('/content/redrob-ranker')
sys.path.insert(0, '/content/redrob-ranker/src')
print('ready')'''

rank = r'''import pandas as pd
from datetime import date
from scipy.stats import rankdata
from redrob import io as cio, baseline, reasoning, embed
from redrob.jobspec import JobSpec

spec = JobSpec.from_yaml('job_spec.yaml')
cands = cio.load_candidates('sample_candidates.json')[:100]   # or upload your own (<=100)

dates = [cio.parse_date(cio.signals(c).get('last_active_date')) for c in cands]
ref = max([d for d in dates if d], default=date.today())

q = embed.embed([embed.jd_query_text(spec)], is_query=True)[0]
E = embed.embed([cio.narrative_text(c) for c in cands])
pct = rankdata(E @ q) / len(cands)
sem = {c['candidate_id']: float(p) for c, p in zip(cands, pct)}

scored = sorted(
    [baseline.score_candidate(c, spec, ref, semantic_pct=sem[c['candidate_id']]) for c in cands],
    key=lambda s: (-s.score, s.cid))
by_id = {c['candidate_id']: c for c in cands}
rows = [{'rank': i, 'candidate_id': s.cid, 'title': s.title, 'years': s.yoe,
         'score': round(s.score, 4), 'reasoning': reasoning.generate(by_id[s.cid], s, spec, i)}
        for i, s in enumerate(scored[:20], 1)]
df = pd.DataFrame(rows)
df.to_csv('ranked_output.csv', index=False)
df'''


def src(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": src(
            "# Redrob Candidate Ranker — Colab demo\n"
            "\n"
            "Ranks candidates against the **Senior AI Engineer** JD using internal "
            "coherence + JD-alignment + a trust layer (career ≠ keywords), not "
            "keyword counting. Full pipeline, end-to-end, on CPU.\n"
            "\n"
            "**Runtime → Run all.** First cell takes ~1 min (install + model).")},
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": src(setup)},
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": src(rank)},
        {"cell_type": "markdown", "metadata": {}, "source": src(
            "**Rank your own pool:** upload a JSON array or JSONL (≤100 candidates "
            "in the Redrob schema) via the Colab file panel (left), change "
            "`sample_candidates.json` above to your filename, and re-run the cell. "
            "The ranked shortlist is also saved to `ranked_output.csv`.")},
    ],
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path(__file__).resolve().parent.parent / "redrob_colab_demo.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out}")
