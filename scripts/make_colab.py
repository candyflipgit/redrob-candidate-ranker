#!/usr/bin/env python3
"""Generate a SELF-CONTAINED redrob_colab_demo.ipynb.

The notebook writes the ranker's source (read from src/redrob here), the JobSpec,
and the sample into itself via %%writefile cells, then runs the full pipeline.
It depends on nothing external (no HF Space, no GitHub) — bulletproof + private.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULES = ["__init__", "io", "jobspec", "embed", "trust", "baseline", "reasoning"]


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def writefile_cell(path, content):
    return code(f"%%writefile {path}\n{content}")


cells = [md(
    "# Redrob Candidate Ranker — Colab demo (self-contained)\n"
    "\n"
    "Ranks candidates against the **Senior AI Engineer** JD using internal "
    "coherence + JD-alignment + a trust layer (career ≠ keywords), not keyword "
    "counting. Everything is embedded below — no external downloads of code.\n"
    "\n"
    "**Runtime → Run all.** Takes ~1 min (installs the embedding model).")]

cells.append(code(
    "!pip install -q sentence-transformers pyyaml\n"
    "import os\n"
    "os.makedirs('src/redrob', exist_ok=True)\n"
    "print('environment ready')"))

for m in MODULES:
    content = (ROOT / "src" / "redrob" / f"{m}.py").read_text(encoding="utf-8")
    cells.append(writefile_cell(f"src/redrob/{m}.py", content))

cells.append(writefile_cell("job_spec.yaml",
                            (ROOT / "job_spec.yaml").read_text(encoding="utf-8")))
cells.append(writefile_cell("sample.json",
                            (ROOT / "sample_candidates.json").read_text(encoding="utf-8")))

cells.append(code(
    "import sys; sys.path.insert(0, 'src')\n"
    "import pandas as pd\n"
    "from datetime import date\n"
    "from scipy.stats import rankdata\n"
    "from redrob import io as cio, baseline, reasoning, embed\n"
    "from redrob.jobspec import JobSpec\n"
    "\n"
    "spec = JobSpec.from_yaml('job_spec.yaml')\n"
    "cands = cio.load_candidates('sample.json')   # swap for your own file (<=100)\n"
    "\n"
    "dates = [cio.parse_date(cio.signals(c).get('last_active_date')) for c in cands]\n"
    "ref = max([d for d in dates if d], default=date.today())\n"
    "q = embed.embed([embed.jd_query_text(spec)], is_query=True)[0]\n"
    "E = embed.embed([cio.narrative_text(c) for c in cands])\n"
    "pct = rankdata(E @ q) / len(cands)\n"
    "sem = {c['candidate_id']: float(p) for c, p in zip(cands, pct)}\n"
    "\n"
    "scored = sorted(\n"
    "    [baseline.score_candidate(c, spec, ref, semantic_pct=sem[c['candidate_id']]) for c in cands],\n"
    "    key=lambda s: (-s.score, s.cid))\n"
    "by_id = {c['candidate_id']: c for c in cands}\n"
    "rows = [{'rank': i, 'candidate_id': s.cid, 'title': s.title, 'years': s.yoe,\n"
    "         'score': round(s.score, 4), 'reasoning': reasoning.generate(by_id[s.cid], s, spec, i)}\n"
    "        for i, s in enumerate(scored, 1)]\n"
    "df = pd.DataFrame(rows); df.to_csv('ranked_output.csv', index=False); df"))

cells.append(md(
    "**Rank your own pool:** upload a JSON array or JSONL (≤100 candidates in the "
    "Redrob schema) via the file panel (left), change `sample.json` above to your "
    "filename, and re-run. Output is also saved to `ranked_output.csv`."))

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = ROOT / "redrob_colab_demo.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out}  ({out.stat().st_size//1024} KB, {len(cells)} cells)")
