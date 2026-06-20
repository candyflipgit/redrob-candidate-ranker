#!/usr/bin/env python3
"""OFFLINE pre-computation: embed the JD query + every candidate narrative.

Untimed step (may use network once to download the model). Produces static .npy
artifacts that the CPU/numpy-only ranking step loads — torch is never imported
at rank time.

    python scripts/precompute.py --candidates candidates.jsonl

Writes to artifacts/:  cand_embeddings.npy, cand_ids.json, jd_query_emb.npy,
embed_meta.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from redrob import io as cio, embed  # noqa: E402
from redrob.jobspec import JobSpec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--spec", default="job_spec.yaml")
    ap.add_argument("--model", default=embed.DEFAULT_MODEL)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(exist_ok=True)
    spec = JobSpec.from_yaml(ROOT / args.spec)
    cand_path = ROOT / args.candidates

    t0 = time.time()
    ids, narratives = [], []
    for c in cio.iter_candidates(cand_path, limit=args.limit):
        ids.append(c["candidate_id"])
        narratives.append(cio.narrative_text(c))
    t_load = time.time() - t0
    print(f"loaded {len(ids):,} narratives in {t_load:.1f}s; embedding with {args.model} ...")

    jd_q = embed.jd_query_text(spec)
    jd_emb = embed.embed([jd_q], is_query=True, model_name=args.model)

    t1 = time.time()
    cand_emb = embed.embed(narratives, is_query=False, model_name=args.model,
                           batch_size=args.batch, show=True)
    t_embed = time.time() - t1

    np.save(out / "cand_embeddings.npy", cand_emb)
    np.save(out / "jd_query_emb.npy", jd_emb)
    (out / "cand_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    (out / "embed_meta.json").write_text(json.dumps({
        "model": args.model, "dim": int(cand_emb.shape[1]), "n": len(ids),
        "jd_query": jd_q, "embed_seconds": round(t_embed, 1),
    }, indent=2), encoding="utf-8")

    print(f"embedded {len(ids):,} candidates in {t_embed:.1f}s "
          f"({len(ids)/max(t_embed,0.1):.0f}/s)")
    print(f"saved -> {out}/  (cand_embeddings {cand_emb.shape}, "
          f"{cand_emb.nbytes/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
