# Redrob Intelligent Candidate Discovery — Ranking Engine

A candidate-ranking system for the **India Runs Data & AI Challenge (Track 1)**.
Given a job description and a pool of candidate profiles, it returns the top 100
best-fit candidates, ranked, each with a short human-readable reason.

## The idea in one line

> Read each candidate's **whole story** and check it's **internally consistent and
> available**, rather than counting buzzwords — and do it for *any* JD and *any*
> pool, fast, on CPU.

The dataset is adversarial by design: thousands of profiles are stitched from
mismatched parts (a "Marketing Manager" summary with a random AI skill list), and
~80 are logically impossible "honeypots". Keyword/embedding matching alone walks
straight into these. We win by scoring **JD-alignment × internal-coherence ×
availability**, with a logic-based trust filter.

## Architecture (two phases)

1. **Offline pre-compute** (untimed): parse the JD into a structured `JobSpec`;
   embed every candidate's narrative text into vectors; save to `artifacts/`.
2. **Ranking run** (CPU, < 5 min, ≤ 16 GB, **no network**): load precomputed
   vectors + features → hybrid recall → tiered re-rank → honeypot sink →
   top-100 CSV with reasons.

Retrieve → re-rank → filter → explain → evaluate. Standard production IR shape.

## Reproduce the submission

```bash
# 1. ranking-step deps only (what gets reproduced at Stage 3)
pip install -r requirements.txt

# 2. (offline, once) precompute embeddings — needs the heavier deps
pip install -r requirements-precompute.txt
python scripts/precompute.py --candidates candidates.jsonl   # writes artifacts/

# 3. produce the ranking (CPU, no network)
python scripts/rank.py --candidates candidates.jsonl --out submission.csv

# 4. validate format before submitting
python validate_submission.py submission.csv
```

## Repo layout

```
src/redrob/        core library (io, jobspec, features, scoring, reasoning, eval)
scripts/           CLI entry points (profile_pool, precompute, rank) + smoke_test
eval/              silver labels + offline metrics
artifacts/         generated embeddings/features (gitignored)
```

## Status

Under active development for the hackathon — see the task board / commit history
for iteration. Built walking-skeleton first (text-only baseline → valid CSV),
then layered embeddings, tiered scoring, and the trust layer, each measured
against an offline NDCG/MAP harness (there is no live leaderboard).
