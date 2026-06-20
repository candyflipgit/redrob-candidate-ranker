# Redrob Intelligent Candidate Discovery — Ranking Engine

Track-1 submission for the **India Runs Data & AI Challenge** (Hack2skill × Redrob AI).
Given a job description and a pool of candidate profiles, it returns the top-100
best-fit candidates, ranked, each with a short, honest, fact-grounded reason.

> **One line:** read each candidate's *whole story*, check it is internally
> consistent and the person is actually available — instead of counting
> buzzwords — and do it for *any* JD and *any* pool, fast, on CPU.

---

## 1. The problem, and the insight that drives everything

The dataset is adversarial by design. We verified this directly: most profiles
are **stitched from mismatched parts** — a "Project Manager" whose summary says
"marketing manager", whose job descriptions are about graphic design, with a
random list of AI skills bolted on. The bundled `sample_submission.csv` is the
*naive* answer — it ranks these keyword-stuffers #1.

So keyword and even pure-embedding matching lose. The master signal is
**internal coherence + JD-alignment across all fields at once**. A real Senior AI
Engineer is a profile where title, summary, *every* career description, skills,
and education tell **one consistent story** that matches the JD's archetype.

The pool decomposes into five groups, and the system handles each:

| Group | Signature | Handling |
|---|---|---|
| Elite fits | coherent "production ML, search/retrieval/ranking", 5–9y, product cos | rank top |
| Keyword-stuffers | scrambled fields, AI terms only in the skills list | coherence demotes |
| IR-keyword generalists | CV / services / forecasting career, IR *skills* but not IR *career* | trust layer demotes |
| Lightweight generalists | self-describe work as "lighter weight than ranking", "classical methods" | depth penalty demotes |
| Honeypots (~80) | logically impossible (bad dates, tenure ≫ experience, expert+0-months) | logic sink (0% in top-100) |

## 2. Architecture

Two phases, split so the ranking step is fast, CPU-only, and offline.

```
OFFLINE (untimed)                          RANKING STEP (CPU, numpy, ~35s, no network)
  job_description ─► JobSpec (faceted)        load JobSpec + embeddings (.npy)
  candidates ─► narrative embeddings ──────►  per candidate:
               (artifacts/*.npy)                alignment = 0.3·semantic⊕0.7·lexical
                                                × coherence (cross-field agreement)
                                                × trust  (career≠keywords; depth)
                                                × availability  × band/location
                                                − honeypot logic-sink
                                              → sort → top-100 + grounded reasons → CSV
```

Standard production IR shape: **retrieve → re-rank → filter → explain → evaluate.**
The ranking step depends only on numpy/scipy + precomputed vectors — no torch — so
it reproduces inside the Stage-3 sandbox trivially.

## 3. How each component works

- **JobSpec** (`jobspec.py`) — parses *any* JD into a structured, faceted spec
  (role terms, must/nice-have concepts, disqualifiers, experience band, locations)
  by intersecting a general IR/ML domain ontology with the JD's own section
  structure. Nothing is hardcoded to this posting.
- **Alignment** (`baseline.py`, `embed.py`) — hybrid: a lexical JD-term signal on
  the candidate *narrative* blended with the percentile of semantic cosine
  similarity (MiniLM-L6 embeddings) to a focused JD query. Narrative-based, not
  skills-based, because skills are the easiest field to fake.
- **Coherence** (`baseline.py`) — rewards JD signal appearing across *both* the
  summary and the job descriptions and a genuine ML/IR job title; penalizes the
  "scramble tell" (summary describing a different profession than the title).
- **Trust layer** (`trust.py`) — the key separator of an IR *career* from IR
  *keywords*: graded penalties for cv/speech skill-dominance, services-career
  fraction, and self-described shallow work; a boost for **career-IR-focus** (do
  the job *descriptions* actually describe retrieval/ranking?).
- **Availability** (`baseline.py`) — a multiplicative modifier from the Redrob
  behavioral signals (recruiter response rate, last-active recency, open-to-work,
  verification). A perfect-on-paper candidate who has gone dark is down-weighted.
- **Honeypot sink** (`baseline.py`) — high-precision internal-date logic only.
  Forces impossible profiles to the bottom. (We *tried* cross-field value checks
  and reverted them — see §4.)
- **Scoring** — additive core (alignment, coherence) × multiplicative dealbreakers
  (availability, trust, band/location) − honeypot sink. The multiplicative
  structure makes it **tier-like by construction**, and we verified the top-10 is
  invariant under ±20% weight perturbation (`sensitivity.py`) — so no fragile
  hand-tuning, and a lexicographic refactor was unnecessary.

## 4. Methodology — measured, not guessed

There is no live leaderboard and only three submissions, so the **offline eval
harness is the most important deliverable.** We grade every change against
hand-read **gold labels** (`eval/holdout_labels.csv`, 68 graded profiles read
independently of the model) plus a silver fit-tier model for coverage; metrics
are NDCG@10/@50, MAP, P@10, and the official composite (`evalmetrics.py`).

Findings that shaped the system — each is a measured experiment, not a hunch:

1. **Embeddings first *hurt*.** A generic JD query made semantic blending look
   great on silver labels (0.978); once we hand-labeled the keyword generalists,
   gold-anchored eval showed it *promoted* them. A focused JD query + the trust
   layer flipped semantic back to net-positive.
2. **The trust layer** (career ≠ keywords) was the single biggest gain — it
   removed the CV/services and lightweight generalists from the top-100.
3. **Broadening honeypot detection backfired** — cross-field value checks
   (summary-claimed years vs the YOE field) flagged 13k candidates including real
   elites, because this pool's fields are *deliberately scrambled*. Reverted to
   high-precision date logic. (Documented as a deliberate decision.)

## 5. Results (offline, gold-anchored)

| metric | value |
|---|---|
| NDCG@10 | 1.00 (top-10 all hand-verified tier-5) |
| NDCG@50 | ~0.95 |
| MAP / P@10 | 1.00 / 1.00 |
| Composite | ~0.985 |
| Honeypot rate (top-100) | 0% (DQ threshold 10%) |
| Runtime | ~35s for 100K, CPU-only, ≪16GB |

**Honest caveat:** these are anchored on 68 hand labels (top-35 well-covered);
deeper ranks carry some silver-label noise, so treat the composite as our best
offline estimate, not a guarantee. Spot-checking unlabeled top-25 entrants
confirmed they are genuine senior IR engineers — evidence the system generalizes
rather than overfits the labels.

## 6. Reproduce

```bash
pip install -r requirements.txt                 # ranking-step deps (numpy only)
pip install -r requirements-precompute.txt      # offline embedding deps (torch)

python scripts/extract_docx.py job_description.docx job_description.txt
python scripts/build_jobspec.py                 # JD -> job_spec.yaml
python scripts/precompute.py --candidates candidates.jsonl   # -> artifacts/*.npy (offline, ~20 min)
python scripts/rank.py --candidates candidates.jsonl --out submission.csv   # CPU, ~35s
python validate_submission.py submission.csv
python scripts/evaluate.py --submission submission.csv       # offline metrics
```

## 7. Repo layout

```
src/redrob/    io, jobspec, embed, baseline (ranker), trust, silver, evalmetrics, reasoning
scripts/       profile_pool, scout, lookup (EDA) · precompute · rank · evaluate · sweep_semweight · sensitivity · smoke_test
eval/          holdout_labels.csv (hand-read gold)
artifacts/     generated embeddings (gitignored; produced by precompute)
```

## 8. Generalization & limitations

**General by construction:** the JobSpec is parsed from JD text; trust/coherence
are universal logic + a reusable domain ontology; scoring self-calibrates to the
pool via percentiles. Point it at a different JD or pool and it re-derives the
target — which is what makes it a product, not a one-off.

**Limitations (honest):** the depth-penalty cues are partly tuned to observed
phrasing (the general principle — penalize self-described shallow depth — holds,
but a productized version would learn them); honeypot detection is deliberately
conservative (~30 of ~80) to protect precision; the offline score is anchored on
a finite hand-labeled set.
