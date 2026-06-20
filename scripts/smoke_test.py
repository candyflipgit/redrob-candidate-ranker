#!/usr/bin/env python3
"""Walking-skeleton smoke test: prove our understanding of the data + the loader.

Checks:
  1. Every record in sample_candidates.json validates against candidate_schema.json.
  2. The streaming loader reads candidates.jsonl and those records validate too.
  3. Field helpers (narrative_text, skill_names) return sensible values.

Run:  python scripts/smoke_test.py
Exit code 0 = all good; non-zero = something to look at.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from redrob import io  # noqa: E402

try:
    from jsonschema import Draft7Validator
except ImportError:
    print("FAIL: jsonschema not installed (pip install -r requirements.txt)")
    sys.exit(1)


def fail(msg: str):
    print(f"FAIL: {msg}")
    sys.exit(1)


def main():
    schema_path = ROOT / "candidate_schema.json"
    sample_path = ROOT / "sample_candidates.json"
    pool_path = ROOT / "candidates.jsonl"

    if not schema_path.exists():
        fail(f"missing {schema_path.name}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft7Validator(schema)

    # 1. sample_candidates.json validates
    sample = io.load_candidates(sample_path)
    if len(sample) != 50:
        fail(f"expected 50 sample candidates, got {len(sample)}")

    bad = 0
    for c in sample:
        errs = sorted(validator.iter_errors(c), key=lambda e: e.path)
        if errs:
            bad += 1
            if bad <= 3:
                cid = c.get("candidate_id", "?")
                print(f"  schema issue in {cid}: {errs[0].message} "
                      f"(at /{'/'.join(map(str, errs[0].path))})")
    if bad:
        fail(f"{bad}/{len(sample)} sample candidates fail the schema")
    print(f"ok  : all {len(sample)} sample candidates validate against schema")

    # 2. streaming loader on the real pool
    if pool_path.exists():
        pool_head = list(io.iter_candidates(pool_path, limit=5))
        if len(pool_head) != 5:
            fail(f"streaming loader returned {len(pool_head)} of 5 expected")
        for c in pool_head:
            if list(validator.iter_errors(c)):
                fail(f"pool record {c.get('candidate_id')} fails the schema")
        print(f"ok  : streamed {len(pool_head)} records from {pool_path.name}, all valid")
    else:
        print(f"skip: {pool_path.name} not present (loader tested on sample only)")

    # 3. field helpers
    c0 = sample[0]
    nt = io.narrative_text(c0)
    sk = io.skill_names(c0)
    if len(nt) < 20:
        fail(f"narrative_text too short for {c0.get('candidate_id')}: {nt!r}")
    print(f"ok  : narrative_text() = {len(nt)} chars, skill_names() = {len(sk)} skills "
          f"for {c0.get('candidate_id')}")

    print("\nPASS: smoke test green")


if __name__ == "__main__":
    main()
