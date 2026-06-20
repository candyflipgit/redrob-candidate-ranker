"""Loading and basic field access for candidate records.

Handles the two shapes the bundle ships in:
  - candidates.jsonl[.gz]  : one JSON object per line (the 100K pool)
  - sample_candidates.json : a JSON array (the 50-record sample)

Everything downstream reads candidates through these helpers, so field-name
assumptions live in exactly one place.
"""
from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator, Optional


def _open_text(path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates(path, limit: Optional[int] = None) -> Iterator[dict]:
    """Stream candidate dicts from a .jsonl or .jsonl.gz file (low memory)."""
    with _open_text(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def load_json_array(path) -> list[dict]:
    """Load a JSON array file (e.g. sample_candidates.json) fully into memory."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_candidates(path, limit: Optional[int] = None) -> list[dict]:
    """Load candidates from either shape, auto-detected by extension."""
    p = Path(path)
    if p.suffixes and p.suffixes[-1] == ".json" and ".jsonl" not in p.suffixes:
        data = load_json_array(path)
        return data[:limit] if limit is not None else data
    return list(iter_candidates(path, limit=limit))


# --- field access helpers (one source of truth for the schema) --------------

def parse_date(value) -> Optional[date]:
    """Parse an ISO date string ('YYYY-MM-DD' or longer); None if unparseable."""
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


def career_history(c: dict) -> list[dict]:
    return c.get("career_history", []) or []


def skills(c: dict) -> list[dict]:
    return c.get("skills", []) or []


def signals(c: dict) -> dict:
    return c.get("redrob_signals", {}) or {}


def narrative_text(c: dict) -> str:
    """The candidate's free-text story: headline + summary + every role's title
    and description. This is the honest signal (much harder to fake than a
    skills list); alignment and coherence are computed from it."""
    p = c.get("profile", {}) or {}
    parts = [p.get("headline", ""), p.get("summary", "")]
    for h in career_history(c):
        parts.append(h.get("title", "") or "")
        parts.append(h.get("description", "") or "")
    return " ".join(x for x in parts if x).strip()


def skill_names(c: dict) -> list[str]:
    return [s.get("name", "") for s in skills(c) if s.get("name")]
