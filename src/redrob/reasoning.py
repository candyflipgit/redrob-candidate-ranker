"""Generate the submission's `reasoning` column from each candidate's REAL
features — no LLM, no hallucination.

Targets the Stage-4 manual-review rubric:
  - specific facts   : real title, years, employers, matched terms, signal values
  - JD connection    : ties to the JD's band / retrieval-ranking core / location
  - honest concerns  : surfaces the trust flags + band/availability gaps
  - no hallucination : only cites self-consistent fields (employer named only when
                       the candidate is coherent, so title/description agree)
  - variation        : driven by each candidate's own employers/terms/concerns
  - rank consistency : tone scales with rank (strong at top, qualified lower down)
"""
from __future__ import annotations

from . import io as cio
from . import trust
from .jobspec import JobSpec


def _ir_evidence(c: dict):
    """Strongest IR career entry: (company, matched IR terms) or None."""
    best = None
    for h in cio.career_history(c):
        d = (h.get("description") or "").lower()
        terms = [t for t in trust.IR_DESC_TERMS if t in d]
        if terms and (best is None or len(terms) > len(best[1])):
            best = (h.get("company"), terms)
    return best


def _concerns(c: dict, s, spec: JobSpec) -> list[str]:
    out = []
    yoe = c.get("profile", {}).get("years_of_experience")
    if s.band_fit < 0.85 and isinstance(yoe, (int, float)):
        rel = "below" if yoe < spec.min_years else "above"
        out.append(f"{yoe:.0f}y is {rel} the {spec.min_years:.0f}-{spec.max_years:.0f}y band")
    if s.location < 0.5:
        out.append("outside preferred locations, relocation unflagged")
    for f in s.disq:
        if f.startswith("services"):
            out.append("an IT-services stint")
        elif f.startswith("cv/"):
            out.append("some computer-vision/speech focus")
        elif f == "self-described-lightweight":
            out.append("describes some work as lighter-weight")
        elif f == "no-IR-career-evidence":
            out.append("thin retrieval/ranking evidence in the career history")
    if isinstance(s.response, (int, float)) and s.response < 0.30:
        out.append(f"low recruiter response ({s.response:.2f})")
    if s.recency_days > 150:
        out.append(f"last active {s.recency_days}d ago")
    return out


def generate(c: dict, s, spec: JobSpec, rank: int) -> str:
    yrs = f"{s.yoe:.0f}y" if isinstance(s.yoe, (int, float)) else "experience n/a"
    parts = [f"{s.title}, {yrs}"]

    ev = _ir_evidence(c)
    if s.coherence >= 0.6 and ev and ev[0]:
        terms = ", ".join(list(dict.fromkeys(ev[1]))[:3])
        parts.append(f"career centers on {terms} (e.g. at {ev[0]})")
    elif s.top_terms:
        parts.append("profile cites " + ", ".join(s.top_terms[:3]))
    elif s.semantic is not None and s.semantic >= 0.85:
        parts.append("strong semantic match to the retrieval/ranking role")

    if isinstance(s.response, (int, float)):
        eng = "strong" if s.response >= 0.6 else ("moderate" if s.response >= 0.35 else "weak")
        parts.append(f"{eng} engagement (response {s.response:.2f}, active {s.recency_days}d ago)")

    text = "; ".join(parts)
    concerns = _concerns(c, s, spec)
    if concerns:
        label = "minor concern" if rank <= 15 else "concerns"
        text += f". {label.capitalize()}: " + ", ".join(concerns[:3])
    elif rank > 60:
        text += ". Credible but lower-confidence fit"
    return text[:300]
