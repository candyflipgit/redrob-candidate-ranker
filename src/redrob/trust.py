"""Trust / evidence layer — separates an IR *career* from IR *keywords*.

The remaining traps after lexical+semantic ranking are generalists who carry
retrieval/ranking *skills* (BM25, Qdrant, Vector Search) but whose actual
*career* is computer-vision, forecasting, fraud, or IT-services work. Skills and
the skills-influenced narrative fool both lexical and embedding matching; the
career descriptions do not. These graded signals demote them:

  1. cv/speech/robotics skill-dominance  (wrong-domain, the JD rejects this)
  2. services-career fraction            (the JD rejects services-only careers)
  3. career-IR-focus                     (do the role DESCRIPTIONS describe
                                          retrieval/ranking? the key evidence)

All graded (never a hard solo reject) and general (driven by the JobSpec + a
reusable domain ontology), per the approach memory.
"""
from __future__ import annotations

from . import io as cio
from .jobspec import JobSpec

WRONG_DOMAIN_SKILLS = [
    "computer vision", "object detection", "opencv", "yolo",
    "image classification", "image segmentation", "semantic segmentation",
    "gan", "diffusion", "resnet", "ocr", "face recognition", "pose estimation",
    "optical flow", "vision transformer", "cnn", "asr", "tts",
    "speech recognition", "wav2vec", "speech synthesis", "slam", "robotics",
    "lidar",
]
IR_SKILLS = [
    "retrieval", "ranking", "learning to rank", "recommendation", "recommender",
    "semantic search", "vector search", "embedding", "bm25", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch",
    "dense retrieval", "rag", "sentence-transformers", "re-rank", "rerank",
]
IR_DESC_TERMS = [
    "retrieval", "ranking", "re-rank", "rerank", "recommendation", "recommender",
    "recsys", "semantic search", "vector search", "learning to rank",
    "information retrieval", "search relevance", "candidate matching", "embedding",
]


def skill_domain(c: dict) -> tuple[int, int]:
    names = [(s.get("name") or "").lower() for s in cio.skills(c)]
    ir = sum(any(t in n for t in IR_SKILLS) for n in names)
    cv = sum(any(t in n for t in WRONG_DOMAIN_SKILLS) for n in names)
    return ir, cv


def cv_penalty(ir: int, cv: int) -> float:
    if cv >= 2 and cv > ir:
        return max(0.55, 1.0 - 0.18 * (cv - ir))   # cv-ir: 1->.82 2->.64 3+->.55
    return 1.0


def services_fraction(c: dict, spec: JobSpec) -> float:
    svc = spec.disqualifiers.get("services_only") or []
    if not svc:
        return 0.0
    hist = cio.career_history(c)
    total = sum(int(h.get("duration_months") or 0) for h in hist)
    if total <= 0:
        return 0.0
    months = sum(int(h.get("duration_months") or 0) for h in hist
                 if any(s in (h.get("company") or "").lower() for s in svc))
    return months / total


def services_penalty(frac: float) -> float:
    if frac <= 0.3:
        return 1.0                                  # a single stint is fine
    return max(0.60, 1.0 - 0.57 * (frac - 0.3) / 0.7)   # all-services -> ~0.60


def career_ir_focus(c: dict) -> float:
    """Fraction of career entries whose DESCRIPTION actually describes
    retrieval/ranking/recsys work. The elites ~1.0; keyword generalists ~0."""
    hist = cio.career_history(c)
    if not hist:
        return 0.0
    hits = sum(1 for h in hist
               if any(t in (h.get("description") or "").lower() for t in IR_DESC_TERMS))
    return hits / len(hist)


def focus_factor(focus: float) -> float:
    return 0.85 + 0.30 * focus                      # focus 0->0.85, 1->1.15


def signals(c: dict, spec: JobSpec) -> dict:
    ir, cv = skill_domain(c)
    frac = services_fraction(c, spec)
    focus = career_ir_focus(c)
    penalty = cv_penalty(ir, cv) * services_penalty(frac)
    flags = []
    if cv >= 2 and cv > ir:
        flags.append(f"cv/speech-skills({cv})")
    if frac > 0.5:
        flags.append(f"services({frac:.0%})")
    if focus == 0 and (ir > 0 or cv > 0):
        flags.append("no-IR-career-evidence")
    return {"ir": ir, "cv": cv, "services_frac": frac, "focus": focus,
            "penalty": penalty, "focus_factor": focus_factor(focus),
            "flags": flags}
