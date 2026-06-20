"""Turn a free-text job description into a structured, faceted JobSpec.

Design principle (generalization): nothing here is hardcoded to the literal
"Senior AI Engineer" posting. We keep a *general domain ontology* (reusable
knowledge about IR/ML concepts, infra, and common disqualifiers — not answers
for this dataset) and intersect it with the JD text, using the JD's own section
structure ("absolutely need" / "like to have" / "do NOT want") to bucket terms.
Feed a different JD and a different spec falls out. The ranker consumes only the
resulting JobSpec, so swapping jobs never touches ranking code.

In production this is the "parse once per job" step; the spec can be reviewed or
hand-edited as a YAML artifact before ranking.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

# --- General domain ontology (reusable; NOT specific to this JD/dataset) -----

ONTOLOGY: dict[str, list[str]] = {
    # the heart of a retrieval/ranking role
    "ir_core": [
        "retrieval", "ranking", "re-ranking", "reranking", "recommendation",
        "recommender", "recsys", "semantic search", "vector search",
        "information retrieval", "learning to rank", "dense retrieval",
        "hybrid search", "hybrid retrieval", "neural ranking", "search relevance",
        "candidate matching", "matching system", "query understanding",
        "nearest neighbor", "embeddings", "embedding",
    ],
    "ml_general": [
        "machine learning", "deep learning", "nlp", "natural language", "llm",
        "large language model", "fine-tuning", "fine tune", "lora", "qlora",
        "peft", "rag", "transformer", "bert", "sentence-transformers",
        "representation learning",
    ],
    "vector_infra": [
        "pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch",
        "opensearch", "vespa", "vector database", "vector db",
    ],
    "ml_tools": [
        "pytorch", "tensorflow", "hugging face", "huggingface", "bge", "e5",
        "xgboost", "scikit-learn", "onnx",
    ],
    "eval": [
        "ndcg", "mrr", "mean average precision", "a/b test", "ab test",
        "offline evaluation", "online evaluation", "evaluation framework",
        "eval framework", "offline-to-online",
    ],
    "engineering": [
        "python", "production", "scalable", "latency", "distributed systems",
        "inference", "pipeline", "hybrid", "api",
    ],
    "role_positive": [
        "machine learning engineer", "ml engineer", "ai engineer", "nlp engineer",
        "applied scientist", "research engineer", "search engineer",
        "recommendation systems engineer", "data scientist", "software engineer",
        "engineer", "developer", "scientist",
    ],
}

# Disqualifier ontology: name -> (trigger cues, human description).
DISQUALIFIERS: dict[str, dict] = {
    "services_only": {
        "cues": ["tcs", "infosys", "wipro", "accenture", "cognizant",
                 "capgemini", "tech mahindra", "hcl", "mindtree",
                 "consulting firm", "consultancy", "services company"],
        "desc": "career entirely at IT-services / consulting firms",
    },
    "research_only": {
        "cues": ["research-only", "research only", "pure research", "academic",
                 "professor", "postdoc", "research lab"],
        "desc": "pure research background, no production deployment",
    },
    "framework_only": {
        "cues": ["langchain", "framework enthusiast", "tutorials"],
        "desc": "AI experience is only recent LangChain/framework glue",
    },
    "wrong_domain": {
        "cues": ["computer vision", "speech", "robotics", "image processing",
                 "signal processing"],
        "desc": "CV / speech / robotics primary, without NLP/IR exposure",
    },
    "title_chaser": {
        "cues": [],  # detected from career trajectory, not JD terms
        "desc": "job-hops every ~1.5y to chase Senior/Staff/Principal titles",
    },
}

CITIES = [
    "pune", "noida", "hyderabad", "mumbai", "delhi ncr", "delhi", "gurgaon",
    "gurugram", "bangalore", "bengaluru", "chennai", "kolkata", "ncr",
]
WORK_MODES = ["remote", "hybrid", "onsite", "flexible"]

# Section cue phrases (generic across well-structured JDs).
_MUST_CUES = ["absolutely need", "must have", "you need", "requirements",
              "required skills"]
_NICE_CUES = ["like you to have", "nice to have", "bonus", "preferred",
              "won't reject", "would like you to have"]
_AVOID_CUES = ["do not want", "don't want", "explicitly do not", "disqualif",
               "not a fit", "red flag", "we will not move forward"]


@dataclass
class JobSpec:
    title: str = ""
    role_terms: list[str] = field(default_factory=list)
    must_have_terms: list[str] = field(default_factory=list)
    nice_to_have_terms: list[str] = field(default_factory=list)
    disqualifiers: dict[str, list[str]] = field(default_factory=dict)
    min_years: float = 0.0
    max_years: float = 50.0
    ideal_years: float = 0.0
    locations: list[str] = field(default_factory=list)
    relocate_ok: bool = False
    work_modes: list[str] = field(default_factory=list)

    # --- serialization ----------------------------------------------------
    def to_yaml(self, path) -> None:
        Path(path).write_text(
            yaml.safe_dump(asdict(self), sort_keys=False, allow_unicode=True),
            encoding="utf-8")

    @classmethod
    def from_yaml(cls, path) -> "JobSpec":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def positive_query(self) -> list[str]:
        """Flat list of positive terms for lexical retrieval / term-signal."""
        seen, out = set(), []
        for t in self.role_terms + self.must_have_terms + self.nice_to_have_terms:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out


def _terms_present(terms: list[str], text: str) -> list[str]:
    return [t for t in terms if t in text]


def _slice_sections(text_l: str) -> tuple[str, str, str]:
    """Best-effort split into (must, nice, avoid) regions by cue phrases.
    Falls back to (whole, '', '') when the JD isn't sectioned that way."""
    def first_pos(cues):
        ps = [text_l.find(c) for c in cues if text_l.find(c) >= 0]
        return min(ps) if ps else -1

    must_p, nice_p, avoid_p = (first_pos(_MUST_CUES), first_pos(_NICE_CUES),
                               first_pos(_AVOID_CUES))
    marks = sorted([p for p in (must_p, nice_p, avoid_p) if p >= 0])
    if not marks:
        return text_l, "", ""

    def region(start):
        if start < 0:
            return ""
        nxt = [m for m in marks if m > start]
        return text_l[start:(nxt[0] if nxt else len(text_l))]

    return region(must_p) or text_l, region(nice_p), region(avoid_p)


def from_text(jd_text: str, title: str = "") -> JobSpec:
    text_l = jd_text.lower()
    must_sec, nice_sec, avoid_sec = _slice_sections(text_l)

    concept_buckets = ["ir_core", "ml_general", "vector_infra", "ml_tools",
                       "eval", "engineering"]
    all_concepts = [t for b in concept_buckets for t in ONTOLOGY[b]]

    must = _terms_present(all_concepts, must_sec)
    nice = [t for t in _terms_present(all_concepts, nice_sec) if t not in must]
    # safety net: any core concept mentioned anywhere should count as must-have
    for t in _terms_present(all_concepts, text_l):
        if t not in must and t not in nice:
            must.append(t)

    role_terms = _terms_present(ONTOLOGY["role_positive"], text_l)

    disq = {}
    for name, spec in DISQUALIFIERS.items():
        hit = _terms_present(spec["cues"], text_l)
        # title_chaser has no JD cues but the JD clearly calls it out
        if hit or (name == "title_chaser" and "title" in text_l
                   and "chas" in text_l):
            disq[name] = hit

    yrs = re.findall(r"(\d+)\s*[-–to]+\s*(\d+)\s*\+?\s*year", text_l)
    min_y, max_y, ideal_y = 0.0, 50.0, 0.0
    if yrs:
        lo, hi = int(yrs[0][0]), int(yrs[0][1])
        if 0 < lo <= hi <= 25:
            min_y, max_y, ideal_y = float(lo), float(hi), (lo + hi) / 2.0

    locations = []
    for c in CITIES:
        if c in text_l and not any(c in loc for loc in locations):
            locations.append(c)
    work_modes = [m for m in WORK_MODES if m in text_l]
    relocate_ok = "relocat" in text_l

    return JobSpec(
        title=title,
        role_terms=role_terms,
        must_have_terms=must,
        nice_to_have_terms=nice,
        disqualifiers=disq,
        min_years=min_y, max_years=max_y, ideal_years=ideal_y,
        locations=locations, relocate_ok=relocate_ok, work_modes=work_modes,
    )
