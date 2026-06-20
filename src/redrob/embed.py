"""Embedding helpers (offline pre-compute only).

We use a small CPU-friendly sentence model (bge-small-en-v1.5, 384-dim). The JD
is the *query* and each candidate narrative is a *passage*; bge wants a retrieval
instruction prepended to the query only, so we keep that asymmetry explicit (a
silent prefix mistake is a classic way to quietly wreck cosine quality).

This module is imported by scripts/precompute.py (which needs torch). The ranking
step never imports it — it just loads the saved .npy vectors with numpy.
"""
from __future__ import annotations

import os

import numpy as np

# all-MiniLM-L6-v2: 384-dim, 6-layer, fast on CPU, symmetric similarity (no
# query/passage prefix needed). Chosen over bge-small after timing: ~5-10x
# faster on CPU for a small, measured quality cost on our discrimination check.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# bge-family models want this retrieval instruction on the QUERY side only.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
MAX_CHARS = 800       # narratives are truncated before tokenizing (speed)
MAX_SEQ_LEN = 128     # summary + first description carry most of the signal

_MODELS: dict = {}


def get_model(name: str = DEFAULT_MODEL):
    if name not in _MODELS:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(os.cpu_count() or 4)
        m = SentenceTransformer(name)
        m.max_seq_length = MAX_SEQ_LEN
        _MODELS[name] = m
    return _MODELS[name]


def embed(texts, is_query: bool = False, model_name: str = DEFAULT_MODEL,
          batch_size: int = 64, show: bool = False) -> np.ndarray:
    model = get_model(model_name)
    texts = [(t or "")[:MAX_CHARS] for t in texts]
    if is_query and "bge" in model_name.lower():
        texts = [QUERY_INSTRUCTION + t for t in texts]
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=show, convert_to_numpy=True)
    return emb.astype("float32")


def jd_query_text(spec) -> str:
    """A focused 'ideal candidate' query synthesized from the JobSpec — cleaner
    than embedding the whole rambling JD (culture/comp prose dilutes the signal).
    Emphasizes the precise retrieval/ranking stack; we deliberately omit generic
    "AI/ML" framing that pulls forecasting/CV generalists closer in embedding
    space (measured to hurt top precision on the eval harness)."""
    cores = ", ".join(spec.must_have_terms[:14])
    return ("Built and shipped production systems for "
            f"{cores} at scale at product companies. Hands-on with "
            "embeddings-based retrieval, hybrid search, learning-to-rank, "
            "recommendation and re-ranking, with rigorous ranking evaluation.")
