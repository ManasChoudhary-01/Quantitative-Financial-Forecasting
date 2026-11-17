"""
Cross-encoder reranker.

Hybrid retrieval gives us a good candidate set cheaply. A cross-encoder then
re-scores each (query, product) PAIR jointly -- far more accurate than the
initial vector similarity, because it reads the query and the product together.
We rerank ~50 candidates down to the final top-k.

Loaded lazily and cached; controlled by USE_RERANKER in `.env`.
"""

from __future__ import annotations

import threading

from config import settings

_model = None
_model_lock = threading.Lock()  # guards the lazy load against concurrent requests


def get_reranker():
    global _model
    # Double-checked locking (see embeddings.get_model for the rationale).
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import CrossEncoder

                print(f"[reranker] Loading cross-encoder '{settings.reranker_model}' ...")
                _model = CrossEncoder(settings.reranker_model)
    return _model


def rerank(query: str, candidates: list[dict], top_k: int, text_key: str = "embed_text") -> list[dict]:
    """
    Re-order `candidates` by cross-encoder relevance to `query`.

    Args:
        candidates: product dicts (must contain `text_key`).
        top_k: how many to keep.
        text_key: which field to score against the query.

    Returns:
        The top_k candidates, each with an added `rerank_score`, best-first.
    """
    if not candidates:
        return []
    if not settings.use_reranker:
        return candidates[:top_k]

    model = get_reranker()
    # Build (query, product_text) pairs; fall back to name if text is missing.
    pairs = [[query, (c.get(text_key) or c.get("name") or "")] for c in candidates]
    scores = model.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
