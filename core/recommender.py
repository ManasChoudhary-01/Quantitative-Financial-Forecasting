"""
Recommendation engine.

Two complementary strategies, both reusing the SAME embedding space we already
built for search (no extra infrastructure):

  1. similar_products(id)      -> content-based "more like this": nearest
     neighbours of a product's own embedding.

  2. recommend_for_user(user)  -> personalized: build a "taste vector" by
     averaging the embeddings of products the user viewed / liked / bought
     (weighted by action strength and recency), then find nearest neighbours,
     optionally under a budget.

Because recommendations are pure semantic similarity, we query the store with
alpha=1.0 (dense only) and an empty sparse vector.
"""

from __future__ import annotations

import threading

import numpy as np

from core import db
from core.vector_store import VectorStore, get_vector_store

_store: VectorStore | None = None
_store_lock = threading.Lock()

# How much each interaction type reflects genuine interest.
ACTION_WEIGHTS = {"purchase": 3.0, "like": 2.0, "view": 1.0}
_EMPTY_SPARSE = {"indices": [], "values": []}


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = get_vector_store()
    return _store


def similar_products(product_id: str, top_k: int = 10) -> list[dict]:
    """Return products most similar to the given one (excluding itself)."""
    store = _get_store()
    vec = store.get_dense_vectors([product_id])[0]
    if not np.any(vec):
        return []  # product not in the index

    # Fetch top_k+1 because the item itself will rank first, then drop it.
    hits = store.query(dense=vec, sparse=_EMPTY_SPARSE, top_k=top_k + 1, alpha=1.0)
    hits = [(pid, s) for pid, s in hits if pid != product_id][:top_k]

    products = db.get_products([pid for pid, _ in hits])
    out = []
    for pid, score in hits:
        p = products.get(pid)
        if p:
            p["similarity"] = score
            out.append(p)
    return out


def _build_profile_vector(user_id: str) -> np.ndarray | None:
    """
    Average the embeddings of a user's interactions into one 'taste' vector.

    Weight = action strength * recency factor. Recent interactions matter more,
    so we decay by how far back the event is in the (newest-first) history.
    """
    events = db.get_user_events(user_id, limit=50)
    if not events:
        return None

    store = _get_store()
    ids = [e["product_id"] for e in events]
    vectors = store.get_dense_vectors(ids)  # (n, dim), zeros for missing

    weights = []
    for idx, e in enumerate(events):
        base = ACTION_WEIGHTS.get(e["action"], 1.0)
        recency = 1.0 / (1.0 + 0.1 * idx)  # newest event has factor 1.0
        weights.append(base * recency)
    w = np.array(weights, dtype=np.float32).reshape(-1, 1)

    profile = (vectors * w).sum(axis=0)
    norm = np.linalg.norm(profile)
    if norm == 0:
        return None
    return (profile / norm).astype(np.float32)  # normalize for cosine scoring


def recommend_for_user(
    user_id: str,
    top_k: int = 10,
    price_max: float | None = None,
    exclude_seen: bool = True,
) -> list[dict]:
    """
    Personalized recommendations from the user's behaviour profile.

    Args:
        price_max: optional budget constraint (INR).
        exclude_seen: don't recommend products the user already interacted with.
    """
    profile = _build_profile_vector(user_id)
    if profile is None:
        return []  # cold-start: no history yet

    store = _get_store()
    # Over-fetch so we still have enough after removing already-seen items.
    hits = store.query(
        dense=profile,
        sparse=_EMPTY_SPARSE,
        top_k=top_k * 3,
        alpha=1.0,
        price_max=price_max,
    )

    seen = {e["product_id"] for e in db.get_user_events(user_id, limit=200)} if exclude_seen else set()
    hits = [(pid, s) for pid, s in hits if pid not in seen][:top_k]

    products = db.get_products([pid for pid, _ in hits])
    out = []
    for pid, score in hits:
        p = products.get(pid)
        if p:
            p["match_score"] = score
            out.append(p)
    return out
