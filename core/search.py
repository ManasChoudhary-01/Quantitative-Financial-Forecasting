"""
Hybrid search orchestration -- ties the pieces together for one query:

    query text
      -> dense embedding      (semantic)      [embeddings.py]
      -> sparse BM25 vector    (keyword)       [sparse.py]
      -> vector store query    (fused + filtered) [vector_store.py]
      -> join product metadata (SQLite)        [db.py]
      -> cross-encoder rerank  (precision)      [reranker.py]
      -> ranked products

Singletons (store, encoder) are cached so repeated queries are fast.
"""

from __future__ import annotations

import threading

from config import settings
from core import db, intent, reranker
from core.embeddings import embed_query
from core.sparse import BM25SparseEncoder
from core.vector_store import VectorStore, get_vector_store

_store: VectorStore | None = None
_encoder: BM25SparseEncoder | None = None
_lock = threading.Lock()  # guards lazy init against concurrent requests

BM25_PATH = settings.index_dir / "bm25.json"


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = get_vector_store()
    return _store


def _get_encoder() -> BM25SparseEncoder:
    global _encoder
    if _encoder is None:
        with _lock:
            if _encoder is None:
                if not BM25_PATH.exists():
                    raise RuntimeError(
                        "BM25 model not found. Run the ingestion pipeline first "
                        "(python -m ingestion.ingest)."
                    )
                _encoder = BM25SparseEncoder.load(BM25_PATH)
    return _encoder


def search(
    query: str,
    top_k: int | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    in_stock_only: bool = False,
    use_rerank: bool | None = None,
) -> list[dict]:
    """
    Run a hybrid search and return ranked product dicts.

    Each returned product includes `hybrid_score` and (if reranked) `rerank_score`.
    """
    query = (query or "").strip()
    if not query:
        return []

    top_k = top_k or settings.search_top_k
    store = _get_store()
    encoder = _get_encoder()

    # 1) Encode the query into both representations.
    dense = embed_query(query)
    sparse = encoder.encode_query(query)

    # 2) Retrieve a candidate pool (larger than top_k so the reranker has room).
    hits = store.query(
        dense=dense,
        sparse=sparse,
        top_k=settings.search_candidates,
        alpha=settings.hybrid_alpha,
        price_min=price_min,
        price_max=price_max,
        category=category,
        subcategory=subcategory,
        in_stock_only=in_stock_only,
    )
    if not hits:
        return []

    # 3) Join full product metadata from SQLite (vector store only holds vectors).
    id_to_score = {pid: score for pid, score in hits}
    products_by_id = db.get_products(list(id_to_score.keys()))
    candidates: list[dict] = []
    for pid, score in hits:  # preserve retrieval order
        p = products_by_id.get(pid)
        if p:
            p["hybrid_score"] = score
            candidates.append(p)

    # 4) Rerank the candidate pool for precision.
    use_rerank = settings.use_reranker if use_rerank is None else use_rerank
    if use_rerank:
        # Rerank a bit wider than top_k so intent rescoring has room to promote
        # correct-type items that sit just outside the final cut.
        candidates = reranker.rerank(query, candidates, top_k=max(top_k, 20))

    # 5) Type-intent rescoring: if the query names a confusable product type
    #    (e.g. "headphones"), demote the wrong type (earphones) below it.
    candidates = intent.apply(query, candidates)

    return candidates[:top_k]
