"""
Pluggable vector store with two interchangeable backends:

  - LocalVectorStore    : pure NumPy index persisted to disk. No external
                          service, no API key -> perfect for the local demo.
  - PineconeVectorStore : managed Pinecone serverless index -> the production
                          path highlighted on the resume.

Both expose the SAME interface and both do true HYBRID retrieval
(dense semantic + sparse BM25 keyword), so the rest of the app doesn't care
which one is active. Pick the backend with VECTOR_BACKEND in `.env`.
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod

import numpy as np

from config import settings
from core import db


# =============================================================================
# Common interface
# =============================================================================
class VectorStore(ABC):
    """Contract every backend implements."""

    @abstractmethod
    def build(self, ids: list[str], dense: np.ndarray, sparse: list[dict]) -> None:
        """Index the corpus (called once during ingestion)."""

    @abstractmethod
    def query(
        self,
        dense: np.ndarray,
        sparse: dict,
        top_k: int,
        alpha: float,
        price_min: float | None = None,
        price_max: float | None = None,
        category: str | None = None,
        subcategory: str | None = None,
        in_stock_only: bool = False,
    ) -> list[tuple[str, float]]:
        """Return [(product_id, score), ...] best-first, after metadata filters."""

    @abstractmethod
    def get_dense_vectors(self, ids: list[str]) -> np.ndarray:
        """Return the stored dense vectors for the given ids (for recommendations)."""

    @abstractmethod
    def ready(self) -> bool:
        """True if an index has been built and is queryable."""


_shared_store: VectorStore | None = None
_shared_lock = threading.Lock()


def get_vector_store() -> VectorStore:
    """
    Return the backend chosen in settings, as a process-wide singleton.

    Sharing one instance across search + recommender means the (large) index is
    loaded into memory only once.
    """
    global _shared_store
    if _shared_store is None:
        with _shared_lock:
            if _shared_store is None:
                _shared_store = PineconeVectorStore() if settings.use_pinecone else LocalVectorStore()
    return _shared_store


# =============================================================================
# Local NumPy backend
# =============================================================================
class LocalVectorStore(VectorStore):
    """
    Stores dense vectors as one big matrix and sparse vectors as an inverted
    index. Scoring is done in-process with NumPy -- fast enough for tens of
    thousands of products.
    """

    def __init__(self) -> None:
        self._dense: np.ndarray | None = None      # (N, dim) normalized
        self._ids: list[str] | None = None         # row index -> product id
        self._id_to_row: dict[str, int] = {}
        self._sparse: list[dict] | None = None     # per-doc {indices, values}
        self._inverted: dict[int, list[tuple[int, float]]] = {}  # token -> [(row, w)]
        self._meta_loaded = False
        # Aligned metadata arrays for fast filtering.
        self._price: np.ndarray | None = None
        self._category: list[str] | None = None
        self._subcategory: list[str] | None = None
        self._in_stock: np.ndarray | None = None
        self._load_lock = threading.Lock()  # guards the (heavy) lazy index load

    # ---- Paths ----
    @property
    def _dense_path(self):
        return settings.index_dir / "local_dense.npy"

    @property
    def _ids_path(self):
        return settings.index_dir / "local_ids.json"

    @property
    def _sparse_path(self):
        return settings.index_dir / "local_sparse.json"

    # ---- Build ----
    def build(self, ids: list[str], dense: np.ndarray, sparse: list[dict]) -> None:
        np.save(self._dense_path, dense.astype(np.float32))
        self._ids_path.write_text(json.dumps(ids), encoding="utf-8")
        self._sparse_path.write_text(json.dumps(sparse), encoding="utf-8")
        print(f"[local-store] Saved index for {len(ids)} products to {settings.index_dir}")

    # ---- Load (lazy) ----
    def _ensure_loaded(self) -> None:
        # Double-checked locking so concurrent first-queries don't each reload
        # the index / rebuild the inverted index (expensive and racy).
        if self._dense is not None:
            return
        with self._load_lock:
            if self._dense is not None:
                return
            if not self.ready():
                raise RuntimeError(
                    "Local index not found. Run the ingestion pipeline first "
                    "(python -m ingestion.ingest)."
                )
            ids = json.loads(self._ids_path.read_text(encoding="utf-8"))
            sparse = json.loads(self._sparse_path.read_text(encoding="utf-8"))
            self._ids = ids
            self._sparse = sparse
            self._id_to_row = {pid: i for i, pid in enumerate(ids)}

            # Build the inverted index: token id -> list of (row, weight).
            inverted: dict[int, list[tuple[int, float]]] = {}
            for row, vec in enumerate(sparse):
                for tid, w in zip(vec["indices"], vec["values"]):
                    inverted.setdefault(tid, []).append((row, w))
            self._inverted = inverted

            self._load_meta()
            # Assign _dense LAST: it's the flag other threads check on the fast
            # path, so it must only become non-None once everything is ready.
            self._dense = np.load(self._dense_path)

    def _load_meta(self) -> None:
        """Pull price/category/stock from SQLite, aligned to the vector rows."""
        products = db.get_products(self._ids)  # {id: product}
        n = len(self._ids)
        price = np.zeros(n, dtype=np.float32)
        category: list[str] = [""] * n
        subcategory: list[str] = [""] * n
        in_stock = np.ones(n, dtype=bool)
        for i, pid in enumerate(self._ids):
            p = products.get(pid)
            if p:
                price[i] = p.get("price") or 0.0
                category[i] = p.get("category") or ""
                subcategory[i] = p.get("subcategory") or ""
                in_stock[i] = bool(p.get("in_stock"))
        self._price, self._category, self._subcategory = price, category, subcategory
        self._in_stock = in_stock
        self._meta_loaded = True

    # ---- Query ----
    def query(self, dense, sparse, top_k, alpha, price_min=None, price_max=None,
              category=None, subcategory=None, in_stock_only=False):
        self._ensure_loaded()
        n = self._dense.shape[0]

        # 1) Dense scores: since vectors are normalized, matrix-vector dot == cosine.
        dense_scores = self._dense @ dense.astype(np.float32)

        # 2) Sparse scores: accumulate BM25 contributions via the inverted index.
        sparse_scores = np.zeros(n, dtype=np.float32)
        for tid, qw in zip(sparse["indices"], sparse["values"]):
            for row, dw in self._inverted.get(tid, ()):  # docs containing this term
                sparse_scores[row] += qw * dw

        # 3) Fuse. Min-max normalize each signal to [0,1] so alpha is meaningful.
        fused = alpha * _minmax(dense_scores) + (1 - alpha) * _minmax(sparse_scores)

        # 4) Metadata filtering -> mask out disqualified rows.
        mask = np.ones(n, dtype=bool)
        if price_min is not None:
            mask &= self._price >= price_min
        if price_max is not None:
            mask &= self._price <= price_max
        if category:
            mask &= np.array([c == category for c in self._category])
        if subcategory:
            mask &= np.array([s == subcategory for s in self._subcategory])
        if in_stock_only:
            mask &= self._in_stock
        fused = np.where(mask, fused, -np.inf)

        # 5) Top-k selection (argpartition is O(n), then sort just the winners).
        k = min(top_k, int(mask.sum()))
        if k <= 0:
            return []
        top_idx = np.argpartition(-fused, k - 1)[:k]
        top_idx = top_idx[np.argsort(-fused[top_idx])]
        return [(self._ids[i], float(fused[i])) for i in top_idx if fused[i] > -np.inf]

    def get_dense_vectors(self, ids: list[str]) -> np.ndarray:
        self._ensure_loaded()
        dim = self._dense.shape[1]
        out = np.zeros((len(ids), dim), dtype=np.float32)
        for i, pid in enumerate(ids):
            row = self._id_to_row.get(pid)
            if row is not None:
                out[i] = self._dense[row]
        return out

    def ready(self) -> bool:
        return self._dense_path.exists() and self._ids_path.exists() and self._sparse_path.exists()


def _minmax(x: np.ndarray) -> np.ndarray:
    """Scale an array to [0,1]; return zeros if the range is degenerate."""
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# =============================================================================
# Pinecone backend
# =============================================================================
class PineconeVectorStore(VectorStore):
    """
    Managed serverless index. Uses a dot-product metric so dense + sparse
    hybrid vectors can be combined natively by Pinecone.
    """

    def __init__(self) -> None:
        self._pc = None
        self._index = None
        self._lock = threading.Lock()

    def _client(self):
        """Lazily create the Pinecone client + index handle (thread-safe)."""
        if self._index is not None:
            return self._index
        with self._lock:
            if self._index is not None:
                return self._index
            from pinecone import Pinecone, ServerlessSpec

            if not settings.pinecone_api_key:
                raise RuntimeError("VECTOR_BACKEND=pinecone but PINECONE_API_KEY is not set.")

            self._pc = Pinecone(api_key=settings.pinecone_api_key)
            existing = [i.name for i in self._pc.list_indexes()]
            if settings.pinecone_index_name not in existing:
                print(f"[pinecone] Creating index '{settings.pinecone_index_name}' ...")
                self._pc.create_index(
                    name=settings.pinecone_index_name,
                    dimension=settings.embedding_dim,
                    metric="dotproduct",  # required for sparse-dense hybrid
                    spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
                )
            self._index = self._pc.Index(settings.pinecone_index_name)
            return self._index

    def build(self, ids, dense, sparse):
        index = self._client()
        products = db.get_products(ids)
        batch, BATCH = [], 100
        for i, pid in enumerate(ids):
            p = products.get(pid, {})
            batch.append({
                "id": pid,
                "values": dense[i].tolist(),
                "sparse_values": {"indices": sparse[i]["indices"], "values": sparse[i]["values"]},
                # Only the fields we filter on need to live in Pinecone metadata.
                "metadata": {
                    "price": float(p.get("price") or 0.0),
                    "category": p.get("category") or "",
                    "subcategory": p.get("subcategory") or "",
                    "in_stock": bool(p.get("in_stock", True)),
                },
            })
            if len(batch) >= BATCH:
                index.upsert(vectors=batch)
                batch = []
        if batch:
            index.upsert(vectors=batch)
        print(f"[pinecone] Upserted {len(ids)} vectors.")

    def query(self, dense, sparse, top_k, alpha, price_min=None, price_max=None,
              category=None, subcategory=None, in_stock_only=False):
        index = self._client()

        # Convex weighting: scale dense by alpha and sparse by (1-alpha) so the
        # dot-product fusion respects the semantic/keyword balance.
        d = (dense * alpha).astype(np.float32).tolist()
        s = {
            "indices": sparse["indices"],
            "values": [v * (1 - alpha) for v in sparse["values"]],
        }

        # Build Pinecone metadata filter.
        flt: dict = {}
        price_cond: dict = {}
        if price_min is not None:
            price_cond["$gte"] = price_min
        if price_max is not None:
            price_cond["$lte"] = price_max
        if price_cond:
            flt["price"] = price_cond
        if category:
            flt["category"] = {"$eq": category}
        if subcategory:
            flt["subcategory"] = {"$eq": subcategory}
        if in_stock_only:
            flt["in_stock"] = {"$eq": True}

        res = index.query(
            vector=d,
            sparse_vector=s if s["indices"] else None,
            top_k=top_k,
            include_metadata=False,
            filter=flt or None,
        )
        return [(m["id"], float(m["score"])) for m in res.get("matches", [])]

    def get_dense_vectors(self, ids):
        index = self._client()
        res = index.fetch(ids=ids)
        vectors = res.vectors if hasattr(res, "vectors") else res.get("vectors", {})
        dim = settings.embedding_dim
        out = np.zeros((len(ids), dim), dtype=np.float32)
        for i, pid in enumerate(ids):
            v = vectors.get(pid)
            if v is not None:
                values = v["values"] if isinstance(v, dict) else v.values
                out[i] = np.array(values, dtype=np.float32)
        return out

    def ready(self) -> bool:
        try:
            index = self._client()
            stats = index.describe_index_stats()
            total = stats.get("total_vector_count", 0) if isinstance(stats, dict) else stats.total_vector_count
            return total > 0
        except Exception:
            return False
