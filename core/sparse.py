"""
BM25 sparse encoder -- the "keyword" half of hybrid search.

Dense embeddings capture *meaning* but can miss exact tokens (model numbers,
brand names, "cotton", "32GB"). BM25 is the classic keyword-relevance scoring
used by search engines. We turn text into **sparse vectors** {indices, values}
so the exact same representation works for:

  - the local NumPy backend (we score BM25 ourselves via an inverted index), and
  - Pinecone (which natively supports sparse-dense hybrid vectors).

Math (standard BM25, split doc-side / query-side like pinecone-text):
  doc term weight   = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
  query term weight = idf = ln((N - df + 0.5) / (df + 0.5) + 1)
  score(q, d)       = sum over shared terms of  query_weight * doc_weight
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

# Token ids are hashed into a large fixed space so we never need an explicit
# vocabulary map. Pinecone accepts arbitrary uint32 sparse indices.
_HASH_DIM = 2 ** 20  # ~1M buckets; collisions are negligible for this corpus

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens, dropping 1-char noise."""
    if not text:
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]


def _token_id(token: str) -> int:
    """Deterministic hash of a token into the sparse index space."""
    # Python's built-in hash is salted per-process, so use a stable hash.
    h = 0
    for ch in token:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return h % _HASH_DIM


class BM25SparseEncoder:
    """Fit BM25 statistics on a corpus, then encode docs and queries."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.avgdl: float = 0.0
        self.n_docs: int = 0
        # idf keyed by token id (only tokens seen during fit).
        self.idf: dict[int, float] = {}

    # ---- Fitting ---------------------------------------------------------
    def fit(self, corpus: list[str]) -> "BM25SparseEncoder":
        """Compute document frequencies, idf and average document length."""
        self.n_docs = len(corpus)
        doc_freq: dict[int, int] = {}
        total_len = 0

        for text in corpus:
            tokens = _tokenize(text)
            total_len += len(tokens)
            # Each unique token in the doc contributes 1 to its document freq.
            for tid in set(_token_id(t) for t in tokens):
                doc_freq[tid] = doc_freq.get(tid, 0) + 1

        self.avgdl = (total_len / self.n_docs) if self.n_docs else 0.0
        # Precompute idf for every token seen in the corpus.
        for tid, df in doc_freq.items():
            self.idf[tid] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
        return self

    # ---- Encoding --------------------------------------------------------
    def encode_document(self, text: str) -> dict:
        """Return a sparse vector {indices, values} with BM25 doc weights."""
        tokens = _tokenize(text)
        if not tokens:
            return {"indices": [], "values": []}

        dl = len(tokens)
        # Term frequencies keyed by token id.
        tf: dict[int, int] = {}
        for t in tokens:
            tid = _token_id(t)
            tf[tid] = tf.get(tid, 0) + 1

        indices, values = [], []
        denom_norm = self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
        for tid, freq in tf.items():
            weight = freq * (self.k1 + 1) / (freq + denom_norm)
            indices.append(tid)
            values.append(float(weight))
        return {"indices": indices, "values": values}

    def encode_query(self, text: str) -> dict:
        """Return a sparse vector {indices, values} with idf query weights."""
        tokens = _tokenize(text)
        indices, values = [], []
        seen = set()
        for t in tokens:
            tid = _token_id(t)
            if tid in seen:
                continue
            seen.add(tid)
            idf = self.idf.get(tid)
            if idf:  # skip tokens never seen in the corpus (they match nothing)
                indices.append(tid)
                values.append(float(idf))
        return {"indices": indices, "values": values}

    # ---- Persistence -----------------------------------------------------
    def save(self, path: Path) -> None:
        """Persist fitted stats to JSON so we don't re-fit at query time."""
        payload = {
            "k1": self.k1,
            "b": self.b,
            "avgdl": self.avgdl,
            "n_docs": self.n_docs,
            # JSON keys must be strings; convert token ids back to int on load.
            "idf": {str(k): v for k, v in self.idf.items()},
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25SparseEncoder":
        data = json.loads(path.read_text(encoding="utf-8"))
        enc = cls(k1=data["k1"], b=data["b"])
        enc.avgdl = data["avgdl"]
        enc.n_docs = data["n_docs"]
        enc.idf = {int(k): v for k, v in data["idf"].items()}
        return enc
