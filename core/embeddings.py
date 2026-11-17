"""
Embedding model wrapper (the "semantic" half of hybrid search).

Uses a free Sentence-Transformer model to turn text into dense vectors. The
model is loaded lazily and cached as a module-level singleton so we pay the
load cost only once per process.
"""

from __future__ import annotations

import threading

import numpy as np

from config import settings

_model = None  # cached SentenceTransformer instance
_model_lock = threading.Lock()  # guards the lazy load (FastAPI runs sync routes in threads)


def get_model():
    """Load (once) and return the Sentence-Transformer model (thread-safe)."""
    global _model
    # Double-checked locking: fast path avoids the lock once loaded; the lock
    # ensures concurrent first-requests don't each load their own copy (which
    # wastes memory and can crash torch during simultaneous init).
    if _model is None:
        with _model_lock:
            if _model is None:
                # Imported lazily so importing this module stays cheap.
                from sentence_transformers import SentenceTransformer

                print(f"[embeddings] Loading model '{settings.embedding_model}' ...")
                _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_texts(texts: list[str], batch_size: int = 128, show_progress: bool = False) -> np.ndarray:
    """
    Embed a list of texts into an (N, dim) float32 array.

    Vectors are L2-normalized, so a dot product between two vectors equals their
    cosine similarity -- this keeps the retrieval math simple and consistent.
    """
    model = get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # <-- makes dot product == cosine similarity
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string into a 1-D normalized vector."""
    return embed_texts([text])[0]
