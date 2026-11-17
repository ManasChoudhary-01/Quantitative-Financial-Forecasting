"""
The full offline ingestion pipeline (run this ONCE before using the app):

    for each selected dataset: download -> parse to canonical records
    combine + de-duplicate -> write SQLite -> embed (dense) -> fit BM25 (sparse)
    -> build the vector index (local or Pinecone)

Which datasets are ingested is controlled by DATASETS in `.env`
(e.g. "amazon,myntra"). Run with:

    python -m ingestion.ingest
"""

from __future__ import annotations

from config import settings
from core import db
from core.embeddings import embed_texts
from core.search import BM25_PATH
from core.sparse import BM25SparseEncoder
from core.vector_store import get_vector_store
from ingestion.sources import get_sources


def _load_all_records() -> list[dict]:
    """Load + combine records from every selected dataset, de-duplicating ids."""
    sources = get_sources(settings.dataset_keys)
    seen: set[str] = set()
    combined: list[dict] = []
    for src in sources:
        print(f"--- loading dataset: {src.TITLE} ---")
        # Per-source cap keeps a single huge dataset from crowding out the others.
        limit = settings.max_products or None
        for rec in src.load_records(limit=limit):
            if rec["id"] in seen:
                continue  # ids are namespaced per source, so this is rare
            seen.add(rec["id"])
            combined.append(rec)
    return combined


def run() -> None:
    print("=" * 70)
    print("INGESTION PIPELINE")
    print(f"  datasets       : {', '.join(settings.dataset_keys)}")
    print(f"  vector backend : {settings.vector_backend}")
    print(f"  embedding model: {settings.embedding_model}")
    print(f"  max/ dataset   : {settings.max_products or 'ALL'}")
    print("=" * 70)

    # 1) + 2) Download & parse every selected dataset into canonical records.
    records = _load_all_records()
    if not records:
        raise SystemExit("No records after loading -- aborting.")
    print(f"[ingest] Combined records: {len(records)}")

    # 3) Persist product metadata to SQLite (source of truth for display/filtering).
    db.init_db()
    written = db.upsert_products(records)
    print(f"[ingest] Wrote {written} products to SQLite ({settings.db_path}).")

    # 4) Dense embeddings (the semantic half).
    texts = [r["embed_text"] for r in records]
    ids = [r["id"] for r in records]
    print(f"[ingest] Embedding {len(texts)} products (this can take a few minutes)...")
    dense = embed_texts(texts, show_progress=True)

    # 5) Fit BM25 and encode every document (the keyword half).
    print("[ingest] Fitting BM25 and encoding sparse vectors...")
    encoder = BM25SparseEncoder().fit(texts)
    encoder.save(BM25_PATH)
    sparse = [encoder.encode_document(t) for t in texts]

    # 6) Build the vector index (local NumPy files or Pinecone upserts).
    store = get_vector_store()
    store.build(ids, dense, sparse)

    print("=" * 70)
    print(f"DONE. Indexed {len(ids)} products from {len(settings.dataset_keys)} dataset(s).")
    print("Next: start the API  ->  uvicorn api.main:app --reload")
    print("      start the UI   ->  streamlit run ui/streamlit_app.py")
    print("=" * 70)


if __name__ == "__main__":
    run()
