"""
Dataset source registry.

Each adapter module exposes `KEY`, `TITLE`, and `load_records(limit)` returning
records in the canonical schema (see ingestion/schema.py). Select which sources
to ingest via the DATASETS setting (comma-separated keys), e.g. "amazon,myntra".
"""

from __future__ import annotations

from ingestion.sources import amazon, flipkart, myntra

# key -> adapter module
REGISTRY = {
    amazon.KEY: amazon,
    myntra.KEY: myntra,
    flipkart.KEY: flipkart,
}


def get_sources(keys: list[str]) -> list:
    """Resolve a list of dataset keys to adapter modules (unknown keys error)."""
    sources = []
    for k in keys:
        k = k.strip().lower()
        if not k:
            continue
        if k not in REGISTRY:
            raise ValueError(
                f"Unknown dataset '{k}'. Available: {', '.join(REGISTRY)}"
            )
        sources.append(REGISTRY[k])
    return sources
