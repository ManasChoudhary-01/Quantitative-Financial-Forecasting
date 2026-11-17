"""
Adapter: Flipkart products (PromptCloud, 2016).

~20,000 diverse Indian products (electronics, fashion, home) with INR prices and
hierarchical category trees. Downside: 2016 image URLs are mostly dead. Kept as
an option for category diversity.

Source: https://huggingface.co/datasets/jason1966/PromptCloudHQ_flipkart-products
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ingestion import schema
from ingestion.sources._util import download_file

KEY = "flipkart"
TITLE = "Flipkart (diverse, dead images)"

_URL = (
    "https://huggingface.co/datasets/jason1966/PromptCloudHQ_flipkart-products/resolve/main/"
    "flipkart_com-ecommerce_sample.csv"
)
_FILENAME = "flipkart_com-ecommerce_sample.csv"


def _parse_category_tree(raw: Any) -> tuple[str, str, str]:
    """
    Parse Flipkart's tree into (top, subcategory, clean_path).

    Trees look like: "L0 >> L1 >> L2 >> {Brand} L2 >> {full product name}".
    The deepest real taxonomy level is the one whose child is a "{brand} {level}"
    repeat (e.g. "DEBOCK Headphones".endswith("Headphones")); everything below
    that is brand/product noise we drop.
    """
    if not isinstance(raw, str) or not raw.strip():
        return ("Uncategorized", "Uncategorized", "Uncategorized")

    text = raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed:
            text = str(parsed[0])
    except (json.JSONDecodeError, TypeError):
        text = raw.strip().lstrip("[").rstrip("]").strip().strip('"')

    levels = [lvl.strip() for lvl in text.split(">>") if lvl.strip()]
    if not levels:
        return ("Uncategorized", "Uncategorized", "Uncategorized")

    leaf_idx = 0
    for i in range(len(levels) - 1):
        cur, nxt = levels[i], levels[i + 1]
        if nxt != cur and nxt.endswith(cur):
            leaf_idx = i  # last match wins -> deepest real level

    return (levels[0], levels[leaf_idx], " > ".join(levels[: leaf_idx + 1]))


def load_records(limit: int | None = None) -> list[dict]:
    path = download_file(_URL, _FILENAME)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    print(f"[flipkart] Raw rows: {len(df)}")

    records: list[dict] = []
    for _, row in df.iterrows():
        category, subcategory, category_path = _parse_category_tree(row.get("product_category_tree"))
        rec = schema.assemble(
            id=f"{KEY}:{row.get('uniq_id') or row.get('pid')}",
            name=row.get("product_name"),
            description=row.get("description"),
            brand=row.get("brand"),
            category=category,
            subcategory=subcategory,
            category_path=category_path,
            retail_price=schema.to_float(row.get("retail_price")),
            discounted_price=schema.to_float(row.get("discounted_price")),
            rating=schema.parse_rating(row.get("product_rating")),
            in_stock=True,
            image_url=schema.first_image(row.get("image")),
            product_url=row.get("product_url"),
        )
        if rec:
            records.append(rec)
        if limit and len(records) >= limit:
            break

    print(f"[flipkart] Clean records: {len(records)}")
    return records
