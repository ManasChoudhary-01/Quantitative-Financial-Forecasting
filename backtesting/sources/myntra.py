"""
Adapter: Myntra fashion products.

A small, pristine fashion catalog with INR prices, brands, gender, and **live**
Myntra CDN images. Great for visual demos.

Source: https://huggingface.co/datasets/alexkstern/myntra_fashion_dataset
"""

from __future__ import annotations

import json

from ingestion import schema
from ingestion.sources._util import download_file

KEY = "myntra"
TITLE = "Myntra (fashion)"

_URL = (
    "https://huggingface.co/datasets/alexkstern/myntra_fashion_dataset/resolve/main/"
    "myntra_fashion_products_free_dataset.json"
)
_FILENAME = "myntra_fashion.json"


def load_records(limit: int | None = None) -> list[dict]:
    path = download_file(_URL, _FILENAME)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[myntra] Raw rows: {len(data)}")

    records: list[dict] = []
    for row in data:
        gender = (row.get("gender") or "").strip() or "Unisex"
        # Fashion is one top category; gender is a clean, reliable facet.
        rec = schema.assemble(
            id=f"{KEY}:{row.get('sku') or row.get('mpn')}",
            name=row.get("name"),
            description=row.get("description"),
            brand=row.get("brand"),
            category="Fashion",
            subcategory=gender,
            category_path=f"Fashion > {gender}",
            # Only one price is given (INR); treat it as the selling price.
            retail_price=None,
            discounted_price=schema.to_float(row.get("price")),
            rating=None,
            in_stock=bool(row.get("in_stock", True)),
            image_url=schema.first_image(row.get("images")),
            product_url="",
        )
        if rec:
            records.append(rec)
        if limit and len(records) >= limit:
            break

    print(f"[myntra] Clean records: {len(records)}")
    return records
