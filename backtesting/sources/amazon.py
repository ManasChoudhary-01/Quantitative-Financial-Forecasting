"""
Adapter: Amazon India products (PromptCloud marketing sample, 2019).

~30,000 Indian products (beauty, grocery, personal care) with INR MRP + selling
price, discount, stock, clean flat categories, and **live** Amazon CDN images.

Source: https://huggingface.co/datasets/pgurazada1/amazon_india_products
"""

from __future__ import annotations

import pandas as pd

from ingestion import schema
from ingestion.sources._util import download_file

KEY = "amazon"
TITLE = "Amazon India (beauty & grocery)"

_URL = (
    "https://huggingface.co/datasets/pgurazada1/amazon_india_products/resolve/main/"
    "marketing_sample_for_amazon_in-ecommerce__20191001_20191031__30k_data.csv"
)
_FILENAME = "amazon_india_30k.csv"


def _stock(raw: str) -> bool:
    """'In Stock.' / 'Only 2 left' -> True; 'Currently unavailable' -> False."""
    return "unavailable" not in (raw or "").lower()


def load_records(limit: int | None = None) -> list[dict]:
    path = download_file(_URL, _FILENAME)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    print(f"[amazon] Raw rows: {len(df)}")

    records: list[dict] = []
    for _, row in df.iterrows():
        category = schema.clean_text(row.get("Category"), limit=120) or "Uncategorized"
        asin = (row.get("Product Asin") or "").strip()
        product_url = f"https://www.amazon.in/dp/{asin}" if asin else ""

        rec = schema.assemble(
            id=f"{KEY}:{row.get('Uniq Id') or asin}",
            name=row.get("Product Title"),
            description=row.get("Product Description"),
            brand=row.get("Brand"),
            # Amazon's category is flat, so top == subcategory (already clean).
            category=category,
            subcategory=category,
            category_path=category,
            retail_price=schema.to_float(row.get("Mrp")),
            discounted_price=schema.to_float(row.get("Price")),
            rating=None,  # not present in this dataset
            in_stock=_stock(row.get("Stock Availibility")),
            image_url=schema.first_image(row.get("Image Urls")),
            product_url=product_url,
        )
        if rec:
            records.append(rec)
        if limit and len(records) >= limit:
            break

    print(f"[amazon] Clean records: {len(records)}")
    return records
