"""
Canonical product schema + shared cleaning helpers.

Every dataset adapter (see `ingestion/sources/`) parses its own raw format and
produces records in THIS one schema, so the rest of the app never needs to know
which dataset a product came from. Prices are always INR.
"""

from __future__ import annotations

import json
import re
from typing import Any

# The canonical, cleaned product schema (also the columns of the SQLite table).
PRODUCT_FIELDS = [
    "id",
    "name",
    "description",
    "brand",
    "category",         # top-level category, e.g. "Fashion", "Skin Care"
    "subcategory",      # finer facet, e.g. "Women", "Headphones"
    "category_path",    # clean taxonomy path
    "retail_price",     # MRP in INR
    "discounted_price", # selling price in INR
    "price",            # effective price (discounted if available, else retail)
    "discount_pct",     # % off, 0 if no discount
    "rating",           # float 0-5, or None
    "in_stock",         # bool
    "image_url",        # first product image
    "product_url",      # source product page
    "embed_text",       # the text we actually embed for semantic search
]


# ---- Generic field cleaners (reused by all adapters) -----------------------

def clean_text(raw: Any, limit: int = 1500) -> str:
    """Collapse whitespace, strip HTML tags, and cap length."""
    if not isinstance(raw, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", raw)        # drop HTML tags (Myntra uses <br>)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def to_float(value: Any) -> float | None:
    """Best-effort float conversion; returns None for blanks/garbage/NaN."""
    if value is None:
        return None
    try:
        # Strip currency symbols / commas that sometimes appear in price fields.
        if isinstance(value, str):
            value = re.sub(r"[^\d.]", "", value)
            if not value:
                return None
        f = float(value)
        return f if f == f else None  # NaN != NaN
    except (ValueError, TypeError):
        return None


def parse_rating(raw: Any) -> float | None:
    """Parse a rating; non-numeric ('No rating available') becomes None."""
    f = to_float(raw)
    return round(f, 1) if f is not None else None


def first_image(raw: Any) -> str:
    """
    Extract the first image URL from a variety of raw formats:
    a JSON list string, a pipe/comma separated string, or a single URL.
    """
    if not isinstance(raw, str) or not raw.strip():
        return ""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0])
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"https?://[^\s\"'|,]+", raw)
    return match.group(0) if match else ""


def build_embed_text(name: str, brand: str, subcategory: str,
                     category_path: str, description: str) -> str:
    """
    Compose the string that gets embedded. Front-load the fields a shopper would
    describe (name, brand, product type, category), then the description.
    Good embedding text matters more than the choice of model.
    """
    parts = [p for p in [
        name,
        f"Brand: {brand}" if brand and brand != "Unknown" else "",
        f"Type: {subcategory}" if subcategory and subcategory != "Uncategorized" else "",
        category_path,
        description,
    ] if p]
    return ". ".join(parts)


def assemble(
    *,
    id: str,
    name: str,
    description: str = "",
    brand: str = "Unknown",
    category: str = "Uncategorized",
    subcategory: str = "Uncategorized",
    category_path: str = "",
    retail_price: float | None = None,
    discounted_price: float | None = None,
    rating: float | None = None,
    in_stock: bool = True,
    image_url: str = "",
    product_url: str = "",
) -> dict | None:
    """
    Build one canonical product record from already-parsed fields.

    Centralizes the price/discount/embed-text logic so every adapter stays
    consistent. Returns None (skip the row) when there's no usable name or price.
    """
    # Clean all text fields up front so downstream (incl. embed text) is safe
    # even when a raw value is numeric/NaN rather than a string.
    name = clean_text(name, limit=300)
    if not name:
        return None

    price = discounted_price if discounted_price else retail_price
    if not price:
        return None  # no price -> can't show or budget-filter it

    description = clean_text(description)
    brand = clean_text(brand, limit=120) or "Unknown"
    category = category or "Uncategorized"
    subcategory = subcategory or "Uncategorized"
    category_path = category_path or category

    discount_pct = 0.0
    if retail_price and discounted_price and retail_price > discounted_price:
        discount_pct = round((retail_price - discounted_price) / retail_price * 100, 1)

    return {
        "id": str(id)[:80],
        "name": name,
        "description": description,
        "brand": brand,
        "category": category,
        "subcategory": subcategory,
        "category_path": category_path,
        "retail_price": retail_price or price,
        "discounted_price": discounted_price or price,
        "price": price,
        "discount_pct": discount_pct,
        "rating": rating,
        "in_stock": bool(in_stock),
        "image_url": image_url or "",
        "product_url": clean_text(product_url, limit=500),
        "embed_text": build_embed_text(name, brand, subcategory, category_path, description),
    }
