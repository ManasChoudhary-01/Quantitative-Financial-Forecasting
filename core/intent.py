"""
Query type-intent rescoring.

Embedding models (and even Flipkart's own taxonomy) conflate near-synonym
product types -- "headphones" vs "earphones"/"earbuds" being the classic case:
they're semantically close AND Flipkart files earphones under "Headphones", so
neither the vector score nor the category cleanly separates them.

What DOES separate them is the product *name* ("...Headphone..." vs
"...Earphone..."). So when a query clearly names one type from a confusable
group, we reorder results so that:

    exact-type matches   > neutral items > the *other* confusable type

This runs AFTER reranking and only changes ordering when the query actually
names a type -- otherwise results pass through untouched.

To extend: add another set to CONFUSABLE_GROUPS (e.g. laptop vs tablet).
"""

from __future__ import annotations

import re

# Each group is a set of interchangeable surface terms for ONE real product
# type. Terms within a group are treated as equivalent; terms across groups are
# treated as *different* types that should not be mixed when a type is requested.
CONFUSABLE_GROUPS: list[set[str]] = [
    {"headphone", "headphones", "over-ear", "over ear", "on-ear", "on ear"},
    {"earphone", "earphones", "earbud", "earbuds", "in-ear", "in ear"},
]


def _matches_group(text: str, group: set[str]) -> bool:
    """True if any term in the group appears as a word/phrase in `text`."""
    for term in group:
        # \b word boundaries so "ear" doesn't match "earn"; phrases matched literally.
        if re.search(rf"\b{re.escape(term)}\b", text):
            return True
    return False


def detect_query_group(query: str) -> int | None:
    """Return the index of the confusable group the query names, or None."""
    q = query.lower()
    for idx, group in enumerate(CONFUSABLE_GROUPS):
        if _matches_group(q, group):
            return idx
    return None


def _product_bucket(product: dict, query_group: int) -> int:
    """
    Rank bucket for a product given the query's requested type group:
        2 -> name matches the requested type   (put first)
        1 -> name matches no confusable type    (neutral, keep in middle)
        0 -> name matches a DIFFERENT type       (wrong type, push to bottom)
    """
    name = (product.get("name") or "").lower()
    if _matches_group(name, CONFUSABLE_GROUPS[query_group]):
        return 2
    for idx, group in enumerate(CONFUSABLE_GROUPS):
        if idx != query_group and _matches_group(name, group):
            return 0
    return 1


def apply(query: str, products: list[dict]) -> list[dict]:
    """
    Reorder `products` by type-intent when the query names a confusable type.

    Stable within each bucket: preserves the existing (rerank) order, so this
    only *demotes* wrong-type items rather than reshuffling good results.
    """
    group = detect_query_group(query)
    if group is None or not products:
        return products  # no type intent -> leave ordering untouched

    # Sort by bucket (desc). Python's sort is stable, so ties keep rerank order.
    ranked = sorted(products, key=lambda p: _product_bucket(p, group), reverse=True)
    # Tag for transparency/debugging in the API response.
    for p in ranked:
        p["type_match"] = _product_bucket(p, group)
    return ranked
