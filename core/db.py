"""
SQLite data layer.

Holds two things:
  1. `products`  -> the cleaned catalog metadata (price, category, image, ...).
     The vector store only stores embeddings + ids; the *displayable* product
     data lives here and is joined back in after retrieval.
  2. `events`    -> lightweight user behaviour log (view / like / purchase),
     used by the recommender to build a personalized profile vector.

SQLite is used for the local demo. Because everything goes through this thin
module, swapping to Postgres later means changing only this file.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterable, Iterator

from config import settings
from ingestion.schema import PRODUCT_FIELDS


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """Context-managed SQLite connection with dict-like rows."""
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                id               TEXT PRIMARY KEY,
                name             TEXT,
                description      TEXT,
                brand            TEXT,
                category         TEXT,
                subcategory      TEXT,
                category_path    TEXT,
                retail_price     REAL,
                discounted_price REAL,
                price            REAL,
                discount_pct     REAL,
                rating           REAL,
                in_stock         INTEGER,
                image_url        TEXT,
                product_url      TEXT,
                embed_text       TEXT
            )
            """
        )
        # Index on price/category so metadata filtering stays fast.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_products_price ON products(price)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_products_subcategory ON products(subcategory)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                product_id TEXT NOT NULL,
                action     TEXT NOT NULL,      -- 'view' | 'like' | 'purchase'
                ts         INTEGER NOT NULL    -- unix seconds
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id)")


def upsert_products(records: Iterable[dict]) -> int:
    """Insert or replace product rows. Returns the number written."""
    cols = PRODUCT_FIELDS
    placeholders = ",".join("?" for _ in cols)
    sql = f"INSERT OR REPLACE INTO products ({','.join(cols)}) VALUES ({placeholders})"

    count = 0
    with get_conn() as conn:
        for r in records:
            # `in_stock` is a bool in Python but stored as 0/1 in SQLite.
            values = [int(r[c]) if c == "in_stock" else r.get(c) for c in cols]
            conn.execute(sql, values)
            count += 1
    return count


def _row_to_product(row: sqlite3.Row) -> dict:
    """Convert a SQLite row into a plain product dict (with real bool/None)."""
    d = dict(row)
    d["in_stock"] = bool(d.get("in_stock"))
    return d


def get_product(product_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()
    return _row_to_product(row) if row else None


def get_products(product_ids: list[str]) -> dict[str, dict]:
    """Fetch many products at once; returns {id: product} for the ids found."""
    if not product_ids:
        return {}
    placeholders = ",".join("?" for _ in product_ids)
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM products WHERE id IN ({placeholders})", product_ids
        ).fetchall()
    return {row["id"]: _row_to_product(row) for row in rows}


def all_products() -> list[dict]:
    """Return every product (used to build the local NumPy index)."""
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM products").fetchall()
    return [_row_to_product(r) for r in rows]


def list_categories(min_count: int = 10) -> list[str]:
    """
    Distinct top-level categories for UI filter dropdowns.

    The raw dataset has some malformed single-level category trees that are
    really product names (count == 1). We only surface categories that have at
    least `min_count` products, which cleanly keeps the real categories and
    drops the noise.
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT category, COUNT(*) AS n FROM products "
            "WHERE category != '' AND category != 'Uncategorized' "
            "GROUP BY category HAVING n >= ? ORDER BY category",
            (min_count,),
        ).fetchall()
    return [r["category"] for r in rows]


def list_subcategories(category: str | None = None, min_count: int = 5) -> list[str]:
    """
    Distinct subcategories (finest taxonomy level) for the facet dropdown.

    If `category` is given, only subcategories within that top-level category are
    returned -- so selecting "Mobiles & Accessories" narrows the facet to
    "Headphones", "Power Banks", etc. `min_count` drops noisy one-off values.
    """
    sql = (
        "SELECT subcategory, COUNT(*) AS n FROM products "
        "WHERE subcategory != '' AND subcategory != 'Uncategorized' "
    )
    params: list = []
    if category:
        sql += "AND category = ? "
        params.append(category)
    sql += "GROUP BY subcategory HAVING n >= ? ORDER BY subcategory"
    params.append(min_count)
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [r["subcategory"] for r in rows]


def price_bounds() -> tuple[float, float]:
    """Min/max price in the catalog (for UI slider bounds)."""
    with get_conn() as conn:
        row = conn.execute("SELECT MIN(price) AS lo, MAX(price) AS hi FROM products").fetchone()
    return (row["lo"] or 0.0, row["hi"] or 0.0)


def product_count() -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM products").fetchone()
    return int(row["n"])


# ---- Behaviour events (for personalized recommendations) -------------------

def log_event(user_id: str, product_id: str, action: str, ts: int) -> None:
    """Record a user interaction. `ts` is passed in (unix seconds)."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO events (user_id, product_id, action, ts) VALUES (?, ?, ?, ?)",
            (user_id, product_id, action, ts),
        )


def get_user_events(user_id: str, limit: int = 100) -> list[dict]:
    """Most recent events for a user (newest first)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE user_id = ? ORDER BY ts DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]
