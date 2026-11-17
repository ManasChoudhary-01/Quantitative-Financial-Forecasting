"""
FastAPI application -- the HTTP layer the Streamlit UI (or any client) calls.

Endpoints:
    GET  /health              -> liveness check
    GET  /meta                -> catalog facts for the UI (categories, price range...)
    POST /search              -> hybrid semantic + keyword search
    GET  /product/{id}        -> single product
    GET  /similar/{id}        -> "more like this" recommendations
    POST /recommend           -> personalized recommendations for a user
    POST /event               -> log a user interaction (powers personalization)
    POST /chat                -> RAG shopping assistant

Run: uvicorn api.main:app --reload
Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from core import db, llm, rag, recommender, search
from api.schemas import ChatRequest, EventRequest, RecommendRequest, SearchRequest

app = FastAPI(
    title="Semantic Product Search & Recommendation Engine",
    description="Hybrid (semantic + keyword) search, recommendations, and a RAG "
                "shopping assistant over an Indian e-commerce catalog (prices in INR).",
    version="1.0.0",
)

# Allow the Streamlit UI (different port) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    """Make sure the DB schema exists before serving requests."""
    db.init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/meta")
def meta() -> dict:
    """Catalog facts used to populate UI filters and status."""
    lo, hi = db.price_bounds()
    return {
        "product_count": db.product_count(),
        "categories": db.list_categories(),
        "price_min": lo,
        "price_max": hi,
        "vector_backend": settings.vector_backend,
        "gemini_available": llm.is_available(),
        "reranker_enabled": settings.use_reranker,
    }


@app.get("/subcategories")
def subcategories(category: str | None = None) -> dict:
    """Facet values (finest product type), optionally within a top category."""
    return {"category": category, "subcategories": db.list_subcategories(category)}


@app.post("/search")
def search_endpoint(req: SearchRequest) -> dict:
    results = search.search(
        query=req.query,
        top_k=req.top_k,
        price_min=req.price_min,
        price_max=req.price_max,
        category=req.category,
        subcategory=req.subcategory,
        in_stock_only=req.in_stock_only,
    )
    return {"query": req.query, "count": len(results), "results": results}


@app.get("/product/{product_id}")
def get_product(product_id: str) -> dict:
    product = db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


@app.get("/similar/{product_id}")
def similar(product_id: str, top_k: int = 10) -> dict:
    results = recommender.similar_products(product_id, top_k=top_k)
    return {"product_id": product_id, "count": len(results), "results": results}


@app.post("/recommend")
def recommend(req: RecommendRequest) -> dict:
    results = recommender.recommend_for_user(
        user_id=req.user_id, top_k=req.top_k, price_max=req.price_max
    )
    return {"user_id": req.user_id, "count": len(results), "results": results}


@app.post("/event")
def log_event(req: EventRequest) -> dict:
    """Record a user interaction (view/like/purchase) for personalization."""
    db.log_event(req.user_id, req.product_id, req.action, ts=int(time.time()))
    return {"status": "logged"}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    return rag.chat(
        message=req.message,
        top_k=req.top_k,
        price_max=req.price_max,
        category=req.category,
        subcategory=req.subcategory,
    )
