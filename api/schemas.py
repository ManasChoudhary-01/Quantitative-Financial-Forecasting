"""
Pydantic request models for the API.

Responses are returned as plain dicts (products carry a variable set of score
fields depending on the endpoint), so we validate inputs strictly here and keep
outputs flexible.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural-language search query")
    top_k: int | None = Field(None, ge=1, le=50)
    price_min: float | None = Field(None, ge=0, description="Min price in INR")
    price_max: float | None = Field(None, ge=0, description="Max price in INR (budget)")
    category: str | None = None
    subcategory: str | None = Field(None, description="Finest product-type facet")
    in_stock_only: bool = False


class RecommendRequest(BaseModel):
    user_id: str
    top_k: int = Field(10, ge=1, le=50)
    price_max: float | None = Field(None, ge=0)


class EventRequest(BaseModel):
    user_id: str
    product_id: str
    action: Literal["view", "like", "purchase"]


class ChatRequest(BaseModel):
    message: str
    top_k: int = Field(5, ge=1, le=20)
    price_max: float | None = Field(None, ge=0)
    category: str | None = None
    subcategory: str | None = None
