"""
RAG shopping assistant: Retrieval-Augmented Generation.

Flow:
  1. RETRIEVE  -> run hybrid search to find products relevant to the user's message.
  2. AUGMENT   -> build a prompt that includes those products as grounded CONTEXT.
  3. GENERATE  -> Gemini answers using ONLY that context (no hallucinated products).

If Gemini isn't configured, we fall back to a simple extractive summary so the
assistant still works offline. Either way the answer is grounded in real,
retrieved products -- that's the whole point of RAG.
"""

from __future__ import annotations

import re

from core import llm, search

# Instructions that keep the model grounded and on-brand (Indian store, INR).
SYSTEM_PROMPT = """You are a helpful shopping assistant for an Indian e-commerce store.
All prices are in Indian Rupees (₹).
Rules:
- Recommend ONLY products from the CONTEXT below. Never invent products or prices.
- If nothing in the context fits the request, say so honestly.
- Recommend the best 2-3 options, compare them briefly, and mention price and why each fits.
- Respect any stated budget. Be concise and friendly."""


def _extract_budget(text: str) -> float | None:
    """
    Pull a budget out of natural language, e.g. 'under 2000', 'below ₹5,000',
    'budget 1500'. Returns the amount in INR, or None.
    """
    t = text.lower().replace(",", "")
    patterns = [
        r"(?:under|below|less than|within|budget(?: of)?|max(?:imum)?)\s*(?:rs\.?|inr|₹)?\s*(\d+)",
        r"(?:rs\.?|inr|₹)\s*(\d+)\s*(?:or less|max|budget)?",
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            return float(m.group(1))
    return None


def _build_context(products: list[dict]) -> str:
    """Format retrieved products as a numbered CONTEXT block for the prompt."""
    lines = []
    for i, p in enumerate(products, 1):
        rating = f"{p['rating']}★" if p.get("rating") else "no rating"
        desc = (p.get("description") or "")[:180]
        lines.append(
            f"[{i}] {p['name']} — ₹{int(p['price'])} "
            f"(brand: {p.get('brand', 'Unknown')}, category: {p.get('category', '')}, {rating})\n"
            f"    {desc}"
        )
    return "\n".join(lines)


def _fallback_answer(query: str, products: list[dict]) -> str:
    """Extractive answer used when no LLM is available (offline mode)."""
    if not products:
        return f"I couldn't find products matching '{query}'. Try different keywords or a higher budget."
    top = products[:3]
    lines = [f"Here are the best matches I found for '{query}':", ""]
    for p in top:
        rating = f" · {p['rating']}★" if p.get("rating") else ""
        lines.append(f"• {p['name']} — ₹{int(p['price'])} ({p.get('brand', 'Unknown')}{rating})")
    lines.append("\n(Set a GEMINI_API_KEY to get conversational, comparative recommendations.)")
    return "\n".join(lines)


def chat(
    message: str,
    top_k: int = 5,
    price_max: float | None = None,
    category: str | None = None,
    subcategory: str | None = None,
) -> dict:
    """
    Answer a shopping question, grounded in retrieved products.

    Returns:
        {"answer": str, "products": [product dicts used as context]}
    """
    message = (message or "").strip()
    if not message:
        return {"answer": "What are you shopping for today?", "products": []}

    # If the user didn't pass an explicit budget, try to read one from the text.
    if price_max is None:
        price_max = _extract_budget(message)

    # 1) RETRIEVE relevant products via hybrid search.
    products = search.search(message, top_k=top_k, price_max=price_max,
                             category=category, subcategory=subcategory)

    # 2) + 3) AUGMENT and GENERATE.
    if not products:
        return {"answer": _fallback_answer(message, products), "products": []}

    if not llm.is_available():
        return {"answer": _fallback_answer(message, products), "products": products}

    context = _build_context(products)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT (retrieved products):\n{context}\n\n"
        f"Customer message: {message}\n\n"
        f"Your recommendation:"
    )
    try:
        answer = llm.generate(prompt)
    except Exception as e:  # network/model errors -> graceful fallback
        answer = _fallback_answer(message, products) + f"\n\n(LLM error: {e})"

    return {"answer": answer, "products": products}
