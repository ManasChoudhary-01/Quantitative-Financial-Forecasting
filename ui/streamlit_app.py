"""
Streamlit UI for the Semantic Product Search & Recommendation Engine.

Thin client: every action calls the FastAPI backend over HTTP. Start the API
first (uvicorn api.main:app), then run:

    streamlit run ui/streamlit_app.py

Tabs:
    🔍 Search    -> natural-language hybrid search; filters sit under the bar
    💬 Assistant -> RAG shopping assistant (grounded, conversational)
    ✨ For You    -> personalized recommendations from your behaviour

Liking / buying / viewing a product logs an event that feeds "For You".
"""

from __future__ import annotations

import os

import requests
import streamlit as st

# The backend base URL (override with API_URL env var if deployed elsewhere).
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Semantic Product Search", page_icon="🛍️", layout="wide")


# ----------------------------------------------------------------------------
# API helpers
# ----------------------------------------------------------------------------
def api_get(path: str, **params):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error on GET {path}: {e}")
        return None


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error on POST {path}: {e}")
        return None


@st.cache_data(ttl=60)
def get_meta():
    """Catalog metadata (cached briefly to avoid hammering the API)."""
    return api_get("/meta")


@st.cache_data(ttl=60)
def get_subcategories(category: str | None):
    """Subcategory facet values for a category (cached)."""
    data = api_get("/subcategories", category=category) if category else api_get("/subcategories")
    return data.get("subcategories", []) if data else []


def log_event(product_id: str, action: str):
    api_post("/event", {"user_id": st.session_state.user_id, "product_id": product_id, "action": action})


# ----------------------------------------------------------------------------
# Product card rendering
# ----------------------------------------------------------------------------
def render_product(p: dict, key_prefix: str):
    """Render one product as a card with image, price (INR) and action buttons."""
    with st.container(border=True):
        if p.get("image_url"):
            # Streamlit shows a placeholder if an image URL can't load.
            st.image(p["image_url"], use_container_width=True)

        name = p.get("name", "")
        st.markdown(f"**{name[:80]}**")

        # Price line: show discount with struck-through MRP when applicable.
        price = int(p.get("price", 0))
        if p.get("discount_pct", 0) and p.get("retail_price", 0) > p.get("price", 0):
            st.markdown(
                f"### ₹{price:,} "
                f"<span style='color:gray;text-decoration:line-through;font-size:0.8em'>"
                f"₹{int(p['retail_price']):,}</span> "
                f"<span style='color:green;font-size:0.7em'>{p['discount_pct']:.0f}% off</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"### ₹{price:,}")

        meta_bits = [p.get("brand", "")]
        if p.get("rating"):
            meta_bits.append(f"{p['rating']}★")
        st.caption(" · ".join(b for b in meta_bits if b))
        st.caption(p.get("category_path", ""))

        # Action buttons -> each logs an event that personalizes "For You".
        c1, c2, c3 = st.columns(3)
        pid = p.get("id", "")
        if c1.button("👍 Like", key=f"{key_prefix}_like_{pid}"):
            log_event(pid, "like")
            st.toast("Liked! Check the 'For You' tab.")
        if c2.button("🛒 Buy", key=f"{key_prefix}_buy_{pid}"):
            log_event(pid, "purchase")
            st.toast("Purchase logged!")
        if c3.button("🔎 Similar", key=f"{key_prefix}_sim_{pid}"):
            log_event(pid, "view")
            st.session_state.similar_to = pid


def render_grid(products: list[dict], key_prefix: str, cols: int = 3):
    """Lay out product cards in a responsive grid."""
    if not products:
        st.info("No products to show.")
        return
    columns = st.columns(cols)
    for i, p in enumerate(products):
        with columns[i % cols]:
            render_product(p, key_prefix=f"{key_prefix}_{i}")


# ----------------------------------------------------------------------------
# Session state defaults
# ----------------------------------------------------------------------------
st.session_state.setdefault("user_id", "demo-user")
st.session_state.setdefault("chat_history", [])     # list of {role, content, products}
st.session_state.setdefault("similar_to", None)
st.session_state.setdefault("searching", False)     # True while a search is in flight
st.session_state.setdefault("search_results", None) # last search response (persists across reruns)
st.session_state.setdefault("query_input", "")


def _start_search():
    """Callback fired on Search click / example chip -> arms the search."""
    st.session_state.searching = True


def _set_query_and_search(text: str):
    """Example-chip callback: fill the box and run."""
    st.session_state.query_input = text
    st.session_state.searching = True


# ----------------------------------------------------------------------------
# Sidebar: identity + backend status only (filters now live under the search bar)
# ----------------------------------------------------------------------------
meta = get_meta()

with st.sidebar:
    st.title("🛍️ Shop AI")
    st.session_state.user_id = st.text_input("User ID", st.session_state.user_id,
                                              help="Switch users to see different recommendations.")
    st.divider()
    if meta:
        st.metric("Products indexed", f"{meta['product_count']:,}")
        st.caption(f"Vector backend: **{meta['vector_backend']}**")
        st.caption(f"Reranker: {'🟢 on' if meta['reranker_enabled'] else '⚪ off'}")
        st.caption(f"Gemini RAG: {'🟢 on' if meta['gemini_available'] else '⚪ offline (extractive)'}")
    else:
        st.error("Backend not reachable.\n\nStart it with:\n\n`uvicorn api.main:app`")
        st.stop()

max_price = int(meta.get("price_max", 100000)) or 100000


# ----------------------------------------------------------------------------
# Header + tabs
# ----------------------------------------------------------------------------
st.title("Semantic Product Search & Recommendation Engine")
st.caption("Natural-language search, personalized recommendations, and an AI shopping assistant — prices in ₹.")
tab_search, tab_chat, tab_reco = st.tabs(["🔍 Search", "💬 Assistant", "✨ For You"])


# ============================================================================
# Search tab
# ============================================================================
with tab_search:
    # --- Search bar ---
    query = st.text_input(
        "Search products in natural language",
        key="query_input",
        placeholder="e.g. printed cotton shirt for men under 1000",
        label_visibility="collapsed",
    )

    # --- Filters, directly under the search bar ---
    with st.container(border=True):
        st.caption("Filters")
        f1, f2, f3, f4 = st.columns([2, 2, 3, 1])

        categories = ["All categories"] + meta.get("categories", [])
        sel_category = f1.selectbox("Category", categories, key="f_category")
        category = None if sel_category == "All categories" else sel_category

        # Product-type facet depends on the chosen category.
        sub_options = ["All types"] + get_subcategories(category)
        sel_subcategory = f2.selectbox("Product type", sub_options, key="f_subcat")
        subcategory = None if sel_subcategory == "All types" else sel_subcategory

        budget = f3.slider("Max budget (₹)", 0, max_price, max_price,
                           step=max(1, max_price // 100), key="f_budget")
        in_stock_only = f4.checkbox("In stock", value=False, key="f_stock")

    # --- Search button (disabled while a search is running) ---
    st.button(
        "🔍 Searching…" if st.session_state.searching else "🔍 Search",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.searching,
        on_click=_start_search,
    )

    # --- Example query chips (quick, intuitive starting points) ---
    st.caption("Try:")
    ex_cols = st.columns(4)
    examples = ["face serum for glowing skin", "slim fit jeans for men",
                "gift under 1500", "organic green tea"]
    for col, ex in zip(ex_cols, examples):
        col.button(ex, key=f"ex_{ex}", use_container_width=True,
                   disabled=st.session_state.searching,
                   on_click=_set_query_and_search, args=(ex,))

    # --- Run the search when armed (button/chip set `searching=True`) ---
    if st.session_state.searching:
        q = st.session_state.query_input.strip()
        if not q:
            st.session_state.searching = False
            st.warning("Please type something to search for.")
        else:
            with st.spinner(f"🔍 Searching for “{q}” …"):
                data = api_post("/search", {
                    "query": q,
                    "top_k": 12,
                    "price_max": budget,
                    "category": category,
                    "subcategory": subcategory,
                    "in_stock_only": in_stock_only,
                })
            st.session_state.search_results = data
            st.session_state.searching = False
            st.rerun()  # rerun to re-enable the button and render results cleanly

    # --- Results (persisted across reruns) ---
    results = st.session_state.search_results
    if results is not None:
        st.markdown(f"**{results['count']} results** for _“{results['query']}”_")
        render_grid(results["results"], key_prefix="search")

    # --- "Similar products" drawer (triggered by a card's 🔎 button) ---
    if st.session_state.similar_to:
        st.divider()
        top = st.columns([4, 1])
        top[0].subheader("🔎 Similar products")
        if top[1].button("✕ Close", use_container_width=True):
            st.session_state.similar_to = None
            st.rerun()
        sim = api_get(f"/similar/{st.session_state.similar_to}", top_k=6)
        if sim:
            render_grid(sim["results"], key_prefix="similar", cols=3)


# ============================================================================
# Assistant tab
# ============================================================================
with tab_chat:
    st.caption("Ask in plain language — answers are grounded in real products, and "
               "a budget mentioned in your message (e.g. “under ₹1500”) is applied automatically.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("products"):
                render_grid(msg["products"][:3], key_prefix=f"chat_{id(msg)}", cols=3)

    prompt = st.chat_input("e.g. I need a birthday gift for my sister under ₹1500")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🤖 Finding the best products for you…"):
                data = api_post("/chat", {"message": prompt, "top_k": 5})
            if data:
                st.markdown(data["answer"])
                render_grid(data.get("products", [])[:3], key_prefix="chatnew", cols=3)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": data["answer"], "products": data.get("products", [])}
                )


# ============================================================================
# For You tab
# ============================================================================
with tab_reco:
    st.caption(f"Personalized for **{st.session_state.user_id}**, based on products "
               f"you liked / bought / viewed in the Search tab.")
    reco_budget = st.slider("Max budget (₹)", 0, max_price, max_price,
                            step=max(1, max_price // 100), key="reco_budget")
    with st.spinner("Building your recommendations…"):
        data = api_post("/recommend", {
            "user_id": st.session_state.user_id,
            "top_k": 12,
            "price_max": reco_budget if reco_budget < max_price else None,
        })
    if data and data["results"]:
        render_grid(data["results"], key_prefix="reco")
    else:
        st.info("No recommendations yet. Like or buy a few products in the Search "
                "tab, then come back here.")
