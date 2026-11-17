"""
Gemini LLM wrapper for the RAG shopping assistant.

Kept deliberately thin so `rag.py` doesn't care about SDK details. If no
GEMINI_API_KEY is configured, `is_available()` returns False and the RAG layer
falls back to a non-LLM extractive answer -- so the app runs fully offline.
"""

from __future__ import annotations

import threading

from config import settings

_model = None
_model_lock = threading.Lock()


def is_available() -> bool:
    """True only when a Gemini API key is configured."""
    return settings.use_gemini


def _get_model():
    """Lazily configure and cache the Gemini model client."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                import google.generativeai as genai

                genai.configure(api_key=settings.gemini_api_key)
                _model = genai.GenerativeModel(settings.gemini_model)
    return _model


def generate(prompt: str, temperature: float = 0.3, max_output_tokens: int = 600) -> str:
    """
    Send a prompt to Gemini and return the text response.

    Raises if called without a key -- callers should gate on `is_available()`.
    """
    if not is_available():
        raise RuntimeError("Gemini is not configured (missing GEMINI_API_KEY).")

    import google.generativeai as genai

    model = _get_model()
    resp = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )
    # `.text` conveniently concatenates the response parts.
    return (resp.text or "").strip()
