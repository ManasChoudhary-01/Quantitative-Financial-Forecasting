"""
Central configuration for the whole project.

Everything is loaded from environment variables (or a local `.env` file) using
pydantic-settings. Import the singleton `settings` anywhere in the codebase:

    from config import settings
    print(settings.embedding_model)

This "config-first" approach means no magic constants are scattered around the
code -- change behaviour by editing `.env`, not the source.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the folder that contains this `config/` package's parent.
# We resolve paths relative to it so the app works no matter where it's launched.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    """Typed application settings, populated from `.env` / environment."""

    # pydantic-settings config: read from `.env`, ignore unknown keys, case-insensitive.
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Vector store ----------------------------------------------------
    vector_backend: str = "local"  # "local" or "pinecone"

    # ---- Pinecone --------------------------------------------------------
    pinecone_api_key: str = ""
    pinecone_index_name: str = "semantic-products"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # ---- Embeddings ------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # ---- Reranker --------------------------------------------------------
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True

    # ---- Gemini LLM ------------------------------------------------------
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"

    # ---- Hybrid search tuning -------------------------------------------
    hybrid_alpha: float = 0.6          # weight on semantic vs keyword
    search_candidates: int = 50        # candidates fetched before reranking
    search_top_k: int = 10             # final results returned

    # ---- Data / ingestion -----------------------------------------------
    # Comma-separated dataset keys to ingest. Available: amazon, myntra, flipkart.
    datasets: str = "amazon,myntra"
    max_products: int = 0              # 0 = all; otherwise a cap PER dataset

    @property
    def dataset_keys(self) -> list[str]:
        return [k.strip().lower() for k in self.datasets.split(",") if k.strip()]

    # ---- Derived filesystem paths (not read from env) -------------------
    @property
    def data_dir(self) -> Path:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return DATA_DIR

    @property
    def raw_dir(self) -> Path:
        d = self.data_dir / "raw"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def index_dir(self) -> Path:
        d = self.data_dir / "index"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def raw_csv_path(self) -> Path:
        """Where the downloaded Flipkart CSV lives."""
        return self.raw_dir / "flipkart_com-ecommerce_sample.csv"

    @property
    def db_path(self) -> Path:
        """SQLite file holding product metadata + user behaviour events."""
        return self.data_dir / "products.db"

    @property
    def use_gemini(self) -> bool:
        """True only when a Gemini key is configured."""
        return bool(self.gemini_api_key.strip())

    @property
    def use_pinecone(self) -> bool:
        return self.vector_backend.strip().lower() == "pinecone"


# Singleton imported everywhere. Reads `.env` once at import time.
settings = Settings()
