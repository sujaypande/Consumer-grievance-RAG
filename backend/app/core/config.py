"""Centralized settings for the whole backend."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ directory -- everything else (data/, .env) is relative to this
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    Typed environment configuration, loaded once from backend/.env.

    Why pydantic-settings: it validates that required values (like the API
    key) are actually present at startup, instead of failing deep inside a
    RAG call with a confusing error.
    """

    google_api_key: str

    generation_model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-2"

    # Source document (single document only, per spec)
    source_pdf_path: Path = BACKEND_DIR.parent / "data" / "consumer protection act 2019.pdf"

    # Chroma's own on-disk format -- vectors, text, and metadata together
    chroma_persist_dir: Path = BACKEND_DIR / "data" / "chroma_db"

    # Final number of chunks handed to the LLM, after reranking
    retrieval_top_k: int = 4
    # Candidate pool size pulled from EACH of the dense/BM25 retrievers
    # before the ensemble merges and the reranker cuts back down to
    # retrieval_top_k. Wider than retrieval_top_k on purpose -- the whole
    # point of reranking is to have more candidates than needed so the
    # cross-encoder has real choices to rank between.
    hybrid_candidate_k: int = 10
    # Ensemble weights for combining dense (semantic) and BM25 (keyword)
    # retrieval -- equal weight since neither should dominate by default
    dense_weight: float = 0.5
    bm25_weight: float = 0.5
    # Local cross-encoder used to rerank the merged candidate pool
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
