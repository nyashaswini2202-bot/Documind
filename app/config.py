"""Central configuration.

Every tunable value lives here and can be overridden with environment variables
(or a local `.env` file). Nothing else in the codebase reads os.environ directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # ---- storage ----------------------------------------------------------
    data_dir: Path = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))

    # ---- models -----------------------------------------------------------
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    use_reranker: bool = _bool("USE_RERANKER", True)
    warmup: bool = _bool("WARMUP", True)  # load models in the background at startup

    # ---- chunking (characters) ---------------------------------------------
    chunk_size: int = _int("CHUNK_SIZE", 800)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 150)

    # ---- retrieval ---------------------------------------------------------
    top_k: int = _int("TOP_K", 5)                       # passages sent to the LLM
    candidates: int = _int("RETRIEVAL_CANDIDATES", 20)  # passages fetched before reranking
    rrf_k: int = _int("RRF_K", 60)                      # Reciprocal Rank Fusion constant
    min_rerank_score: float = _float("MIN_RERANK_SCORE", 0.01)  # 0..1 relevance probability
    min_dense_score: float = _float("MIN_DENSE_SCORE", 0.15)    # cosine similarity (dense-only mode)

    # ---- LLM ---------------------------------------------------------------
    llm_provider: str = os.getenv("LLM_PROVIDER", "auto").lower()  # auto | groq | ollama | none
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    llm_temperature: float = _float("LLM_TEMPERATURE", 0.1)
    llm_max_tokens: int = _int("LLM_MAX_TOKENS", 700)

    # ---- limits (protect a public demo) -------------------------------------
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 15)
    max_docs: int = _int("MAX_DOCS_PER_WORKSPACE", 10)
    max_chunks: int = _int("MAX_CHUNKS_PER_WORKSPACE", 6000)

    @property
    def resolved_provider(self) -> str:
        if self.llm_provider == "auto":
            return "groq" if self.groq_api_key else "none"
        if self.llm_provider == "groq" and not self.groq_api_key:
            return "none"
        return self.llm_provider if self.llm_provider in {"groq", "ollama"} else "none"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
if settings.chunk_overlap >= settings.chunk_size:
    raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
