"""Pydantic-settings configuration loaded from .env."""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py → backend/ → project root
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent


def _resolve_relative(path_str: str) -> str:
    """Resolve relative paths against the project root (CWD-independent)."""
    p = Path(path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p.resolve())


def _resolve_sqlite_url(url: str) -> str:
    """Make sqlite URLs point at an absolute DB path under the project root."""
    prefixes = ("sqlite+aiosqlite:///", "sqlite:///")
    for prefix in prefixes:
        if url.startswith(prefix):
            db_path = url[len(prefix) :]
            # Already absolute (Unix /path or Windows C:/path)
            if db_path.startswith("/") or (len(db_path) > 2 and db_path[1] == ":"):
                return url
            resolved = (PROJECT_ROOT / db_path).resolve()
            return f"{prefix}{resolved.as_posix()}"
    return url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    LLM_PROVIDER: str = "groq"
    LLM_FALLBACK_PROVIDER: str = "gemini"
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_FALLBACK_MODEL: str = "gemini-2.0-flash"

    # ── Embeddings ───────────────────────────────────────
    EMBEDDING_PROVIDER: str = "fastembed"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM: int = 384

    # ── PDF ──────────────────────────────────────────────
    PDF_PARSER: str = "auto"

    # ── Retrieval ────────────────────────────────────────
    HYBRID_ENABLED: bool = True
    RERANK_ENABLED: bool = True
    RERANK_MODEL: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    RELEVANCE_THRESHOLD: float = 0.35
    RERANK_THRESHOLD: float = 0.05

    # ── Features ─────────────────────────────────────────
    GROUNDEDNESS_ENABLED: bool = True
    AGENTIC_ENABLED: bool = True
    CONTEXTUAL_RETRIEVAL_LLM: bool = True
    HYDE_ENABLED: bool = False

    # ── Database ─────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./ragchatbot.db"

    # ── Qdrant ───────────────────────────────────────────
    QDRANT_PATH: str = "./qdrant_data"
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "document_chunks"

    # ── Server ───────────────────────────────────────────
    BACKEND_CORS_ORIGINS: str = "http://localhost:3000"
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_MB: int = 50

    @model_validator(mode="after")
    def _absolutize_paths(self) -> "Settings":
        self.DATABASE_URL = _resolve_sqlite_url(self.DATABASE_URL)
        self.UPLOAD_DIR = _resolve_relative(self.UPLOAD_DIR)
        self.QDRANT_PATH = _resolve_relative(self.QDRANT_PATH)
        return self

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.BACKEND_CORS_ORIGINS.split(",") if o.strip()]

    @property
    def upload_path(self) -> Path:
        p = Path(self.UPLOAD_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
