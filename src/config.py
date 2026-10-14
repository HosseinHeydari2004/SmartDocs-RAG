"""Central, environment-driven configuration.

Secrets (the OpenRouter API key) are only read from the environment here; the
runtime may additionally hold a session-scoped override in memory (see
``src.services.runtime``). Nothing in this module is ever logged or returned
by the API.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(minimum, int(raw))
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    # Vector DB
    vectordb_path: Path
    collection_name: str
    chroma_add_batch_size: int
    # Embeddings
    embedding_model: str
    embedding_device: str  # "auto" | "cpu" | "cuda" | "cuda:N" | "mps"
    embedding_batch_size: int
    # LLM (OpenRouter)
    llm_model: str
    llm_timeout_s: int
    openrouter_api_key: str | None
    openrouter_base_url: str | None
    # Retrieval / chat
    embedding_preload: bool
    search_k_default: int
    search_k_max: int
    max_chat_history: int
    # Uploads
    upload_max_mb: int
    upload_max_files: int
    # OCR
    ocr_default_lang: str
    ocr_dpi: int
    ocr_max_pages: int
    # Jobs
    max_retained_jobs: int

    @property
    def upload_max_bytes(self) -> int:
        return self.upload_max_mb * 1024 * 1024


def load_settings() -> Settings:
    key = os.getenv("OPENROUTER_API_KEY", "").strip() or None
    return Settings(
        vectordb_path=Path(_env_str("VECTORDB_PATH", str(PROJECT_ROOT / "vectordb"))),
        collection_name=_env_str("CHROMA_COLLECTION", "SmartDB"),
        chroma_add_batch_size=_env_int("CHROMA_ADD_BATCH_SIZE", 64),
        embedding_model=_env_str("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        embedding_device=_env_str("EMBEDDING_DEVICE", "auto").lower(),
        embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", 32),
        llm_model=_env_str("OPENROUTER_MODEL", "openrouter/auto"),
        llm_timeout_s=_env_int("LLM_TIMEOUT_S", 30),
        openrouter_api_key=key,
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "").strip() or None,
        embedding_preload=_env_str("EMBEDDING_PRELOAD", "1") != "0",
        search_k_default=_env_int("SEARCH_K", 4),
        search_k_max=_env_int("SEARCH_K_MAX", 10),
        max_chat_history=_env_int("MAX_CHAT_HISTORY", 6, minimum=0),
        upload_max_mb=_env_int("UPLOAD_MAX_MB", 25),
        upload_max_files=_env_int("UPLOAD_MAX_FILES", 10),
        ocr_default_lang=_env_str("OCR_LANG", "eng"),
        ocr_dpi=_env_int("OCR_DPI", 300),
        ocr_max_pages=_env_int("OCR_MAX_PAGES", 100),
        max_retained_jobs=_env_int("MAX_RETAINED_JOBS", 50),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
