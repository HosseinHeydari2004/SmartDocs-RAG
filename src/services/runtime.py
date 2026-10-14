"""Process-wide singletons: embedding model, Chroma store, LLM client, preprocessing.

Everything is created lazily and exactly once per process. The embedding model is
only loaded on first real use (ingest or search), so health/stats/list calls stay fast.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.embeddings import Embeddings

from src.config import Settings
from src.errors import (
    AppError,
    EmbeddingModelError,
    LLMNotConfiguredError,
    LLMUnavailableError,
    translate_embedding_exception,
    translate_llm_exception,
)

logger = logging.getLogger(__name__)


class LazyEmbeddings(Embeddings):
    """Loads the real embedding model on first use, once, thread-safely."""

    def __init__(self, loader: Callable[[], Embeddings]) -> None:
        self._loader = loader
        self._instance: Embeddings | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._instance is not None

    def _get(self) -> Embeddings:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    try:
                        self._instance = self._loader()
                    except Exception as exc:  # not cached: a later call may retry
                        logger.exception("Embedding model failed to load")
                        raise EmbeddingModelError() from exc
        return self._instance

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._get().embed_documents(texts)
        except Exception as exc:
            raise translate_embedding_exception(exc) from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._get().embed_query(text)
        except Exception as exc:
            raise translate_embedding_exception(exc) from exc


@dataclass
class LlmConfig:
    model: str
    api_key: str | None
    key_source: str  # "env" | "session" | "none"


class Runtime:
    def __init__(
        self,
        settings: Settings,
        *,
        embeddings_loader: Callable[[], Embeddings] | None = None,
        chat_model_factory: Callable[[str, str], Any] | None = None,
    ) -> None:
        self.settings = settings
        if embeddings_loader is None:
            from src.embeddings import build_embeddings

            embeddings_loader = lambda: build_embeddings(  # noqa: E731
                settings.embedding_model, settings.embedding_device, settings.embedding_batch_size
            )
        self.embeddings = LazyEmbeddings(embeddings_loader)
        self._chat_model_factory = chat_model_factory or self._default_chat_factory
        self._lock = threading.RLock()
        self._vector_store = None
        self._preprocessing = None
        self._metadata_extractor = None
        self._chat_cache: tuple[tuple[str, str], Any] | None = None
        self._llm = LlmConfig(
            model=settings.llm_model,
            api_key=settings.openrouter_api_key,
            key_source="env" if settings.openrouter_api_key else "none",
        )

    # ---------------------------------------------------------------- factories
    def _default_chat_factory(self, model: str, api_key: str) -> Any:
        from src.llm import build_chat_model

        return build_chat_model(
            model, api_key, timeout=self.settings.llm_timeout_s, base_url=self.settings.openrouter_base_url
        )

    @property
    def vector_store(self):
        if self._vector_store is None:
            with self._lock:
                if self._vector_store is None:
                    from src.vectordb.chroma import VectorStore

                    self._vector_store = VectorStore(self.settings, embeddings=self.embeddings)
        return self._vector_store

    @property
    def metadata_extractor(self):
        if self._metadata_extractor is None:
            with self._lock:
                if self._metadata_extractor is None:
                    from src.preprocessing.parsing.metadata_extraction import MetadataExtractor

                    self._metadata_extractor = MetadataExtractor(self.chat_model)
        return self._metadata_extractor

    @property
    def preprocessing(self):
        if self._preprocessing is None:
            with self._lock:
                if self._preprocessing is None:
                    from src.preprocessing.chunking.semantic import SemanticChuker
                    from src.preprocessing.pipeline import PreprocessingPipeline

                    self._preprocessing = PreprocessingPipeline(
                        chunker_factory=lambda: SemanticChuker(embeddings=self.embeddings),
                        metadata_extractor=self.metadata_extractor,
                    )
        return self._preprocessing

    def preload_embeddings(self) -> threading.Thread:
        """Load the embedding model in the background so the first request is not slow."""

        def _load() -> None:
            try:
                self.embeddings.embed_query("warm-up")
                logger.info("Embedding model preloaded")
            except Exception:
                logger.warning("Embedding preload failed (it will be retried on first use)")

        thread = threading.Thread(target=_load, name="embedding-preload", daemon=True)
        thread.start()
        return thread

    # ---------------------------------------------------------------- LLM
    def chat_model(self) -> Any:
        with self._lock:
            cfg = self._llm
            if not cfg.api_key:
                raise LLMNotConfiguredError()
            cache_key = (cfg.model, cfg.api_key)
            if self._chat_cache is None or self._chat_cache[0] != cache_key:
                try:
                    self._chat_cache = (cache_key, self._chat_model_factory(cfg.model, cfg.api_key))
                except AppError:
                    raise
                except Exception as exc:
                    logger.exception("Could not build the chat model")
                    raise LLMUnavailableError(
                        "The LLM client could not be created. Check the model name and installed packages."
                    ) from exc
            return self._chat_cache[1]

    def configure_llm(self, *, api_key: str | None = None, model: str | None = None) -> None:
        """Update the in-memory LLM configuration (never persisted, never logged)."""
        with self._lock:
            if model:
                self._llm.model = model
            if api_key:
                self._llm.api_key = api_key
                self._llm.key_source = "session"
            self._chat_cache = None

    def clear_session_key(self) -> None:
        with self._lock:
            env_key = self.settings.openrouter_api_key
            self._llm.api_key = env_key
            self._llm.key_source = "env" if env_key else "none"
            self._chat_cache = None

    def llm_status(self) -> dict[str, Any]:
        cfg = self._llm
        return {
            "provider": "openrouter",
            "model": cfg.model,
            "key_configured": bool(cfg.api_key),
            "key_source": cfg.key_source,
        }

    def test_llm(self) -> None:
        """One tiny real call; raises a translated ``AppError`` on failure."""
        model = self.chat_model()
        try:
            model.invoke("Reply with the single word: OK", max_tokens=8)
        except Exception as exc:
            logger.warning("LLM connection test failed: %s", type(exc).__name__)
            raise translate_llm_exception(exc) from exc
