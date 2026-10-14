"""Question answering: LLM filter extraction -> filtered retrieval -> grounded answer."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from src.errors import (
    AppError,
    EmptyIndexError,
    LLMAuthError,
    LLMNotConfiguredError,
    LLMUnavailableError,
    translate_embedding_exception,
    translate_llm_exception,
)
from src.schemas.query_filter import QueryMetadataFilter
from src.services.runtime import Runtime
from src.vectordb.filters import build_chroma_where

logger = logging.getLogger(__name__)

NO_EVIDENCE = "No sufficiently relevant evidence was found in the indexed documents."

_SYSTEM = (
    "You answer questions using ONLY the numbered context passages provided. "
    "If the passages do not contain enough information, say clearly that the documents "
    "do not provide enough evidence - do not guess. Cite passages inline like [1] or [2]. "
    "Never invent sources, page numbers or quotations. Answer in the language of the question."
)


@dataclass
class ChatResult:
    answer: str
    status: str  # "answered" | "no_evidence"
    sources: list[dict[str, Any]] = field(default_factory=list)
    extracted_filter: dict[str, Any] = field(default_factory=dict)
    manual_filter: dict[str, Any] = field(default_factory=dict)
    applied_filter: dict[str, Any] = field(default_factory=dict)
    chroma_where: dict[str, Any] | None = None
    filter_status: str = "skipped"  # skipped|none_detected|extracted|failed
    filter_error: str | None = None
    model: str = ""
    timings_ms: dict[str, int] = field(default_factory=dict)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
        return "".join(parts).strip()
    return str(content).strip()


class RagService:
    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    def answer(
        self,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
        k: int | None = None,
        auto_filter: bool = True,
        manual_filter: QueryMetadataFilter | None = None,
    ) -> ChatResult:
        settings = self.runtime.settings
        store = self.runtime.vector_store
        if store.count_chunks() == 0:
            raise EmptyIndexError()
        chat = self.runtime.chat_model()  # raises LLM_NOT_CONFIGURED early
        k = max(1, min(k or settings.search_k_default, settings.search_k_max))
        result = ChatResult(answer="", status="answered", model=self.runtime.llm_status()["model"])
        result.manual_filter = manual_filter.active() if manual_filter else {}

        # 1) metadata filter extraction -------------------------------------------------
        extracted = QueryMetadataFilter()
        t = time.perf_counter()
        if auto_filter:
            try:
                extracted = self.runtime.metadata_extractor.extract_query_filter(question)
                result.filter_status = "extracted" if not extracted.is_empty() else "none_detected"
            except (LLMAuthError, LLMNotConfiguredError):
                raise
            except Exception as exc:
                invalid_output = isinstance(exc, (ValidationError, ValueError, TypeError, KeyError, AttributeError))
                if not invalid_output:
                    # Connection/timeout/rate-limit/provider errors: generation would fail the same way,
                    # so stop now instead of waiting through more timeouts.
                    raise translate_llm_exception(exc) from exc
                result.filter_status = "failed"
                result.filter_error = "The model returned an invalid filter."
                logger.warning("Query filter extraction failed: %s", type(exc).__name__)
                extracted = QueryMetadataFilter()
        result.timings_ms["filter"] = int((time.perf_counter() - t) * 1000)
        result.extracted_filter = extracted.active()
        applied = extracted.merged_with(manual_filter)
        result.applied_filter = applied.active()
        result.chroma_where = build_chroma_where(applied)

        # 2) retrieval ------------------------------------------------------------------
        t = time.perf_counter()
        try:
            hits = store.search_with_scores(question, result.chroma_where, k)
        except AppError:
            raise
        except Exception as exc:
            raise translate_embedding_exception(exc) from exc
        result.timings_ms["retrieval"] = int((time.perf_counter() - t) * 1000)
        if not hits:
            result.status = "no_evidence"
            result.answer = NO_EVIDENCE
            return result

        sources = []
        for n, (doc, score) in enumerate(hits, start=1):
            meta = doc.metadata
            sources.append(
                {
                    "index": n,
                    "filename": meta.get("filename") or meta.get("source"),
                    "title": meta.get("title"),
                    "page": meta.get("page_number"),
                    "similarity": round(score["similarity"], 4),
                    "distance": round(score["distance"], 4),
                    "snippet": doc.page_content[:500],
                    "content": doc.page_content[:3000],
                }
            )
        result.sources = sources

        # 3) generation -----------------------------------------------------------------
        context = "\n\n".join(
            f"[{s['index']}] (file: {s['filename']}"
            + (f", page {s['page']}" if s["page"] is not None else "")
            + f")\n{s['content']}"
            for s in sources
        )
        messages: list[tuple[str, str]] = [("system", _SYSTEM)]
        for turn in (history or [])[-settings.max_chat_history :] if settings.max_chat_history else []:
            messages.append(("human" if turn["role"] == "user" else "ai", turn["content"]))
        messages.append(("human", f"Context passages:\n\n{context}\n\nQuestion: {question}"))
        t = time.perf_counter()
        try:
            reply = chat.invoke(messages)
        except Exception as exc:
            logger.warning("LLM generation failed: %s", type(exc).__name__)
            raise translate_llm_exception(exc) from exc
        result.timings_ms["generation"] = int((time.perf_counter() - t) * 1000)
        text = _message_text(reply)
        if not text:
            raise LLMUnavailableError("The model returned an empty answer.")
        result.answer = text
        return result
