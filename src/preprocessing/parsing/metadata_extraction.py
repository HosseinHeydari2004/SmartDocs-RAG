from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

from langchain_core.documents import Document

from src.schemas.metadata import DocumentMetadata
from src.schemas.query_filter import QueryMetadataFilter

logger = logging.getLogger(__name__)

_DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def _default_chat_model() -> Any:
    """Fallback provider used when no runtime is injected (env key only)."""
    from src.config import get_settings
    from src.errors import LLMNotConfiguredError
    from src.llm import build_chat_model

    settings = get_settings()
    if not settings.openrouter_api_key:
        raise LLMNotConfiguredError()
    return build_chat_model(
        settings.llm_model, settings.openrouter_api_key, timeout=settings.llm_timeout_s,
        base_url=settings.openrouter_base_url,
    )


class MetadataExtractor:
    """LLM-based extraction of document metadata and of query metadata filters.

    The chat model is resolved lazily on every call through ``chat_model_provider``,
    so constructing the extractor needs no API key and a key changed at runtime
    is picked up immediately.
    """

    def __init__(self, chat_model_provider: Callable[[], Any] | None = None) -> None:
        self._provider = chat_model_provider or _default_chat_model

    def _structured(self, schema: type):
        return self._provider().with_structured_output(schema)

    def extract_metadata(self, document: list[Document]) -> DocumentMetadata:
        full_text = "\n".join(docs.page_content for docs in document[:6])
        result = self._structured(DocumentMetadata).invoke(
            f"""
            Extract metadata from the following document.
            Return:
            - title
            - author (null if not stated)
            - year (integer, null if not stated)
            - document_type (research_paper, book, report, documentation, article or other)
            - language (lowercase ISO 639-1 code such as en or fa)

            Document:

            {full_text[:6000]}
            """
        )
        if not isinstance(result, DocumentMetadata):
            raise ValueError("The LLM did not return valid document metadata.")
        return result

    def extract_query_filter(self, query: str) -> QueryMetadataFilter:
        """Extract only the metadata constraints the user explicitly stated."""
        normalized = query.translate(_DIGIT_MAP)
        this_year = datetime.now(timezone.utc).year
        result = self._structured(QueryMetadataFilter).invoke(
            f"""
            You extract metadata filters from a search question about a document library.
            Fill a field ONLY if the user explicitly states that constraint. Otherwise
            leave it null. Never guess and never infer a value from the topic of the question.

            Fields:
            - filename: an exact file name the user wants to search in (e.g. report.pdf).
            - author: an author name explicitly mentioned as the writer of the documents.
            - document_type: research_paper, book, report, documentation, article or other
              (only if the user names the kind of document).
            - language: lowercase ISO 639-1 code of the documents' language (e.g. en, fa),
              only if the user states it.
            - year: one exact publication year.
            - year_gte / year_lte: "after 2020" -> year_gte 2021, "since 2020" -> year_gte 2020,
              "before 2015" -> year_lte 2014, "until 2015" -> year_lte 2015.
            - years: several specific years.

            The current year is {this_year}; use it only for relative ranges such as
            "in the last 3 years". Convert Persian/Arabic digits to ASCII digits.
            If the question has no such constraints, return every field as null.

            User question:
            {normalized}
            """
        )
        if not isinstance(result, QueryMetadataFilter):
            raise ValueError("The LLM did not return a valid metadata filter.")
        return result
