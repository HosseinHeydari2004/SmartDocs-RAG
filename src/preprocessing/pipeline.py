from __future__ import annotations

from typing import Callable

from langchain_core.documents import Document

from .chunking.semantic import SemanticChuker
from .cleaner.pipeline import CleanerPipeline
from .parsing.metadata_extraction import MetadataExtractor
from src.schemas.metadata import DocumentMetadata


class PreprocessingPipeline:
    """Clean -> extract metadata -> chunk.

    The three stages are also exposed individually so callers (e.g. the ingestion
    service) can report real progress between them. ``process`` keeps the original
    one-shot behaviour. Heavy components are created lazily.
    """

    def __init__(
        self,
        chunker_factory: Callable[[], SemanticChuker] | None = None,
        metadata_extractor: MetadataExtractor | None = None,
    ):
        self._cleaner = CleanerPipeline()
        self._chunker_factory = chunker_factory or SemanticChuker
        self._semantic_chuker: SemanticChuker | None = None
        self._metadata_extractor = metadata_extractor

    @property
    def metadata_extractor(self) -> MetadataExtractor:
        if self._metadata_extractor is None:
            self._metadata_extractor = MetadataExtractor()
        return self._metadata_extractor

    def clean(self, documents: list[Document]) -> list[Document]:
        return [
            Document(page_content=self._cleaner.clean(text=doc.page_content), metadata=doc.metadata.copy())
            for doc in documents
        ]

    def extract_metadata(self, documents: list[Document]) -> DocumentMetadata:
        return self.metadata_extractor.extract_metadata(document=documents)

    @staticmethod
    def apply_metadata(documents: list[Document], metadata: DocumentMetadata | None) -> list[Document]:
        extra = metadata.model_dump(exclude_none=True) if metadata is not None else {}
        return [
            Document(page_content=doc.page_content, metadata={**doc.metadata, **extra})
            for doc in documents
        ]

    def chunk(self, documents: list[Document]) -> list[Document]:
        if self._semantic_chuker is None:
            self._semantic_chuker = self._chunker_factory()
        return self._semantic_chuker.split_document(document=documents)

    def process(self, document: list[Document]) -> list[Document]:
        cleaned = self.clean(document)
        merged = self.apply_metadata(cleaned, self.extract_metadata(cleaned))
        return self.chunk(merged)
