from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker


class SemanticChuker:
    def __init__(self, embeddings: Embeddings | None = None):
        if embeddings is None:
            from .embedding_chunker.embedding import ChunkingEmbedder

            embeddings = ChunkingEmbedder().model
        self.semantic_chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_amount=90,
            breakpoint_threshold_type="percentile",
            min_chunk_size=400,
        )

    def split_document(self, document: list[Document]) -> list[Document]:
        return self.semantic_chunker.split_documents(document)
