from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from src.embeddings import build_embeddings


class ChunkingEmbedder:
    """Embedding model used by the semantic chunker.

    Defaults come from ``Settings`` (``EMBEDDING_MODEL`` / ``EMBEDDING_DEVICE``).
    Prefer injecting a shared model instead of building a second copy.
    """

    def __init__(self, embedding_model_name: str | None = None, device: str | None = None):
        self.model: HuggingFaceEmbeddings = build_embeddings(embedding_model_name, device)
