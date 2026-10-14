"""Embedding model construction (device selection, batching, normalisation)."""
from __future__ import annotations

import logging

from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from src.config import get_settings

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> str:
    """Resolve ``auto`` to cuda/mps/cpu depending on what torch reports."""
    requested = (requested or "auto").lower()
    if requested != "auto":
        return requested
    try:
        import torch  # local import: torch is heavy
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def build_embeddings(
    model_name: str | None = None,
    device: str | None = None,
    batch_size: int | None = None,
) -> HuggingFaceEmbeddings:
    settings = get_settings()
    model_name = model_name or settings.embedding_model
    resolved = resolve_device(device or settings.embedding_device)
    logger.info("Loading embedding model %s on %s", model_name, resolved)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": resolved},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": batch_size or settings.embedding_batch_size,
        },
    )
