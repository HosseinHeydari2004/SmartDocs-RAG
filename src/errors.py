"""Application errors with a stable ``error_code`` and HTTP status."""
from __future__ import annotations

from typing import Any


class AppError(Exception):
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"
    default_message: str = "Internal server error."

    def __init__(
        self,
        message: str | None = None,
        *,
        error_code: str | None = None,
        status_code: int | None = None,
        details: Any = None,
    ) -> None:
        self.message = message or self.default_message
        if error_code:
            self.error_code = error_code
        if status_code:
            self.status_code = status_code
        self.details = details
        super().__init__(self.message)


class BadRequestError(AppError):
    status_code = 400
    error_code = "BAD_REQUEST"
    default_message = "Bad request."


class FileValidationError(AppError):
    status_code = 400
    error_code = "INVALID_FILE"
    default_message = "The uploaded file is not valid."


class NotFoundError(AppError):
    status_code = 404
    error_code = "NOT_FOUND"
    default_message = "Resource not found."


class ConflictError(AppError):
    status_code = 409
    error_code = "CONFLICT"
    default_message = "The request conflicts with the current state."


class EmptyIndexError(ConflictError):
    error_code = "INDEX_EMPTY"
    default_message = "The vector index is empty. Upload documents first."


class EmbeddingMismatchError(ConflictError):
    error_code = "EMBEDDING_MODEL_MISMATCH"
    default_message = (
        "The collection was built with a different embedding model. "
        "Clear the collection and re-index, or restore the original EMBEDDING_MODEL."
    )


class EmbeddingModelError(AppError):
    status_code = 503
    error_code = "EMBEDDING_MODEL_ERROR"
    default_message = "The embedding model could not be loaded or run."


class VectorStoreError(AppError):
    status_code = 500
    error_code = "VECTOR_STORE_ERROR"
    default_message = "The vector database reported an error."


class OCRUnavailableError(AppError):
    status_code = 503
    error_code = "OCR_UNAVAILABLE"
    default_message = "OCR engine (Tesseract) is not available on the server."


class LLMNotConfiguredError(AppError):
    status_code = 503
    error_code = "LLM_NOT_CONFIGURED"
    default_message = "No OpenRouter API key is configured."


class LLMAuthError(AppError):
    status_code = 401
    error_code = "LLM_AUTH_FAILED"
    default_message = "The OpenRouter API key was rejected."


class LLMRateLimitError(AppError):
    status_code = 429
    error_code = "LLM_RATE_LIMITED"
    default_message = "The LLM provider rate limit was reached. Try again shortly."


class LLMConnectionError(AppError):
    status_code = 502
    error_code = "LLM_CONNECTION_FAILED"
    default_message = (
        "Could not connect to OpenRouter. Check the internet connection, firewall/proxy, VPN or regional blocking."
    )


class LLMTimeoutError(AppError):
    status_code = 504
    error_code = "LLM_TIMEOUT"
    default_message = "The LLM provider did not respond in time."


class LLMUnavailableError(AppError):
    status_code = 502
    error_code = "LLM_UNAVAILABLE"
    default_message = "The LLM provider returned an error."


def _status_of(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def translate_llm_exception(exc: BaseException) -> AppError:
    """Map provider/client exceptions to an ``AppError`` without leaking details."""
    if isinstance(exc, AppError):
        return exc
    status = _status_of(exc)
    name = type(exc).__name__.lower()
    chain, cur = [], exc
    while cur is not None and len(chain) < 5:  # also look at wrapped causes
        chain.append(type(cur).__name__.lower())
        cur = cur.__cause__ or cur.__context__
    if any("connect" in n for n in chain) or isinstance(exc, ConnectionError):
        return LLMConnectionError()
    if status in (401, 403) or "authentication" in name:
        return LLMAuthError()
    if status == 429 or "ratelimit" in name:
        return LLMRateLimitError()
    if "timeout" in name or status in (408, 504):
        return LLMTimeoutError()
    return LLMUnavailableError()


def translate_embedding_exception(exc: BaseException) -> AppError:
    if isinstance(exc, AppError):
        return exc
    text = str(exc).lower()
    if "out of memory" in text:
        return EmbeddingModelError(
            "Embedding ran out of memory. Lower EMBEDDING_BATCH_SIZE or set EMBEDDING_DEVICE=cpu.",
            error_code="EMBEDDING_OOM",
        )
    return EmbeddingModelError()
