from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator

from src.schemas.query_filter import QueryMetadataFilter


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: str
    details: Any | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str


class LlmStatus(BaseModel):
    provider: str
    model: str
    key_configured: bool
    key_source: Literal["env", "session", "none"]


class StatsResponse(BaseModel):
    chroma_status: Literal["ok", "error"]
    chroma_error_code: str | None = None
    collection: str
    document_count: int | None = None
    chunk_count: int | None = None
    embedding_model: str
    indexed_embedding_model: str | None = None
    embedding_mismatch: bool = False
    embedding_loaded: bool
    embedding_device_setting: str
    active_jobs: bool


class DocumentInfo(BaseModel):
    doc_id: str | None
    filename: str | None
    title: str | None = None
    author: str | None = None
    year: int | None = None
    document_type: str | None = None
    language: str | None = None
    kind: str | None = None
    ingested_at: str | None = None
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int


class DeleteResponse(BaseModel):
    deleted_chunks: int
    deleted_documents: int | None = None


class FileResult(BaseModel):
    filename: str
    status: str
    stage: str | None = None
    chunks: int = 0
    doc_id: str | None = None
    error_code: str | None = None
    error: str | None = None
    warnings: list[str] = []


class JobResponse(BaseModel):
    id: str
    kind: str
    status: str
    stage: str
    stage_label: str
    percent: int
    files: list[FileResult]
    files_total: int
    files_succeeded: int
    files_failed: int
    files_skipped: int
    documents_stored: int
    chunks_stored: int
    index_total_chunks: int | None = None
    created_at: float
    finished_at: float | None = None


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    k: int | None = Field(default=None, ge=1, le=50)
    auto_filter: bool = True
    manual_filter: QueryMetadataFilter | None = None

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class SourceInfo(BaseModel):
    index: int
    filename: str | None
    title: str | None = None
    page: int | None = None
    similarity: float
    distance: float
    snippet: str
    content: str


class ChatResponse(BaseModel):
    answer: str
    status: Literal["answered", "no_evidence"]
    sources: list[SourceInfo]
    extracted_filter: dict[str, Any]
    manual_filter: dict[str, Any]
    applied_filter: dict[str, Any]
    chroma_where: dict[str, Any] | None
    filter_status: Literal["skipped", "none_detected", "extracted", "failed"]
    filter_error: str | None
    model: str
    timings_ms: dict[str, int]


_MODEL_RE = re.compile(r"^[A-Za-z0-9][\w.\-:/~]{0,99}$")


class LlmSettingsUpdate(BaseModel):
    api_key: SecretStr | None = None
    model: str | None = None

    @field_validator("api_key")
    @classmethod
    def _key(cls, v: SecretStr | None) -> SecretStr | None:
        if v is None:
            return v
        raw = v.get_secret_value()
        if not (8 <= len(raw) <= 512) or any(c.isspace() for c in raw):
            raise ValueError("api_key has an invalid format")
        return v

    @field_validator("model")
    @classmethod
    def _model(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        if not _MODEL_RE.match(v):
            raise ValueError("model has an invalid format")
        return v


class LlmTestResponse(BaseModel):
    ok: bool
    model: str


class OcrStatus(BaseModel):
    engine: str
    available: bool
    version: str | None = None
    languages: list[str]
    pdf_supported: bool
    command: str | None = None
    reason: str | None = None


class SettingsStatusResponse(BaseModel):
    llm: LlmStatus
    ocr: OcrStatus
    embedding: dict[str, Any]
    limits: dict[str, Any]
    search: dict[str, Any]
