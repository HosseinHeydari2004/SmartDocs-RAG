from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import Container, get_container
from src.api.schemas import HealthResponse, LlmStatus, OcrStatus, SettingsStatusResponse, StatsResponse
from src.errors import AppError
from src.services.ocr_status import ocr_status
from src.services.uploads import DIGITAL_EXTENSIONS, SCANNED_EXTENSIONS

router = APIRouter()
API_VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(version=API_VERSION)


@router.get("/api/v1/stats", response_model=StatsResponse, tags=["system"])
def stats(c: Container = Depends(get_container)) -> StatsResponse:
    s = c.settings
    base = dict(
        collection=s.collection_name,
        embedding_model=s.embedding_model,
        embedding_loaded=c.runtime.embeddings.loaded,
        embedding_device_setting=s.embedding_device,
        active_jobs=c.jobs.has_active(),
    )
    try:
        store = c.runtime.vector_store
        docs = store.list_documents()
        chunks = store.count_chunks()
    except AppError as exc:
        return StatsResponse(chroma_status="error", chroma_error_code=exc.error_code, **base)
    return StatsResponse(
        chroma_status="ok",
        document_count=len(docs),
        chunk_count=chunks,
        indexed_embedding_model=store.indexed_model,
        embedding_mismatch=store.embedding_mismatch,
        **base,
    )


@router.get("/api/v1/settings/status", response_model=SettingsStatusResponse, tags=["settings"])
def settings_status(c: Container = Depends(get_container)) -> SettingsStatusResponse:
    s = c.settings
    return SettingsStatusResponse(
        llm=LlmStatus(**c.runtime.llm_status()),
        ocr=OcrStatus(**ocr_status()),
        embedding={"model": s.embedding_model, "device_setting": s.embedding_device, "batch_size": s.embedding_batch_size},
        limits={
            "upload_max_mb": s.upload_max_mb,
            "upload_max_files": s.upload_max_files,
            "digital_extensions": sorted(DIGITAL_EXTENSIONS),
            "scanned_extensions": sorted(SCANNED_EXTENSIONS),
            "ocr_max_pages": s.ocr_max_pages,
        },
        search={"k_default": s.search_k_default, "k_max": s.search_k_max},
    )
