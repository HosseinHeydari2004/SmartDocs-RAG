from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, Path as PathParam, Query, UploadFile

from src.api.deps import Container, get_container
from src.api.schemas import DeleteResponse, DocumentInfo, DocumentListResponse, JobResponse
from src.errors import (
    AppError,
    BadRequestError,
    ConflictError,
    EmbeddingMismatchError,
    FileValidationError,
    NotFoundError,
    OCRUnavailableError,
)
from src.services.ocr_status import ocr_status
from src.services.uploads import DIGITAL_EXTENSIONS, SCANNED_EXTENSIONS, sanitize_filename, stage_upload

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["documents"])


def _stage_all(files: list[UploadFile], allowed, c: Container):
    if not files:
        raise BadRequestError("No files were provided.", error_code="NO_FILES")
    if len(files) > c.settings.upload_max_files:
        raise BadRequestError(
            f"At most {c.settings.upload_max_files} files per request.", error_code="TOO_MANY_FILES"
        )
    tmp = Path(tempfile.mkdtemp(prefix="smartdocs_"))
    staged, failures, errors = [], [], []
    for up in files:
        try:
            staged.append(stage_upload(up.file, up.filename, tmp, allowed, c.settings.upload_max_bytes))
        except FileValidationError as exc:
            failures.append((sanitize_filename(up.filename), exc.error_code, exc.message))
            errors.append(exc)
    if not staged:
        shutil.rmtree(tmp, ignore_errors=True)
        first = errors[0]
        raise FileValidationError(
            first.message if len(errors) == 1 else "None of the files could be accepted.",
            error_code=first.error_code,
            status_code=first.status_code,
            details=[{"filename": n, "error_code": code, "message": msg} for n, code, msg in failures],
        )
    return tmp, staged, failures


def _submit(kind: str, files, on_duplicate: str, ocr_lang: str, c: Container) -> JobResponse:
    if c.runtime.vector_store.embedding_mismatch:
        raise EmbeddingMismatchError()
    tmp, staged, failures = _stage_all(files, DIGITAL_EXTENSIONS if kind == "digital" else SCANNED_EXTENSIONS, c)
    job_id = c.ingestion.submit(
        kind, staged, tmp_dir=tmp, on_duplicate=on_duplicate, ocr_lang=ocr_lang, precreated_failures=failures
    )
    return JobResponse(**c.jobs.snapshot(job_id))


@router.post("/documents/upload", response_model=JobResponse, status_code=202, summary="Upload digital documents")
def upload_documents(
    files: list[UploadFile] = File(...),
    on_duplicate: Literal["skip", "replace"] = Form("skip"),
    c: Container = Depends(get_container),
) -> JobResponse:
    """Accepts PDF (text), DOCX, TXT, MD and CSV. Processing runs as a background job."""
    return _submit("digital", files, on_duplicate, "eng", c)


@router.post("/documents/upload-scanned", response_model=JobResponse, status_code=202, summary="Upload scanned documents (OCR)")
def upload_scanned(
    files: list[UploadFile] = File(...),
    ocr_lang: str = Form("eng"),
    on_duplicate: Literal["skip", "replace"] = Form("skip"),
    c: Container = Depends(get_container),
) -> JobResponse:
    """Accepts scanned PDFs and images (PNG/JPG/TIFF/BMP/WEBP); text comes from Tesseract OCR."""
    status = ocr_status()
    if not status["available"]:
        raise OCRUnavailableError()
    parts = [p for p in ocr_lang.split("+") if p]
    if not parts or any(p not in status["languages"] for p in parts):
        raise BadRequestError(
            f"OCR language '{ocr_lang}' is not installed. Available: {', '.join(status['languages']) or 'none'}.",
            error_code="OCR_LANG_UNAVAILABLE",
        )
    if not status["pdf_supported"] and any((f.filename or "").lower().endswith(".pdf") for f in files):
        raise OCRUnavailableError("Scanned PDFs need Poppler (pdftoppm), which is not installed.")
    return _submit("scanned", files, on_duplicate, ocr_lang, c)


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(c: Container = Depends(get_container)) -> DocumentListResponse:
    store = c.runtime.vector_store
    docs = store.list_documents()
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total_documents=len(docs),
        total_chunks=store.count_chunks(),
    )


def _ensure_idle(c: Container) -> None:
    if c.jobs.has_active():
        raise ConflictError(
            "A document-processing job is running. Wait for it to finish or cancel it.",
            error_code="INGESTION_IN_PROGRESS",
        )


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(
    doc_id: str = PathParam(pattern=r"^[0-9a-f]{32}$"),
    c: Container = Depends(get_container),
) -> DeleteResponse:
    _ensure_idle(c)
    removed = c.runtime.vector_store.delete_document(doc_id)
    if removed == 0:
        raise NotFoundError("No document with this id.", error_code="DOCUMENT_NOT_FOUND")
    return DeleteResponse(deleted_chunks=removed, deleted_documents=1)


@router.delete("/documents", response_model=DeleteResponse, summary="Clear the whole collection")
def clear_documents(
    confirm: bool = Query(False, description="Must be true to clear the collection."),
    c: Container = Depends(get_container),
) -> DeleteResponse:
    if not confirm:
        raise BadRequestError("Pass confirm=true to clear the entire collection.", error_code="CONFIRMATION_REQUIRED")
    _ensure_idle(c)
    store = c.runtime.vector_store
    before = store.count_chunks()
    n_docs = len(store.list_documents())
    store.reset()
    return DeleteResponse(deleted_chunks=before, deleted_documents=n_docs)
