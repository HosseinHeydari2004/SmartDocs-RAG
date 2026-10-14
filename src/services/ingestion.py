"""Document ingestion: staged files -> text -> clean -> metadata -> chunks -> Chroma.

Progress comes from the real pipeline (stage completion, OCR pages, embedding batches).
Files are processed one at a time; a failure in one file never stops the others.
"""
from __future__ import annotations

import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from langchain_core.documents import Document

from src.errors import AppError, translate_embedding_exception
from src.loaders.ocr.ocr_loader import OcrError, OcrLoader
from src.loaders.pipeline.loader_pipeline import DocumentLoaderPipeline
from src.services.jobs import JobStore, file_fraction
from src.services.runtime import Runtime
from src.services.uploads import StagedFile

logger = logging.getLogger(__name__)

# Only these keys are kept on chunks (scalar, predictable, no server paths).
_KEEP = {"page_number", "total_pages", "row", "extraction"}


class _Cancelled(Exception):
    pass


def normalize_metadata(docs: list[Document], staged: StagedFile, kind: str) -> list[Document]:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: list[Document] = []
    for doc in docs:
        raw = dict(doc.metadata)
        meta: dict[str, Any] = {k: v for k, v in raw.items() if k in _KEEP and isinstance(v, (str, int, float, bool))}
        if "page_number" not in meta and isinstance(raw.get("page"), int):
            meta["page_number"] = raw["page"] + 1  # PyMuPDF pages are 0-based
        meta.update(
            filename=staged.original_name,
            source=staged.original_name,
            doc_id=staged.doc_id,
            kind=kind,
            ingested_at=now,
        )
        out.append(Document(page_content=doc.page_content, metadata=meta))
    return out


class IngestionService:
    def __init__(self, runtime: Runtime, jobs: JobStore) -> None:
        self.runtime = runtime
        self.jobs = jobs
        # One worker: serialises embedding/Chroma writes and bounds memory.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ API
    def submit(
        self,
        kind: str,
        staged: list[StagedFile],
        *,
        tmp_dir,
        on_duplicate: str = "skip",
        ocr_lang: str = "eng",
        precreated_failures: list[tuple[str, str, str]] | None = None,
    ) -> str:
        failures = precreated_failures or []
        names = [f[0] for f in failures] + [s.original_name for s in staged]
        job = self.jobs.create(kind, names)
        for idx, (_, code, message) in enumerate(failures):
            self._set_file(job.id, idx, status="failed", error_code=code, error=message, fraction=1.0)
        offset = len(failures)
        self.jobs.recompute_percent(job.id)
        self._executor.submit(self._run, job.id, staged, offset, kind, tmp_dir, on_duplicate, ocr_lang)
        return job.id

    # ------------------------------------------------------------------ helpers
    def _set_file(self, job_id: str, idx: int, **fields: Any) -> None:
        def _fn(job):
            state = job.files[idx]
            for k, v in fields.items():
                setattr(state, k, v)

        self.jobs.mutate(job_id, _fn)

    def _stage(self, job_id: str, idx: int, stage: str, completed: str | None, within: float = 0.0) -> None:
        def _fn(job):
            job.stage = stage
            f = job.files[idx]
            f.stage = stage
            f.fraction = file_fraction(completed, within, current=stage)

        self.jobs.mutate(job_id, _fn)
        self.jobs.recompute_percent(job_id)

    def _cancelled(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        return bool(job and job.cancel_requested)

    # ------------------------------------------------------------------ worker
    def _run(self, job_id, staged, offset, kind, tmp_dir, on_duplicate, ocr_lang) -> None:
        self.jobs.mutate(job_id, lambda j: setattr(j, "status", "running") if not j.cancel_requested else None)
        try:
            seen: set[str] = set()
            for i, sf in enumerate(staged):
                idx = offset + i
                if self._cancelled(job_id):
                    self._set_file(job_id, idx, status="cancelled", fraction=1.0)
                    continue
                try:
                    self._process_file(job_id, idx, sf, kind, on_duplicate, ocr_lang, seen)
                except _Cancelled:
                    self._set_file(job_id, idx, status="cancelled", fraction=1.0)
                except AppError as exc:
                    logger.warning("File %s failed: %s", sf.original_name, exc.error_code)
                    self._set_file(job_id, idx, status="failed", error_code=exc.error_code, error=exc.message, fraction=1.0)
                except Exception:
                    logger.exception("Unexpected error while processing %s", sf.original_name)
                    self._set_file(
                        job_id, idx, status="failed", error_code="INTERNAL_ERROR",
                        error="Unexpected error while processing this file.", fraction=1.0,
                    )
                self.jobs.recompute_percent(job_id)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            self._finish(job_id)

    def _finish(self, job_id: str) -> None:
        try:
            total = self.runtime.vector_store.count_chunks()
        except Exception:
            total = None

        def _fn(job):
            states = [f.status for f in job.files]
            job.chunks_stored = sum(f.chunks for f in job.files if f.status == "success")
            job.index_total_chunks = total
            if job.cancel_requested:
                job.status = "cancelled"
            elif states and all(s in ("failed", "cancelled") for s in states):
                job.status = "failed"
            elif any(s == "failed" for s in states):
                job.status = "completed_with_errors"
            else:
                job.status = "completed"
            job.stage = "done"
            job.percent = 100
            job.finished_at = __import__("time").time()

        self.jobs.mutate(job_id, _fn)

    def _check_cancel(self, job_id: str) -> None:
        if self._cancelled(job_id):
            raise _Cancelled

    def _process_file(self, job_id, idx, sf: StagedFile, kind, on_duplicate, ocr_lang, seen) -> None:
        store = self.runtime.vector_store
        settings = self.runtime.settings
        self._set_file(job_id, idx, status="processing", doc_id=sf.doc_id)

        # --- duplicate policy -------------------------------------------------
        if sf.doc_id in seen or store.has_document(sf.doc_id):
            if sf.doc_id in seen or on_duplicate != "replace":
                self._set_file(
                    job_id, idx, status="skipped_duplicate", fraction=1.0, error_code="DUPLICATE",
                    error="This exact file is already indexed (skipped).",
                )
                return
            store.delete_document(sf.doc_id)
        seen.add(sf.doc_id)

        # --- extract ----------------------------------------------------------
        self._stage(job_id, idx, "extract", None, 0.0)
        warnings: list[str] = []
        if kind == "scanned":
            loader = OcrLoader()

            def on_page(page: int, total: int) -> None:
                self._stage(job_id, idx, "extract", None, page / max(total, 1))
                self._check_cancel(job_id)

            try:
                docs = loader.load(sf.path, lang=ocr_lang, dpi=settings.ocr_dpi,
                                   max_pages=settings.ocr_max_pages, on_page=on_page)
            except OcrError as exc:
                raise AppError(str(exc), error_code="OCR_FAILED", status_code=422) from exc
            rep = loader.last_report
            if rep.pages_failed:
                warnings.append(f"OCR failed on page(s): {', '.join(map(str, rep.pages_failed))}.")
            if rep.pages_empty:
                warnings.append(f"No text recognised on page(s): {', '.join(map(str, rep.pages_empty))}.")
            if not docs:
                raise AppError("OCR recognised no text in this file.", error_code="OCR_NO_TEXT", status_code=422)
        else:
            try:
                docs = DocumentLoaderPipeline().load(sf.path)
            except Exception as exc:
                logger.warning("Loader failed for %s: %s", sf.original_name, type(exc).__name__)
                raise AppError(
                    "The file could not be read (it may be corrupt, encrypted or unsupported).",
                    error_code="EXTRACTION_FAILED", status_code=422,
                ) from exc
            docs = [d for d in docs if d.page_content and d.page_content.strip()]
            if not docs:
                hint = " If this is a scanned PDF, use the scanned-documents upload." if sf.path.suffix.lower() == ".pdf" else ""
                raise AppError("No extractable text was found." + hint, error_code="NO_TEXT_FOUND", status_code=422)
        docs = normalize_metadata(docs, sf, kind)
        self._check_cancel(job_id)

        # --- clean ------------------------------------------------------------
        pre = self.runtime.preprocessing
        self._stage(job_id, idx, "clean", "extract")
        docs = pre.clean(docs)
        docs = [d for d in docs if d.page_content.strip()]
        if not docs:
            raise AppError("No text remained after cleaning.", error_code="NO_TEXT_FOUND", status_code=422)
        self._check_cancel(job_id)

        # --- metadata (LLM; failure is reported, never faked) -------------------
        self._stage(job_id, idx, "metadata", "clean")
        try:
            meta = pre.extract_metadata(docs)
            docs = pre.apply_metadata(docs, meta)
        except Exception as exc:
            app = exc if isinstance(exc, AppError) else None
            code = app.error_code if app else "METADATA_EXTRACTION_FAILED"
            logger.warning("Metadata extraction failed for %s (%s)", sf.original_name, code)
            warnings.append(f"Metadata extraction failed ({code}); indexed without title/author/year/type/language.")
        self._check_cancel(job_id)

        # --- chunk ------------------------------------------------------------
        self._stage(job_id, idx, "chunk", "metadata")
        try:
            chunks = pre.chunk(docs)
        except Exception as exc:
            logger.exception("Chunking failed for %s", sf.original_name)
            raise translate_embedding_exception(exc) from exc
        chunks = [c for c in chunks if c.page_content.strip()]
        if not chunks:
            raise AppError("Chunking produced no chunks.", error_code="NO_TEXT_FOUND", status_code=422)
        self._check_cancel(job_id)

        # --- embed + store ----------------------------------------------------
        self._stage(job_id, idx, "embed_store", "chunk", 0.0)
        ids = [f"{sf.doc_id}:{n}" for n in range(len(chunks))]

        def on_batch(done: int, total: int) -> None:
            self._stage(job_id, idx, "embed_store", "chunk", done / total)

        try:
            stored = store.add_document(
                chunks, ids, batch_size=settings.chroma_add_batch_size, on_batch=on_batch,
                should_cancel=lambda: self._cancelled(job_id),
            )
        except Exception as exc:
            store.delete_document(sf.doc_id)  # roll back a partial write
            raise (exc if isinstance(exc, AppError) else translate_embedding_exception(exc)) from exc
        if stored < len(chunks):  # cancelled mid-way
            store.delete_document(sf.doc_id)
            raise _Cancelled

        self._set_file(job_id, idx, status="success", chunks=stored, warnings=warnings, fraction=1.0, stage="done")
