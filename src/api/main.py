"""FastAPI application. Run: ``uvicorn src.api.main:app --reload`` (single worker)."""
from __future__ import annotations

import logging
import re
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.deps import Container, build_container
from src.api.routers import chat, documents, health, jobs, settings
from src.errors import AppError
from src.logging_config import configure_logging, request_id_var

logger = logging.getLogger(__name__)
_RID = re.compile(r"^[\w\-]{1,64}$")


def _error(request: Request, status: int, code: str, message: str, details=None) -> JSONResponse:
    body = {"error_code": code, "message": message, "request_id": getattr(request.state, "request_id", "-")}
    if details is not None:
        body["details"] = details
    return JSONResponse(status_code=status, content=body)


def create_app(container: Container | None = None) -> FastAPI:
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = container or build_container()
        if app.state.container.settings.embedding_preload:
            app.state.container.runtime.preload_embeddings()
        yield
        app.state.container.ingestion.shutdown()

    app = FastAPI(
        title="SmartDocs-RAG API",
        version=health.API_VERSION,
        description="Document ingestion (digital + OCR), metadata-aware RAG chat and settings.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if _RID.match(incoming) else uuid.uuid4().hex[:12]
        request.state.request_id = rid
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        if exc.status_code >= 500:
            logger.error("AppError %s: %s", exc.error_code, exc.message)
        return _error(request, exc.status_code, exc.error_code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        # Only location + message: never echo submitted values (could contain secrets).
        details = [{"loc": [str(p) for p in e.get("loc", [])], "message": e.get("msg", "")} for e in exc.errors()]
        return _error(request, 422, "VALIDATION_ERROR", "The request is not valid.", details)

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException):
        return _error(request, exc.status_code, f"HTTP_{exc.status_code}", str(exc.detail))

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return _error(request, 500, "INTERNAL_ERROR", "Internal server error.")

    for module in (health, documents, jobs, chat, settings):
        app.include_router(module.router)
    return app


app = create_app()
