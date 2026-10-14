from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from src.config import Settings, get_settings
from src.services.ingestion import IngestionService
from src.services.jobs import JobStore
from src.services.rag import RagService
from src.services.runtime import Runtime


@dataclass
class Container:
    settings: Settings
    runtime: Runtime
    jobs: JobStore
    ingestion: IngestionService
    rag: RagService


def build_container(settings: Settings | None = None, runtime: Runtime | None = None) -> Container:
    settings = settings or get_settings()
    runtime = runtime or Runtime(settings)
    jobs = JobStore(settings.max_retained_jobs)
    return Container(
        settings=settings,
        runtime=runtime,
        jobs=jobs,
        ingestion=IngestionService(runtime, jobs),
        rag=RagService(runtime),
    )


def get_container(request: Request) -> Container:
    return request.app.state.container
