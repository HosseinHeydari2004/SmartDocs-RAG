from __future__ import annotations

from fastapi import APIRouter, Depends, Path

from src.api.deps import Container, get_container
from src.api.schemas import JobResponse
from src.errors import ConflictError, NotFoundError
from src.services.jobs import TERMINAL

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])
_ID = Path(pattern=r"^[0-9a-f]{32}$")


@router.get("/{job_id}", response_model=JobResponse, summary="Job status and progress")
def get_job(job_id: str = _ID, c: Container = Depends(get_container)) -> JobResponse:
    snap = c.jobs.snapshot(job_id)
    if snap is None:
        raise NotFoundError("Job not found (jobs are kept in memory only).", error_code="JOB_NOT_FOUND")
    return JobResponse(**snap)


@router.post("/{job_id}/cancel", response_model=JobResponse, summary="Request cancellation")
def cancel_job(job_id: str = _ID, c: Container = Depends(get_container)) -> JobResponse:
    snap = c.jobs.snapshot(job_id)
    if snap is None:
        raise NotFoundError("Job not found.", error_code="JOB_NOT_FOUND")
    if snap["status"] in TERMINAL:
        raise ConflictError("The job has already finished.", error_code="JOB_FINISHED")
    c.jobs.request_cancel(job_id)
    return JobResponse(**c.jobs.snapshot(job_id))
