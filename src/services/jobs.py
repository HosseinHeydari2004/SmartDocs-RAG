"""In-memory job registry (single-process). Jobs are lost on restart."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

STAGES: list[tuple[str, str]] = [
    ("receive", "Receiving and validating files"),
    ("extract", "Extracting text / running OCR"),
    ("clean", "Cleaning and preprocessing"),
    ("metadata", "Extracting metadata"),
    ("chunk", "Chunking"),
    ("embed_store", "Generating embeddings and storing in Chroma"),
    ("done", "Finished"),
]
STAGE_LABELS = dict(STAGES)
# Share of one file's work attributed to each stage (progress is the weighted sum
# of stages that really completed plus real intra-stage progress where available).
STAGE_WEIGHTS = {"extract": 0.25, "clean": 0.05, "metadata": 0.10, "chunk": 0.20, "embed_store": 0.40}
_ORDER = list(STAGE_WEIGHTS)
RECEIVE_SHARE = 5  # percent reserved for the receive/validate step

TERMINAL = {"completed", "completed_with_errors", "failed", "cancelled"}


@dataclass
class FileState:
    filename: str
    status: str = "pending"  # pending|processing|success|failed|skipped_duplicate|cancelled
    stage: str | None = None
    chunks: int = 0
    doc_id: str | None = None
    error_code: str | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    fraction: float = 0.0  # 0..1 of this file's pipeline


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"  # queued|running|cancelling + TERMINAL
    stage: str = "receive"
    percent: int = RECEIVE_SHARE
    files: list[FileState] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    chunks_stored: int = 0
    index_total_chunks: int | None = None
    cancel_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stage_label"] = STAGE_LABELS.get(self.stage, self.stage)
        data["files_total"] = len(self.files)
        data["files_succeeded"] = sum(f.status == "success" for f in self.files)
        data["files_failed"] = sum(f.status == "failed" for f in self.files)
        data["files_skipped"] = sum(f.status == "skipped_duplicate" for f in self.files)
        data["documents_stored"] = data["files_succeeded"]
        data.pop("cancel_requested", None)
        for f in data["files"]:
            f.pop("fraction", None)
        return data


def file_fraction(completed_stage: str | None, within: float = 0.0, current: str | None = None) -> float:
    """Progress of one file. ``completed_stage`` = last fully finished stage."""
    done = 0.0
    if completed_stage in _ORDER:
        for name in _ORDER[: _ORDER.index(completed_stage) + 1]:
            done += STAGE_WEIGHTS[name]
    if current in STAGE_WEIGHTS:
        done += STAGE_WEIGHTS[current] * max(0.0, min(1.0, within))
    return min(1.0, done)


class JobStore:
    def __init__(self, max_retained: int = 50) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._max = max_retained

    def create(self, kind: str, filenames: list[str]) -> Job:
        job = Job(id=uuid.uuid4().hex, kind=kind, files=[FileState(filename=n) for n in filenames])
        with self._lock:
            self._jobs[job.id] = job
            self._evict()
        return job

    def _evict(self) -> None:
        finished = sorted((j for j in self._jobs.values() if j.status in TERMINAL), key=lambda j: j.created_at)
        while len(self._jobs) > self._max and finished:
            self._jobs.pop(finished.pop(0).id, None)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def has_active(self) -> bool:
        with self._lock:
            return any(j.status not in TERMINAL for j in self._jobs.values())

    def mutate(self, job_id: str, fn) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                fn(job)

    def recompute_percent(self, job_id: str) -> None:
        def _fn(job: Job) -> None:
            if not job.files:
                job.percent = 100
                return
            mean = sum(f.fraction for f in job.files) / len(job.files)
            job.percent = int(round(RECEIVE_SHARE + (100 - RECEIVE_SHARE) * mean))

        self.mutate(job_id, _fn)

    def request_cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status in TERMINAL:
                return False
            job.cancel_requested = True
            if job.status == "running":
                job.status = "cancelling"
            return True
