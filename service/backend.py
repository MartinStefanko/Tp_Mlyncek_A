"""Interface between the HTTP routes and the job implementation.

Routes validate the request (API key, options, files) and then call a JobBackend.
Now there is only MockBackend (in-memory simulation, service/mock/); the real
backend (job.json in /data, RQ queue) will implement the same interface, so the
routes do not change.

Backend methods raise service.errors.ApiError for API errors (404, 409, 429, ...).
`owner` identifies the API key; external_ref is unique per owner and a job of
another owner behaves as if it did not exist.
"""

from dataclasses import dataclass
from typing import Optional, Protocol

from qgen.contract import JobListItem, JobOptions, JobStatus, Report, UnitsResult


@dataclass(frozen=True)
class UploadedFile:
    """A checked PDF from the request."""

    filename: str
    content: bytes


@dataclass(frozen=True)
class CreateJobRequest:
    owner: str
    options: JobOptions  # complete options (after resolve_options)
    files: list[UploadedFile]


@dataclass(frozen=True)
class CreateJobResult:
    status: JobStatus
    replayed: bool  # True: external_ref already existed, no new job


class JobBackend(Protocol):
    async def start(self) -> None:
        """Called once when the API starts."""

    async def stop(self) -> None:
        """Called once when the API stops."""

    def health(self) -> dict[str, str]:
        """State of the components, e.g. {"api": "ok", "redis": "ok", "worker": "ok"}."""

    def create(self, request: CreateJobRequest) -> CreateJobResult:
        """New job, or the existing one for the same (owner, external_ref). 429 too_many_jobs."""

    def get(self, job_id: str, owner: str) -> JobStatus:
        """404 job_not_found."""

    def list_by_external_ref(self, external_ref: str, owner: str) -> list[JobListItem]:
        """Jobs with this external_ref, newest first (may be empty)."""

    def result(self, job_id: str, owner: str) -> UnitsResult:
        """404 job_not_found; 409 result_not_ready (queued/running), job_not_completed (failed/cancelled)."""

    def report(self, job_id: str, owner: str) -> Report:
        """Same errors as result()."""

    def cancel_or_delete(self, job_id: str, owner: str) -> Optional[JobStatus]:
        """queued/running → cancelled, returns the status; final state → deleted, returns None."""

    def retry(self, job_id: str, owner: str) -> JobStatus:
        """failed → queued with the same job_id. 409 job_not_retryable, 429 too_many_jobs."""
