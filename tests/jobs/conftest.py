"""Fixtures for the job store tests: store in tmp_path, fake clock, helpers."""

from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pytest

from qgen.contract import JobError, JobOptions, OptionDefaults, resolve_options
from service.jobs import JobRecord, JobStore, NewInputFile

T0 = datetime(2026, 11, 4, 10, 0, 0, tzinfo=timezone.utc)
FAILED_TTL = timedelta(hours=24)
PDF = b"%PDF-1.7\n%test\n"


class FakeClock:
    def __init__(self, start: datetime = T0) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> datetime:
        self.now += timedelta(**delta)
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store(tmp_path, clock) -> JobStore:
    return JobStore(tmp_path / "data", failed_input_ttl=FAILED_TTL, clock=clock, lock_timeout_s=10)


def options(external_ref: Optional[str] = "gen-1") -> JobOptions:
    return resolve_options(JobOptions(external_ref=external_ref), OptionDefaults())


def pdf_files(count: int = 2) -> list[NewInputFile]:
    return [NewInputFile(filename=f"doc{i}.pdf", content=PDF + bytes([i]) * 100, pages=i + 3) for i in range(count)]


def new_job(store: JobStore, *, owner: str = "key-a", external_ref: Optional[str] = "gen-1", files: int = 1) -> JobRecord:
    return store.create(owner=owner, options=options(external_ref), files=pdf_files(files))


def set_status(status: str, error: Optional[JobError] = None) -> Callable[[JobRecord], None]:
    def change(record: JobRecord) -> None:
        record.status = status  # type: ignore[assignment]
        if error is not None:
            record.error = error

    return change


LLM_ERROR = JobError(code="llm_error", message="LLM failed.")


def drive(store: JobStore, job_id: str, *statuses: str) -> JobRecord:
    """Move the job through `statuses`; 'failed' gets LLM_ERROR."""
    record = store.get(job_id)
    for status in statuses:
        record = store.update(job_id, set_status(status, LLM_ERROR if status == "failed" else None))
    return record
