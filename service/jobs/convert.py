"""JobRecord → API contract models. Internal fields (owner, options, attempts, …) stay inside."""

from qgen.contract import JobListItem, JobStatus
from service.jobs.models import JobRecord


def to_job_status(record: JobRecord) -> JobStatus:
    return JobStatus(
        job_id=record.job_id,
        external_ref=record.external_ref,
        status=record.status,
        stage=record.stage,
        progress=record.progress,
        error=record.error.model_copy(deep=True) if record.error else None,
        created_at=record.created_at,
        updated_at=record.updated_at,
        retryable_until=record.retryable_until,
    )


def to_list_item(record: JobRecord) -> JobListItem:
    return JobListItem(job_id=record.job_id, status=record.status, created_at=record.created_at)
