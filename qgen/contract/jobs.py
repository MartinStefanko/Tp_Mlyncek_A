"""Job state, webhook body and error responses (Príloha A)."""

from typing import Any, Literal, Optional, get_args

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

JobStatusValue = Literal["queued", "running", "completed", "failed", "cancelled"]

# Processing steps (kap. 4).
JobStage = Literal[
    "extract",
    "language",
    "learning_objects",
    "lo_evaluation",
    "questions",
    "q_evaluation",
    "split",
    "output",
]

WebhookEvent = Literal["job.stage_changed", "job.completed", "job.failed", "job.cancelled"]

# HTTP error codes of the API (Príloha A – Chybové odpovede).
# llm_error is NOT here: it is a job error code (status failed), see JobErrorCode.
# not_found, method_not_allowed and internal_error are not in the table of Príloha A;
# they cover an unknown path, a wrong HTTP method and an unexpected server error,
# so that every error response keeps the uniform shape.
ApiErrorCode = Literal[
    "invalid_options",
    "unauthorized",
    "job_not_found",
    "result_not_ready",
    "job_not_completed",
    "job_not_retryable",
    "file_too_large",
    "unsupported_file",
    "validation_error",
    "too_many_jobs",
    "not_found",
    "method_not_allowed",
    "internal_error",
]
API_ERROR_CODES: tuple[str, ...] = get_args(ApiErrorCode)

# Error codes of a failed job (status failed, field error.code); not HTTP responses.
JobErrorCode = Literal["llm_error", "worker_lost", "extraction_failed", "internal_error"]
JOB_ERROR_CODES: tuple[str, ...] = get_args(JobErrorCode)


class _JobModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JobError(_JobModel):
    """Error of a failed job (e.g. code 'llm_error')."""

    code: JobErrorCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class JobStatus(_JobModel):
    """GET /v1/jobs/{id}."""

    job_id: str = Field(min_length=1)
    external_ref: Optional[str] = None
    status: JobStatusValue
    stage: Optional[JobStage] = None
    progress: float = Field(ge=0.0, le=1.0)
    error: Optional[JobError] = None
    created_at: AwareDatetime
    updated_at: AwareDatetime
    retryable_until: Optional[AwareDatetime] = None


class WebhookPayload(_JobModel):
    """Body of the webhook POST to callback_url. Carries only the state, not the result."""

    event: WebhookEvent
    job_id: str = Field(min_length=1)
    external_ref: Optional[str] = None
    status: JobStatusValue
    stage: Optional[JobStage] = None
    progress: float = Field(ge=0.0, le=1.0)
    error: Optional[JobError] = None
    occurred_at: AwareDatetime


class JobLinks(_JobModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    self_: str = Field(alias="self", description="Path of the job, e.g. '/v1/jobs/7f3c2a1e'.")


class CreateJobResponse(_JobModel):
    """POST /v1/jobs: 202 for a new job, 200 with replayed=true for an existing external_ref."""

    job_id: str = Field(min_length=1)
    status: JobStatusValue
    replayed: bool = False
    links: JobLinks


class JobListItem(_JobModel):
    job_id: str = Field(min_length=1)
    status: JobStatusValue
    created_at: AwareDatetime


class JobListResponse(_JobModel):
    """GET /v1/jobs?external_ref=…, newest first."""

    jobs: list[JobListItem]


class ErrorBody(_JobModel):
    code: ApiErrorCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(_JobModel):
    """Uniform API error: {"error": {"code": ..., "message": ..., "details": {...}}}."""

    error: ErrorBody
