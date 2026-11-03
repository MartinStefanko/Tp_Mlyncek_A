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
# llm_error is NOT here: it is a job error code (status failed), see JobError.
ApiErrorCode = Literal[
    "invalid_options",
    "unauthorized",
    "job_not_found",
    "result_not_ready",
    "job_not_retryable",
    "file_too_large",
    "unsupported_file",
    "validation_error",
    "too_many_jobs",
]
API_ERROR_CODES: tuple[str, ...] = get_args(ApiErrorCode)


class _JobModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JobError(_JobModel):
    """Error of a failed job (e.g. code 'llm_error')."""

    code: str = Field(min_length=1)
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


class ErrorBody(_JobModel):
    code: ApiErrorCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(_JobModel):
    """Uniform API error: {"error": {"code": ..., "message": ..., "details": {...}}}."""

    error: ErrorBody
