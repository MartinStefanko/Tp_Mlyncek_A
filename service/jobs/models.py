"""JobRecord: the content of /data/jobs/{job_id}/job.json.

Internal model of the service; the API returns only its JobStatus view (service.jobs.convert).
"""

from typing import Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from qgen.contract import JobError, JobOptions, JobStage, JobStatusValue

SCHEMA_VERSION = 1

ACTIVE_STATUSES: frozenset[str] = frozenset({"queued", "running"})
FINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})


class _RecordModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=False)


class InputFile(_RecordModel):
    """One uploaded PDF. On disk it is input/{stored_as}; the client name never becomes a path."""

    name: str = Field(min_length=1, description="File name from the upload.")
    stored_as: str = Field(pattern=r"^D\d+\.pdf$", description="File name in input/, e.g. 'D1.pdf'.")
    size: int = Field(ge=0, description="Bytes.")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pages: int = Field(ge=0)


class JobRecord(_RecordModel):
    schema_version: int = SCHEMA_VERSION
    job_id: str = Field(min_length=1)
    owner: str = Field(min_length=1, description="Id of the API key (never the key itself).")
    external_ref: Optional[str] = None
    status: JobStatusValue
    stage: Optional[JobStage] = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    error: Optional[JobError] = None
    options: JobOptions = Field(description="Complete options (after resolve_options).")
    input_files: list[InputFile]
    attempts: int = Field(default=0, ge=0, description="Runs started by a worker since creation or the last retry.")
    created_at: AwareDatetime
    updated_at: AwareDatetime
    finished_at: Optional[AwareDatetime] = None
    retryable_until: Optional[AwareDatetime] = None
    cancel_requested: bool = False

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    @property
    def is_final(self) -> bool:
        return self.status in FINAL_STATUSES
