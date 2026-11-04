"""JobRecord → JobStatus / JobListItem: exactly the contract, nothing internal."""

from qgen.contract import JobListItem, JobStatus
from service.jobs.convert import to_job_status, to_list_item
from tests.jobs.conftest import FAILED_TTL, LLM_ERROR, drive, new_job


def test_queued_job_status(store):
    job = new_job(store, external_ref="gen-9")
    status = to_job_status(store.get(job.job_id))
    assert status.model_dump() == {
        "job_id": job.job_id,
        "external_ref": "gen-9",
        "status": "queued",
        "stage": None,
        "progress": 0.0,
        "error": None,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "retryable_until": None,
    }


def test_failed_job_status_is_valid_contract(store, clock):
    job = drive(store, new_job(store).job_id, "running", "failed")
    status = to_job_status(job)
    data = status.model_dump(mode="json")
    assert JobStatus.model_validate(data) == status  # extra="forbid" – no internal fields
    assert set(data) == set(JobStatus.model_fields)
    assert status.error == LLM_ERROR
    assert status.retryable_until == clock.now + FAILED_TTL
    for internal in ("owner", "options", "attempts", "input_files", "cancel_requested", "finished_at"):
        assert internal not in data


def test_error_is_a_copy(store):
    job = drive(store, new_job(store).job_id, "running", "failed")
    status = to_job_status(job)
    status.error.details["x"] = 1
    assert job.error.details == {}


def test_list_item(store):
    job = new_job(store)
    assert to_list_item(job) == JobListItem(job_id=job.job_id, status="queued", created_at=job.created_at)
