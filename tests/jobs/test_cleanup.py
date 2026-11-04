"""Cleanup by time: PDFs after the final state, whole job after the result TTL, leftovers."""

import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from service.jobs import JobNotFound
from service.jobs.cleanup import run_cleanup, run_cleanup_from_settings
from tests.jobs.conftest import FAILED_TTL, drive, new_job

RESULT_TTL = timedelta(days=7)


def cleanup(store, now):
    return run_cleanup(store, now, result_ttl=RESULT_TTL)


@pytest.mark.parametrize("path", [["running", "completed"], ["cancelled"], ["running", "cancelled"]])
def test_inputs_of_completed_or_cancelled_job_go_at_once(store, clock, path):
    job = drive(store, new_job(store).job_id, *path)
    result = cleanup(store, clock.now)
    assert result.inputs_deleted == [job.job_id]
    assert not store.has_inputs(job.job_id)
    assert store.get(job.job_id).status == path[-1]  # the record stays


def test_inputs_of_failed_job_stay_until_retryable_until(store, clock):
    job = drive(store, new_job(store).job_id, "running", "failed")

    assert cleanup(store, clock.now + FAILED_TTL - timedelta(seconds=1)).inputs_deleted == []
    assert store.has_inputs(job.job_id)

    result = cleanup(store, clock.now + FAILED_TTL)
    assert result.inputs_deleted == [job.job_id]
    assert not store.has_inputs(job.job_id)
    record = store.get(job.job_id)
    assert record.status == "failed" and record.retryable_until == clock.now + FAILED_TTL


@pytest.mark.parametrize("path", [["running", "completed"], ["running", "failed"], ["cancelled"]])
def test_whole_job_goes_after_result_ttl(store, clock, path):
    job = drive(store, new_job(store).job_id, *path)
    finished = store.get(job.job_id).finished_at

    assert cleanup(store, finished + RESULT_TTL - timedelta(seconds=1)).jobs_deleted == []
    result = cleanup(store, finished + RESULT_TTL)
    assert result.jobs_deleted == [job.job_id]
    with pytest.raises(JobNotFound):
        store.get(job.job_id)


@pytest.mark.parametrize("path", [[], ["running"]])
def test_active_jobs_are_never_deleted(store, clock, path):
    job = drive(store, new_job(store).job_id, *path)
    result = cleanup(store, clock.now + timedelta(days=365))
    assert result.inputs_deleted == [] and result.jobs_deleted == []
    assert store.has_inputs(job.job_id)


def test_ttl_counts_from_finished_at_not_created_at(store, clock):
    job = drive(store, new_job(store).job_id, "running")
    clock.advance(days=6)
    drive(store, job.job_id, "completed")
    assert cleanup(store, clock.now + timedelta(days=2)).jobs_deleted == []


def _age(path, hours: float) -> None:
    old = time.time() - hours * 3600
    os.utime(path, (old, old))


def test_leftovers_are_removed_when_stale(store):
    job = new_job(store)
    staging = store.jobs_dir / ".staging-abc"
    staging.mkdir()
    deleting = store.jobs_dir / ".deleting-def-1234"
    deleting.mkdir()
    tmp = store.job_dir(job.job_id) / ".job.json.xyz.tmp"
    tmp.write_bytes(b"{half")
    orphan_lock = store.locks_dir / "gone.lock"
    orphan_lock.write_bytes(b"")
    fresh_staging = store.jobs_dir / ".staging-new"
    fresh_staging.mkdir()
    for path in (staging, deleting, tmp, orphan_lock):
        _age(path, 2)

    result = cleanup(store, datetime.now(timezone.utc))

    assert result.leftovers_removed == 4
    assert not staging.exists() and not deleting.exists() and not tmp.exists() and not orphan_lock.exists()
    assert fresh_staging.exists()
    assert store.get(job.job_id).status == "queued"


def test_from_settings(store, clock):
    from qgen.config import Settings

    job = drive(store, new_job(store).job_id, "cancelled")
    settings = Settings(_env_file=None, result_ttl_days=1)
    assert run_cleanup_from_settings(store, settings, clock.now + timedelta(days=1)).jobs_deleted == [job.job_id]
