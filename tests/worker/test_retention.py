"""Time limits that the container check (tools/resilience_check.py) cannot wait for.

The clock of the job store is moved forward (24 h input TTL after failed, 7 days result TTL);
the periodic maintenance runs as in the worker, then the API answers as Laravel sees it.
Also the dev setting QGEN_MAINTENANCE_INTERVAL_S.
"""

from datetime import timedelta

import pytest
from rq import SimpleWorker
from rq.job import Job
from rq.registry import ScheduledJobRegistry

from service.jobs import tasks
from tests.fakes import FakeLLM, client_error
from tests.worker.conftest import HEADERS, scripted_answer

FAILED_INPUT_TTL = timedelta(hours=24)
RESULT_TTL = timedelta(days=7)


def _work(env) -> None:
    SimpleWorker([env.queue.queue], connection=env.redis).work(burst=True)


def _failed_job(env, external_ref: str) -> str:
    env.use(llm=FakeLLM(default=client_error(400)))
    job_id = env.create({"external_ref": external_ref, "language": "sk"})
    _work(env)
    assert env.status(job_id)["error"]["code"] == "llm_error"
    env.use(llm=FakeLLM(default=scripted_answer))
    return job_id


def _maintenance_at(env, moment) -> None:
    env.store.clock = lambda: moment
    tasks.run_maintenance()


def test_retry_just_before_24_hours_runs_again(env):
    job_id = _failed_job(env, "fail-early")
    finished = env.store.get(job_id).finished_at
    _maintenance_at(env, finished + FAILED_INPUT_TTL - timedelta(seconds=1))
    assert env.store.has_inputs(job_id)

    response = env.client.post(f"/v1/jobs/{job_id}/retry", headers=HEADERS)
    assert response.status_code == 202 and response.json()["job_id"] == job_id
    _work(env)
    assert env.status(job_id)["status"] == "completed"


def test_after_24_hours_pdfs_are_gone_and_retry_is_409(env):
    job_id = _failed_job(env, "fail-late")
    finished = env.store.get(job_id).finished_at
    _maintenance_at(env, finished + FAILED_INPUT_TTL)
    assert not env.store.has_inputs(job_id)

    response = env.client.post(f"/v1/jobs/{job_id}/retry", headers=HEADERS)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "job_not_retryable"
    assert env.status(job_id)["status"] == "failed"  # the record stays until the result TTL
    # the same external_ref still returns the failed job (it exists)
    again = env.create_response({"external_ref": "fail-late", "language": "sk"})
    assert again.status_code == 200 and again.json()["job_id"] == job_id


@pytest.mark.parametrize("failed", [False, True])
def test_after_7_days_job_is_gone_and_external_ref_makes_a_new_job(env, failed):
    if failed:
        job_id = _failed_job(env, "old")
    else:
        job_id = env.create({"external_ref": "old", "language": "sk"})
        _work(env)
        assert env.status(job_id)["status"] == "completed"
    finished = env.store.get(job_id).finished_at

    _maintenance_at(env, finished + RESULT_TTL - timedelta(seconds=1))
    assert env.client.get(f"/v1/jobs/{job_id}", headers=HEADERS).status_code == 200

    _maintenance_at(env, finished + RESULT_TTL)
    for path in ("", "/result", "/report"):
        response = env.client.get(f"/v1/jobs/{job_id}{path}", headers=HEADERS)
        assert response.status_code == 404 and response.json()["error"]["code"] == "job_not_found"

    new = env.create_response({"external_ref": "old", "language": "sk"})
    assert new.status_code == 202
    assert new.json()["job_id"] != job_id


def test_maintenance_interval_from_settings(env_factory):
    env = env_factory(maintenance_interval_s=15)
    env.context.wall_clock = lambda: 1000.0 + 3
    job_id = tasks.schedule_maintenance(env.context)
    assert job_id == tasks.maintenance_job_id(1003.0, 15) == "qgen-maintenance-67"
    scheduled = ScheduledJobRegistry(queue=env.queue.queue).get_scheduled_time(job_id)
    created = Job.fetch(job_id, connection=env.redis).created_at
    # enqueue_in counts from the real time: the delay is the rest of the 15 s slot (1005 - 1003)
    assert scheduled.timestamp() - created.timestamp() == pytest.approx(2.0, abs=1)


def test_default_maintenance_interval_is_ten_minutes(env):
    assert env.settings.maintenance_interval_s == tasks.MAINTENANCE_INTERVAL_S == 600


def test_rq_worker_options_follow_the_maintenance_interval():
    from rq.defaults import DEFAULT_MAINTENANCE_TASK_INTERVAL, DEFAULT_WORKER_TTL

    from service.jobs.rq_queue import rq_worker_options

    # production default: RQ's own defaults
    assert rq_worker_options(600) == {
        "maintenance_interval": DEFAULT_MAINTENANCE_TASK_INTERVAL,
        "worker_ttl": DEFAULT_WORKER_TTL,
    }
    # development (docker-compose.replay.yml): an idle worker waits at most 15 s in a dequeue
    options = rq_worker_options(15)
    assert options == {"maintenance_interval": 15, "worker_ttl": 30}
