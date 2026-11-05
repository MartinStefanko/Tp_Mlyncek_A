"""RqJobQueue, recovery at worker start, periodic maintenance and /health over fakeredis."""

from datetime import timedelta

from rq import SimpleWorker, Worker
from rq.job import Job
from rq.registry import ScheduledJobRegistry, StartedJobRegistry

from service.jobs import tasks
from service.jobs.redis_lock import RedisLock
from service.jobs.rq_queue import JOB_TIMEOUT_GRACE_S, RUN_JOB_TASK
from tests.worker.conftest import HEADERS

SK = {"language": "sk"}


def _mark_started(env, job_id: str, worker_name: str) -> None:
    """Make the RQ job look taken by `worker_name` (left the queue, status started)."""
    env.redis.lrem(env.queue.queue.key, 0, job_id)
    job = Job.fetch(job_id, connection=env.redis)
    env.redis.hset(job.key, mapping={"status": "started", "worker_name": worker_name})
    StartedJobRegistry(queue=env.queue.queue).connection.zadd(
        StartedJobRegistry(queue=env.queue.queue).key, {f"{job_id}:exec": 1}
    )


def _live_worker(env, name: str = "w-live") -> Worker:
    worker = Worker([env.queue.queue], connection=env.redis, name=name)
    worker.register_birth()
    return worker


# --- enqueue / contains ---------------------------------------------------------------------


def test_enqueue_uses_the_job_id_the_task_and_the_time_limit(env_factory):
    env = env_factory(job_timeout_s=3600)
    job_id = env.create(SK)
    job = Job.fetch(job_id, connection=env.redis)
    assert job.id == job_id
    assert job.func_name == RUN_JOB_TASK
    assert job.args == (job_id,)
    assert job.timeout == 3600 + JOB_TIMEOUT_GRACE_S
    assert env.queue.queue.get_job_ids() == [job_id]


def test_enqueue_makes_no_duplicates(env):
    job_id = env.create(SK)
    env.queue.enqueue(job_id)
    env.queue.enqueue(job_id)
    assert env.queue.queue.get_job_ids() == [job_id]


def test_contains_waiting_and_being_dequeued(env):
    job_id = env.create(SK)
    assert env.queue.contains(job_id)
    env.redis.lrem(env.queue.queue.key, 0, job_id)
    assert not env.queue.contains(job_id)
    env.redis.lpush(env.queue.queue.intermediate_queue_key, job_id)
    assert env.queue.contains(job_id)
    assert not env.queue.contains("unknown")


def test_contains_running_on_live_worker_only(env):
    job_id = env.create(SK)
    worker = _live_worker(env)
    _mark_started(env, job_id, worker.name)
    assert env.queue.contains(job_id)

    worker.register_death()  # stopped worker: death date set
    assert not env.queue.contains(job_id)

    _mark_started(env, job_id, "w-gone")  # worker key expired long ago
    assert not env.queue.contains(job_id)


def test_enqueue_replaces_the_job_of_a_dead_worker(env):
    job_id = env.create(SK)
    _mark_started(env, job_id, "w-gone")
    env.queue.enqueue(job_id)
    assert env.queue.queue.get_job_ids() == [job_id]
    assert Job.fetch(job_id, connection=env.redis).get_status() == "queued"


# --- recovery -------------------------------------------------------------------------------


def test_recovery_at_start_requeues_lost_jobs_and_they_run(env):
    queued = env.create({**SK, "external_ref": "a"})
    running = env.create({**SK, "external_ref": "b"})
    env.redis.delete(env.queue.queue.key)  # Redis lost the queue
    env.store.update(running, lambda r: setattr(r, "status", "running"))
    _mark_started(env, running, "w-gone")

    result = tasks.recover_at_start(env.context)
    assert sorted(result.requeued) == sorted([queued, running])
    assert env.queue.contains(queued) and env.queue.contains(running)

    SimpleWorker([env.queue.queue], connection=env.redis).work(burst=True)
    assert env.status(queued)["status"] == "completed"
    assert env.status(running)["status"] == "completed"
    assert env.store.get(running).attempts == 2


def test_recovery_leaves_a_job_on_a_live_worker(env):
    job_id = env.create(SK)
    env.store.update(job_id, lambda r: setattr(r, "status", "running"))
    _mark_started(env, job_id, _live_worker(env).name)
    result = tasks.recover_at_start(env.context)
    assert result.requeued == [] and env.store.get(job_id).status == "running"


def test_recovery_is_skipped_while_another_worker_recovers(env):
    with RedisLock(env.redis, tasks.RECOVERY_LOCK_KEY, ttl_s=60):
        assert tasks.recover_at_start(env.context) is None


# --- maintenance ----------------------------------------------------------------------------


def test_maintenance_runs_cleanup_and_schedules_the_next_run(env, monkeypatch):
    calls = []
    monkeypatch.setattr(tasks, "run_cleanup_from_settings", lambda *a, **k: calls.append(a) or "done")
    env.context.wall_clock = lambda: 6000.0 + 5
    assert tasks.run_maintenance() == "done"
    assert len(calls) == 1 and calls[0][3] is env.refs
    assert ScheduledJobRegistry(queue=env.queue.queue).get_job_ids() == [tasks.maintenance_job_id(6005.0)]
    assert not env.redis.exists(tasks.MAINTENANCE_LOCK_KEY)


def test_maintenance_never_runs_twice_at_once(env, monkeypatch):
    calls = []
    monkeypatch.setattr(tasks, "run_cleanup_from_settings", lambda *a, **k: calls.append(a))
    with RedisLock(env.redis, tasks.MAINTENANCE_LOCK_KEY, ttl_s=60) as held:
        assert held.acquired
        assert tasks.run_maintenance() is None
    assert calls == []
    assert len(ScheduledJobRegistry(queue=env.queue.queue).get_job_ids()) == 1  # chain goes on


def test_more_workers_schedule_one_maintenance(env):
    env.context.wall_clock = lambda: 1200.0 + 30
    first = tasks.schedule_maintenance(env.context)
    second = tasks.schedule_maintenance(env.context)
    assert first == second == "qgen-maintenance-3"
    registry = ScheduledJobRegistry(queue=env.queue.queue)
    assert registry.get_job_ids() == [first]
    job = Job.fetch(first, connection=env.redis)
    assert job.func_name == tasks.MAINTENANCE_TASK
    scheduled_at = registry.get_scheduled_time(first)
    assert scheduled_at.timestamp() - job.created_at.timestamp() <= 600


def test_maintenance_deletes_old_jobs_with_their_reservation(env):
    job_id = env.create({**SK, "external_ref": "old"})
    SimpleWorker([env.queue.queue], connection=env.redis).work(burst=True)
    assert env.status(job_id)["status"] == "completed"
    record = env.store.get(job_id)
    env.store.clock = lambda: record.finished_at + timedelta(days=8)
    tasks.run_maintenance()
    assert env.client.get(f"/v1/jobs/{job_id}", headers=HEADERS).status_code == 404
    assert env.refs.lookup(record.owner, "old") is None


# --- health ---------------------------------------------------------------------------------


def test_health_worker_state(env):
    response = env.client.get("/health")
    assert response.status_code == 503
    assert response.json()["components"] == {"api": "ok", "redis": "ok", "worker": "error", "webhook_worker": "error"}

    worker = _live_worker(env)
    response = env.client.get("/health")
    assert response.status_code == 503  # the webhook worker is still missing
    assert response.json()["components"]["worker"] == "ok"

    hook_worker = Worker([env.hooks.queue], connection=env.redis, name="w-hooks")
    hook_worker.register_birth()
    response = env.client.get("/health")
    assert response.status_code == 200
    assert response.json()["components"] == {"api": "ok", "redis": "ok", "worker": "ok", "webhook_worker": "ok"}

    worker.register_death()
    assert env.client.get("/health").json()["components"]["worker"] == "error"
    assert env.client.get("/health").json()["components"]["webhook_worker"] == "ok"

    hook_worker.register_death()
    assert env.client.get("/health").json()["components"]["webhook_worker"] == "error"


def test_health_job_worker_does_not_count_as_webhook_worker(env):
    _live_worker(env)  # serves only "qgen"
    assert env.client.get("/health").json()["components"]["webhook_worker"] == "error"
