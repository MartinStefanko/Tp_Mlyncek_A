"""Recovery at worker start: requeue lost jobs, count attempts, worker_lost after two runs."""

from service.jobs.queue import InMemoryJobQueue
from service.jobs.recovery import recover_jobs
from tests.jobs.conftest import FAILED_TTL, drive, new_job, set_status


def test_queued_job_lost_from_queue_is_requeued_without_attempt(store):
    job = new_job(store)
    queue = InMemoryJobQueue()

    result = recover_jobs(store, queue)

    assert result.requeued == [job.job_id] and result.failed == []
    assert queue.waiting == [job.job_id]
    record = store.get(job.job_id)
    assert (record.status, record.attempts) == ("queued", 0)


def test_running_job_of_dead_worker_runs_again(store):
    job = drive(store, new_job(store).job_id, "running")
    store.update(job.job_id, lambda r: setattr(r, "progress", 0.4))
    queue = InMemoryJobQueue()

    result = recover_jobs(store, queue)

    assert result.requeued == [job.job_id]
    record = store.get(job.job_id)
    assert (record.status, record.attempts, record.stage, record.progress) == ("queued", 1, None, 0.0)
    assert queue.waiting == [job.job_id]


def test_second_lost_run_fails_with_worker_lost(store, clock):
    job = new_job(store)
    queue = InMemoryJobQueue()
    store.update(job.job_id, set_status("running"))  # run 1, worker dies
    recover_jobs(store, queue)
    queue.waiting.clear()  # the next worker takes it …
    store.update(job.job_id, set_status("running"))  # run 2, worker dies again

    result = recover_jobs(store, queue)

    assert result.failed == [job.job_id] and result.requeued == []
    record = store.get(job.job_id)
    assert record.status == "failed" and record.attempts == 2
    assert record.error.code == "worker_lost" and record.error.details == {"attempts": 2}
    assert record.finished_at == clock.now and record.retryable_until == clock.now + FAILED_TTL
    assert queue.waiting == []


def test_retry_after_worker_lost_starts_a_new_cycle(store):
    job = new_job(store)
    queue = InMemoryJobQueue()
    for _ in range(2):
        store.update(job.job_id, set_status("running"))
        recover_jobs(store, queue)
        queue.waiting.clear()
    assert store.get(job.job_id).status == "failed"

    store.update(job.job_id, set_status("queued"))  # POST /retry
    store.update(job.job_id, set_status("running"))  # attempts 1 again
    recover_jobs(store, queue)
    assert store.get(job.job_id).status == "queued"


def test_jobs_in_queue_or_on_live_worker_are_left_alone(store):
    waiting = new_job(store)
    running = drive(store, new_job(store).job_id, "running")
    queue = InMemoryJobQueue()
    queue.enqueue(waiting.job_id)
    queue.running.add(running.job_id)

    result = recover_jobs(store, queue)

    assert result.requeued == [] and result.failed == []
    assert queue.waiting == [waiting.job_id]
    assert store.get(running.job_id).status == "running"


def test_final_jobs_are_ignored(store):
    for path in (["running", "completed"], ["running", "failed"], ["cancelled"]):
        drive(store, new_job(store).job_id, *path)
    queue = InMemoryJobQueue()
    result = recover_jobs(store, queue)
    assert (result.requeued, result.failed, result.skipped, queue.waiting) == ([], [], [], [])


def test_second_recovery_does_not_duplicate(store):
    job = drive(store, new_job(store).job_id, "running")
    queue = InMemoryJobQueue()
    recover_jobs(store, queue)
    again = recover_jobs(store, queue)
    assert again.requeued == [] and queue.waiting == [job.job_id]
    assert store.get(job.job_id).attempts == 1


def test_job_cancelled_meanwhile_is_skipped(store, monkeypatch):
    job = drive(store, new_job(store).job_id, "running")
    real_iter = store.iter_jobs

    def iter_then_cancel():
        for record in real_iter():
            store.update(record.job_id, set_status("cancelled"))  # DELETE races with recovery
            yield record

    monkeypatch.setattr(store, "iter_jobs", iter_then_cancel)
    queue = InMemoryJobQueue()
    result = recover_jobs(store, queue)
    assert result.skipped == [job.job_id] and queue.waiting == []
    assert store.get(job.job_id).status == "cancelled"
