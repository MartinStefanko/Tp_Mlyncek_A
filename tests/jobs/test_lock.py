"""Job lock: concurrent read-modify-write from threads and processes loses no update."""

import multiprocessing
import threading
from datetime import timedelta
from pathlib import Path

import pytest

from service.jobs import JobRecord, JobStore
from tests.jobs.conftest import new_job, set_status

THREADS = 8
PER_THREAD = 25
PROCESSES = 3
PER_PROCESS = 20


def _bump(record: JobRecord) -> None:
    record.attempts += 1  # read-modify-write of one number


def _bump_many(data_dir: str, job_id: str, count: int) -> None:
    """Runs in a child process (spawn): its own JobStore over the same directory."""
    store = JobStore(Path(data_dir), failed_input_ttl=timedelta(hours=24), lock_timeout_s=60)
    for _ in range(count):
        store.update(job_id, _bump)


def test_threads_lose_no_update(store):
    job = new_job(store)
    store.update(job.job_id, set_status("running"))  # attempts = 1
    errors: list[BaseException] = []

    def work():
        try:
            for _ in range(PER_THREAD):
                store.update(job.job_id, _bump)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert store.get(job.job_id).attempts == 1 + THREADS * PER_THREAD


def test_processes_lose_no_update(store):
    job = new_job(store)
    store.update(job.job_id, set_status("running"))
    ctx = multiprocessing.get_context("spawn")
    processes = [
        ctx.Process(target=_bump_many, args=(str(store.data_dir), job.job_id, PER_PROCESS)) for _ in range(PROCESSES)
    ]
    for process in processes:
        process.start()
    for _ in range(PER_PROCESS):  # the parent writes at the same time
        store.update(job.job_id, _bump)
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0

    assert store.get(job.job_id).attempts == 1 + PER_PROCESS * (PROCESSES + 1)


def test_reader_never_sees_a_broken_record(store):
    job = new_job(store)
    store.update(job.job_id, set_status("running"))
    stop = threading.Event()
    seen: list[int] = []
    errors: list[BaseException] = []

    def read():
        try:
            while not stop.is_set():
                seen.append(store.get(job.job_id).attempts)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    reader = threading.Thread(target=read)
    reader.start()
    for _ in range(50):
        store.update(job.job_id, _bump)
    stop.set()
    reader.join()

    assert errors == []
    assert seen == sorted(seen)  # monotonic: every read is a complete, committed state


def test_lock_timeout(store, tmp_path):
    job = new_job(store)
    impatient = JobStore(store.data_dir, failed_input_ttl=timedelta(hours=1), lock_timeout_s=0.2)
    with store.lock(job.job_id):
        result: list[BaseException] = []

        def other():  # another thread = another file handle, like another process
            try:
                impatient.update(job.job_id, _bump)
            except BaseException as exc:
                result.append(exc)

        thread = threading.Thread(target=other)
        thread.start()
        thread.join()
    assert len(result) == 1 and isinstance(result[0], TimeoutError)
    assert impatient.get(job.job_id).attempts == 0
