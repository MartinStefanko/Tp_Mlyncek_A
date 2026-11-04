"""JobStore: atomic write, create all-or-nothing, state transitions, result/report, delete."""

import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path

import pytest

from qgen.contract import JobError, Report, UnitsResult
from service.jobs import InvalidTransition, JobNotFound, JobRecord, JobStore, NewInputFile
from service.jobs import store as store_module
from service.jobs.store import ALLOWED_TRANSITIONS, atomic_write_bytes
from tests.jobs.conftest import FAILED_TTL, LLM_ERROR, PDF, T0, drive, new_job, options, pdf_files, set_status

FIXTURES = Path(__file__).resolve().parent.parent / "contract" / "fixtures"
STATUSES = ("queued", "running", "completed", "failed", "cancelled")


def _job_json(store: JobStore, job_id: str) -> dict:
    return json.loads((store.job_dir(job_id) / "job.json").read_text(encoding="utf-8"))


def _leftovers(path: Path) -> list[Path]:
    return [p for p in path.rglob("*") if p.name.endswith(".tmp")]


# --- atomic write ----------------------------------------------------------------------


def test_atomic_write_replaces_content(tmp_path):
    target = tmp_path / "a.json"
    atomic_write_bytes(target, b"one")
    atomic_write_bytes(target, b"two")
    assert target.read_bytes() == b"two"
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize("failing", ["fsync", "replace"])
def test_interrupted_write_keeps_old_job_json(store, monkeypatch, failing):
    job = new_job(store)
    before = _job_json(store, job.job_id)

    def boom(*args, **kwargs):
        raise OSError("simulated crash")

    monkeypatch.setattr(store_module.os, failing, boom)
    with pytest.raises(OSError, match="simulated crash"):
        store.update(job.job_id, set_status("running"))
    monkeypatch.undo()

    assert _job_json(store, job.job_id) == before
    assert store.get(job.job_id).status == "queued"
    assert _leftovers(store.data_dir) == []


def test_partial_write_never_reaches_job_json(store, monkeypatch):
    """The write dies after half the bytes: job.json stays complete and valid."""
    job = new_job(store)
    real_fdopen = os.fdopen

    class HalfWriter:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.handle.close()

        def write(self, data):
            self.handle.write(data[: len(data) // 2])
            raise OSError("disk died")

    monkeypatch.setattr(store_module.os, "fdopen", lambda fd, mode: HalfWriter(real_fdopen(fd, mode)))
    with pytest.raises(OSError, match="disk died"):
        store.update(job.job_id, set_status("running"))
    monkeypatch.undo()

    assert JobRecord.model_validate(_job_json(store, job.job_id)).status == "queued"
    assert _leftovers(store.data_dir) == []


# --- create ----------------------------------------------------------------------------


def test_create_writes_inputs_and_record(store):
    files = pdf_files(2)
    job = store.create(owner="key-a", options=options("gen-7"), files=files)

    directory = store.job_dir(job.job_id)
    assert sorted(p.name for p in (directory / "input").iterdir()) == ["D1.pdf", "D2.pdf"]
    assert (directory / "input" / "D2.pdf").read_bytes() == files[1].content
    assert job.status == "queued" and job.attempts == 0 and job.stage is None and job.progress == 0.0
    assert job.external_ref == "gen-7" and job.owner == "key-a"
    assert job.created_at == job.updated_at == T0
    first = job.input_files[0]
    assert (first.name, first.stored_as, first.size, first.pages) == ("doc0.pdf", "D1.pdf", len(files[0].content), 3)
    assert first.sha256 == hashlib.sha256(files[0].content).hexdigest()
    assert store.get(job.job_id) == job
    assert JobRecord.model_validate(_job_json(store, job.job_id)) == job
    assert [p.name for p in store.input_paths(job.job_id)] == ["D1.pdf", "D2.pdf"]
    assert not any(p.name.startswith(".") for p in store.jobs_dir.iterdir())


def test_create_keeps_client_file_name_off_the_disk(store):
    job = store.create(owner="o", options=options(), files=[NewInputFile("../../evil.pdf", PDF, 1)])
    assert job.input_files[0].name == "../../evil.pdf"
    assert [p.name for p in (store.job_dir(job.job_id) / "input").iterdir()] == ["D1.pdf"]
    assert not list(store.data_dir.parent.glob("evil.pdf"))


def test_create_is_all_or_nothing(store, monkeypatch):
    real = store_module._write_new_file
    calls = []

    def fail_on_second(path, data):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("disk full")
        real(path, data)

    monkeypatch.setattr(store_module, "_write_new_file", fail_on_second)
    with pytest.raises(OSError, match="disk full"):
        store.create(owner="o", options=options(), files=pdf_files(3), job_id="job1")

    assert list(store.jobs_dir.iterdir()) == []
    with pytest.raises(JobNotFound):
        store.get("job1")


def test_create_fails_cleanly_when_job_json_cannot_be_written(store, monkeypatch):
    def boom(path, data):
        raise OSError("no space for job.json")

    monkeypatch.setattr(store_module, "atomic_write_bytes", boom)
    with pytest.raises(OSError):
        store.create(owner="o", options=options(), files=pdf_files(1), job_id="job2")
    assert list(store.jobs_dir.iterdir()) == []


def test_create_refuses_an_existing_id(store):
    store.create(owner="o", options=options(), files=pdf_files(1), job_id="same")
    with pytest.raises(FileExistsError):
        store.create(owner="o", options=options(), files=pdf_files(1), job_id="same")


@pytest.mark.parametrize("job_id", ["../x", "a/b", "", "a" * 65, "x.y"])
def test_unsafe_job_id_is_not_found(store, job_id):
    with pytest.raises(JobNotFound):
        store.get(job_id)


def test_missing_job(store):
    with pytest.raises(JobNotFound):
        store.get("nope")
    with pytest.raises(JobNotFound):
        store.update("nope", set_status("running"))


# --- transitions -----------------------------------------------------------------------


def _reach(store, status: str) -> JobRecord:
    job = new_job(store)
    path = {
        "queued": [],
        "running": ["running"],
        "completed": ["running", "completed"],
        "failed": ["running", "failed"],
        "cancelled": ["cancelled"],
    }[status]
    return drive(store, job.job_id, *path)


ALLOWED = sorted(ALLOWED_TRANSITIONS)
FORBIDDEN = sorted((a, b) for a in STATUSES for b in STATUSES if a != b and (a, b) not in ALLOWED_TRANSITIONS)


def test_allowed_transitions_are_the_documented_ones():
    assert set(ALLOWED) == {
        ("queued", "running"),
        ("queued", "cancelled"),
        ("queued", "failed"),
        ("running", "completed"),
        ("running", "failed"),
        ("running", "cancelled"),
        ("running", "queued"),
        ("failed", "queued"),
    }


@pytest.mark.parametrize("old,new", ALLOWED)
def test_allowed_transition(store, old, new):
    job = _reach(store, old)
    updated = store.update(job.job_id, set_status(new, LLM_ERROR if new == "failed" else None))
    assert updated.status == new
    assert store.get(job.job_id).status == new


@pytest.mark.parametrize("old,new", FORBIDDEN)
def test_forbidden_transition_leaves_job_json(store, old, new):
    job = _reach(store, old)
    before = _job_json(store, job.job_id)
    with pytest.raises(InvalidTransition):
        store.update(job.job_id, set_status(new, LLM_ERROR if new == "failed" else None))
    assert _job_json(store, job.job_id) == before


def test_cancelled_is_never_overwritten_by_the_worker(store, clock):
    job = drive(store, new_job(store).job_id, "running")
    store.update(job.job_id, set_status("cancelled"))  # DELETE during the run
    with pytest.raises(InvalidTransition):
        store.update(job.job_id, set_status("completed"))
    record = store.get(job.job_id)
    assert record.status == "cancelled" and record.cancel_requested and record.finished_at == T0


def test_final_job_cannot_change(store):
    job = _reach(store, "completed")

    def touch(record):
        record.progress = 0.5

    with pytest.raises(InvalidTransition):
        store.update(job.job_id, touch)


def test_fixed_fields(store):
    job = new_job(store)

    def steal(record):
        record.owner = "key-b"

    with pytest.raises(InvalidTransition):
        store.update(job.job_id, steal)


def test_failed_needs_an_error(store):
    job = drive(store, new_job(store).job_id, "running")
    with pytest.raises(InvalidTransition):
        store.update(job.job_id, set_status("failed"))


def test_invalid_value_is_rejected(store):
    job = new_job(store)

    def bad(record):
        record.progress = 2.0

    with pytest.raises(ValueError):
        store.update(job.job_id, bad)
    assert store.get(job.job_id).progress == 0.0


def test_transition_effects(store, clock):
    job = new_job(store)

    clock.advance(seconds=10)
    running = store.update(job.job_id, set_status("running"))
    assert running.attempts == 1 and running.updated_at == clock.now

    def progress(record):
        record.stage = "questions"
        record.progress = 0.5

    running = store.update(job.job_id, progress)
    assert (running.stage, running.progress) == ("questions", 0.5)

    clock.advance(minutes=5)
    failed = store.update(job.job_id, set_status("failed", LLM_ERROR))
    assert failed.finished_at == clock.now
    assert failed.retryable_until == clock.now + FAILED_TTL
    assert failed.error == LLM_ERROR and failed.stage == "questions"

    clock.advance(hours=1)
    retried = store.update(job.job_id, set_status("queued"))
    assert retried.attempts == 0  # retry is a new chance, not a continuation
    assert (retried.stage, retried.progress, retried.error) == (None, 0.0, None)
    assert retried.finished_at is None and retried.retryable_until is None
    assert retried.created_at == T0

    running = store.update(job.job_id, set_status("running"))
    assert running.attempts == 1
    done = store.update(job.job_id, set_status("completed"))
    assert (done.stage, done.progress, done.finished_at) == ("output", 1.0, clock.now)


def test_recovery_requeue_resets_progress_but_not_attempts(store):
    job = drive(store, new_job(store).job_id, "running")

    def progress(record):
        record.stage = "learning_objects"
        record.progress = 0.3

    store.update(job.job_id, progress)
    requeued = store.update(job.job_id, set_status("queued"))
    assert (requeued.stage, requeued.progress, requeued.attempts) == (None, 0.0, 1)
    assert store.update(job.job_id, set_status("running")).attempts == 2


# --- listing, result, report, delete ----------------------------------------------------


def test_list_by_external_ref_newest_first_and_per_owner(store, clock):
    first = new_job(store, owner="key-a", external_ref="gen-1")
    clock.advance(seconds=1)
    second = new_job(store, owner="key-a", external_ref="gen-1")
    new_job(store, owner="key-b", external_ref="gen-1")
    new_job(store, owner="key-a", external_ref="gen-2")

    assert [j.job_id for j in store.list_by_external_ref("key-a", "gen-1")] == [second.job_id, first.job_id]
    assert store.list_by_external_ref("key-c", "gen-1") == []


def test_iter_jobs_skips_staging_and_broken(store, caplog):
    job = new_job(store)
    (store.jobs_dir / ".staging-x").mkdir()
    broken = store.jobs_dir / "broken"
    broken.mkdir()
    (broken / "job.json").write_text("{not json", encoding="utf-8")
    assert [j.job_id for j in store.iter_jobs()] == [job.job_id]
    assert "broken" in caplog.text and "{not json" not in caplog.text


def test_result_and_report_roundtrip(store):
    job = drive(store, new_job(store).job_id, "running")
    result = UnitsResult.model_validate_json((FIXTURES / "units_example.json").read_text(encoding="utf-8"))
    report = Report.model_validate_json((FIXTURES / "report_example.json").read_text(encoding="utf-8"))
    store.save_result(job.job_id, result)
    store.save_report(job.job_id, report)
    assert store.load_result(job.job_id) == result
    assert store.load_report(job.job_id) == report
    assert _leftovers(store.data_dir) == []


def test_load_missing_result(store):
    job = new_job(store)
    with pytest.raises(JobNotFound):
        store.load_result(job.job_id)
    with pytest.raises(JobNotFound):
        store.save_result("nope", UnitsResult.model_validate_json((FIXTURES / "units_example.json").read_bytes()))


def test_delete_inputs_and_job(store):
    job = new_job(store, files=2)
    assert store.has_inputs(job.job_id)
    assert store.delete_inputs(job.job_id) is True
    assert store.delete_inputs(job.job_id) is False
    assert not store.has_inputs(job.job_id) and store.input_paths(job.job_id) == []
    assert store.get(job.job_id).input_files  # the record keeps the file list

    store.delete_job(job.job_id)
    with pytest.raises(JobNotFound):
        store.get(job.job_id)
    with pytest.raises(JobNotFound):
        store.delete_job(job.job_id)
    assert list(store.jobs_dir.iterdir()) == []


def test_from_settings(tmp_path):
    from qgen.config import Settings

    settings = Settings(_env_file=None, data_dir=tmp_path / "d", failed_input_ttl_hours=3)
    store = JobStore.from_settings(settings)
    assert store.data_dir == tmp_path / "d" and store.failed_input_ttl == timedelta(hours=3)
    assert (tmp_path / "d" / "jobs").is_dir() and (tmp_path / "d" / "locks").is_dir()


def test_job_error_codes_survive_roundtrip(store):
    job = drive(store, new_job(store).job_id, "running")
    error = JobError(code="worker_lost", message="lost", details={"attempts": 2})
    store.update(job.job_id, set_status("failed", error))
    assert store.get(job.job_id).error == error
