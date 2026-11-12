"""JobStore: jobs as directories under QGEN_DATA_DIR.

Layout:
    {data_dir}/jobs/{job_id}/job.json        JobRecord – source of truth about the job
    {data_dir}/jobs/{job_id}/input/D1.pdf …  uploaded PDFs (the client file name is only in job.json)
    {data_dir}/jobs/{job_id}/result.json     UnitsResult
    {data_dir}/jobs/{job_id}/report.json     Report
    {data_dir}/jobs/.staging-{job_id}/       job being created (renamed to {job_id} when complete)
    {data_dir}/jobs/.deleting-{job_id}-…/    job being deleted (renamed away first, then removed)
    {data_dir}/locks/{job_id}.lock           lock file of the job

Writes are atomic: temporary file in the same directory → flush → fsync → os.replace →
fsync of the directory (skipped on Windows, where a directory cannot be opened for fsync).

Concurrency: the API and the worker (different processes / containers on the same volume)
may write the same job.json. Every operation on a job holds the job's lock
(`filelock.FileLock`: fcntl.flock on Linux, msvcrt.locking on Windows). The OS releases the
lock when the process dies, so a crash never leaves a stuck lock. Reads hold the lock too:
on Windows os.replace fails while another process has the target file open.

State transitions are checked in `update` (see ALLOWED_TRANSITIONS); their side effects
(attempts, finished_at, retryable_until, reset on retry) are applied here, in one place.
"""

import hashlib
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Optional

import filelock
from pydantic import ValidationError

from qgen.contract import JobOptions, Report, UnitsResult
from service.jobs.models import InputFile, JobRecord

if TYPE_CHECKING:
    from qgen.config import Settings

logger = logging.getLogger("qgen.jobs.store")

JOB_FILE = "job.json"
RESULT_FILE = "result.json"
REPORT_FILE = "report.json"
INPUT_DIR = "input"
STAGING_PREFIX = ".staging-"
DELETING_PREFIX = ".deleting-"
TMP_SUFFIX = ".tmp"

JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# (old status, new status) pairs that `update` accepts. The two recovery transitions
# (running → queued, queued → failed) go beyond the basic lifecycle: recovery puts a
# running job of a dead worker back to the queue, and worker_lost can hit a queued job.
ALLOWED_TRANSITIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("queued", "running"),
        ("queued", "cancelled"),
        ("queued", "failed"),  # recovery: worker_lost
        ("running", "completed"),
        ("running", "failed"),
        ("running", "cancelled"),
        ("running", "queued"),  # recovery: run again from the start
        ("failed", "queued"),  # retry
    }
)

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobNotFound(Exception):
    def __init__(self, job_id: str) -> None:
        super().__init__(f"job {job_id} not found")
        self.job_id = job_id


class InvalidTransition(Exception):
    def __init__(self, job_id: str, old: str, new: str, reason: str = "") -> None:
        message = f"job {job_id}: transition {old} -> {new} is not allowed"
        super().__init__(f"{message} ({reason})" if reason else message)
        self.job_id = job_id
        self.old = old
        self.new = new


@dataclass(frozen=True)
class NewInputFile:
    """A checked PDF for `JobStore.create`; the page count comes from the API intake."""

    filename: str
    content: bytes
    pages: int


# --- low-level file helpers --------------------------------------------------------------


def fsync_dir(path: Path) -> None:
    """Make a rename/create inside `path` durable. No-op on Windows (no directory handles)."""
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write `path` so that a reader sees either the old or the new content, never a mix."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=TMP_SUFFIX)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        replace_file(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    fsync_dir(path.parent)


# Windows only: a virus scanner or the search indexer may hold the just written file open for
# a moment without FILE_SHARE_DELETE; os.replace then fails with PermissionError (errno 13).
# The job lock serializes our own readers and writers, so this can only be such a short
# outside handle: wait 10, 20, ... ms and try again. On Linux (Docker) it never happens.
REPLACE_ATTEMPTS = 10
REPLACE_WAIT_S = 0.01
RETRY_REPLACE = os.name == "nt"


def replace_file(source: Path, target: Path, *, sleep: Callable[[float], None] = time.sleep) -> None:
    for attempt in range(1, REPLACE_ATTEMPTS + 1):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if not RETRY_REPLACE or attempt == REPLACE_ATTEMPTS:
                raise
            sleep(REPLACE_WAIT_S * attempt)


def _write_new_file(path: Path, data: bytes) -> None:
    with open(path, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


# --- store -------------------------------------------------------------------------------


class JobStore:
    def __init__(
        self,
        data_dir: Path,
        *,
        failed_input_ttl: timedelta,
        clock: Clock = utc_now,
        lock_timeout_s: float = 30.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.jobs_dir = self.data_dir / "jobs"
        self.locks_dir = self.data_dir / "locks"
        self.failed_input_ttl = failed_input_ttl
        self.clock = clock
        self.lock_timeout_s = lock_timeout_s
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_settings(cls, settings: "Settings", **kwargs) -> "JobStore":
        return cls(
            settings.data_dir,
            failed_input_ttl=timedelta(hours=settings.failed_input_ttl_hours),
            **kwargs,
        )

    # --- paths and lock ----------------------------------------------------------------

    def job_dir(self, job_id: str) -> Path:
        if not JOB_ID_PATTERN.match(job_id):  # never build a path from an unchecked id
            raise JobNotFound(job_id)
        return self.jobs_dir / job_id

    def lock_path(self, job_id: str) -> Path:
        self.job_dir(job_id)  # validates the id
        return self.locks_dir / f"{job_id}.lock"

    @contextmanager
    def lock(self, job_id: str) -> Iterator[None]:
        """Exclusive lock of one job across threads and processes. TimeoutError after lock_timeout_s."""
        lock = filelock.FileLock(self.lock_path(job_id), timeout=self.lock_timeout_s)
        try:
            lock.acquire()
        except filelock.Timeout:
            raise TimeoutError(f"lock of job {job_id} not acquired in {self.lock_timeout_s} s") from None
        try:
            yield
        finally:
            lock.release()

    # --- create ------------------------------------------------------------------------

    def create(
        self,
        *,
        owner: str,
        options: JobOptions,
        files: list[NewInputFile],
        job_id: Optional[str] = None,
    ) -> JobRecord:
        """Write the PDFs and job.json; all or nothing.

        Everything goes into jobs/.staging-{id}/ first, which is then renamed to jobs/{id}/.
        On any error the staging directory is removed and the error propagates.
        """
        job_id = job_id or uuid.uuid4().hex
        final_dir = self.job_dir(job_id)
        staging = self.jobs_dir / f"{STAGING_PREFIX}{job_id}"
        with self.lock(job_id):
            if final_dir.exists():
                raise FileExistsError(f"job {job_id} already exists")
            try:
                input_dir = staging / INPUT_DIR
                input_dir.mkdir(parents=True)
                input_files = []
                for index, upload in enumerate(files, start=1):
                    stored_as = f"D{index}.pdf"
                    _write_new_file(input_dir / stored_as, upload.content)
                    input_files.append(
                        InputFile(
                            name=upload.filename,
                            stored_as=stored_as,
                            size=len(upload.content),
                            sha256=hashlib.sha256(upload.content).hexdigest(),
                            pages=upload.pages,
                        )
                    )
                fsync_dir(input_dir)
                now = self.clock()
                record = JobRecord(
                    job_id=job_id,
                    owner=owner,
                    external_ref=options.external_ref,
                    status="queued",
                    options=options,
                    input_files=input_files,
                    created_at=now,
                    updated_at=now,
                )
                atomic_write_bytes(staging / JOB_FILE, _dump(record))
                os.replace(staging, final_dir)
                fsync_dir(self.jobs_dir)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        logger.info("job stored job_id=%s files=%d", job_id, len(input_files))
        return record

    # --- read --------------------------------------------------------------------------

    def get(self, job_id: str) -> JobRecord:
        with self.lock(job_id):
            return self._read(job_id)

    def _read(self, job_id: str) -> JobRecord:
        try:
            raw = (self.job_dir(job_id) / JOB_FILE).read_bytes()
        except FileNotFoundError:
            raise JobNotFound(job_id) from None
        return JobRecord.model_validate_json(raw)

    def iter_jobs(self) -> Iterator[JobRecord]:
        """All stored jobs (order by directory name). Unreadable job.json is logged and skipped."""
        for path in sorted(self.jobs_dir.iterdir()):
            if path.name.startswith(".") or not path.is_dir() or not JOB_ID_PATTERN.match(path.name):
                continue
            try:
                yield self.get(path.name)
            except JobNotFound:
                continue  # deleted meanwhile
            except (ValidationError, ValueError) as exc:
                logger.warning("unreadable job.json job_id=%s error=%s", path.name, type(exc).__name__)

    def list_by_external_ref(self, owner: str, external_ref: str) -> list[JobRecord]:
        """Jobs of `owner` with this external_ref, newest first."""
        jobs = [j for j in self.iter_jobs() if j.owner == owner and j.external_ref == external_ref]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs

    # --- update ------------------------------------------------------------------------

    def update(self, job_id: str, change: Callable[[JobRecord], None]) -> JobRecord:
        """Read-modify-write under the job lock.

        `change` edits a copy of the record. A change of status must be in
        ALLOWED_TRANSITIONS, else InvalidTransition and job.json stays as it was.
        """
        with self.lock(job_id):
            old = self._read(job_id)
            new = old.model_copy(deep=True)
            change(new)
            now = self.clock()
            self._apply_transition(old, new, now)
            new.updated_at = now
            record = JobRecord.model_validate(new.model_dump())  # validate the edited copy
            atomic_write_bytes(self.job_dir(job_id) / JOB_FILE, _dump(record))
        if old.status != record.status:
            logger.info("job %s -> %s job_id=%s attempts=%d", old.status, record.status, job_id, record.attempts)
        return record

    def _apply_transition(self, old: JobRecord, new: JobRecord, now: datetime) -> None:
        if new.job_id != old.job_id or new.owner != old.owner or new.created_at != old.created_at:
            raise InvalidTransition(old.job_id, old.status, new.status, "job_id, owner and created_at are fixed")

        if old.status == new.status:
            if old.is_final and _changed(old, new):
                raise InvalidTransition(old.job_id, old.status, new.status, "a final job cannot change")
            return
        if (old.status, new.status) not in ALLOWED_TRANSITIONS:
            raise InvalidTransition(old.job_id, old.status, new.status)

        target = new.status
        if old.status == "queued" and target == "running":
            new.attempts = old.attempts + 1
        elif target == "queued":
            new.stage = None
            new.progress = 0.0
            if old.status == "failed":  # retry: a new chance from the caller, not a continuation
                new.attempts = 0
                new.error = None
                new.finished_at = None
                new.retryable_until = None
                new.cancel_requested = False
        elif target == "completed":
            new.finished_at = now
            new.stage = "output"
            new.progress = 1.0
            new.error = None
        elif target == "failed":
            if new.error is None:
                raise InvalidTransition(old.job_id, old.status, target, "a failed job needs an error")
            new.finished_at = now
            new.retryable_until = now + self.failed_input_ttl
        elif target == "cancelled":
            new.finished_at = now
            new.cancel_requested = True

    # --- result, report, inputs ----------------------------------------------------------

    def save_result(self, job_id: str, result: UnitsResult) -> None:
        self._save_json(job_id, RESULT_FILE, result.model_dump_json().encode("utf-8"))

    def save_report(self, job_id: str, report: Report) -> None:
        self._save_json(job_id, REPORT_FILE, report.model_dump_json().encode("utf-8"))

    def load_result(self, job_id: str) -> UnitsResult:
        return UnitsResult.model_validate_json(self._load_json(job_id, RESULT_FILE))

    def load_report(self, job_id: str) -> Report:
        return Report.model_validate_json(self._load_json(job_id, REPORT_FILE))

    def _save_json(self, job_id: str, name: str, data: bytes) -> None:
        with self.lock(job_id):
            directory = self.job_dir(job_id)
            if not (directory / JOB_FILE).exists():
                raise JobNotFound(job_id)
            atomic_write_bytes(directory / name, data)

    def _load_json(self, job_id: str, name: str) -> bytes:
        with self.lock(job_id):
            try:
                return (self.job_dir(job_id) / name).read_bytes()
            except FileNotFoundError:
                raise JobNotFound(job_id) from None

    def input_paths(self, job_id: str) -> list[Path]:
        """Paths of the stored PDFs in the order of input_files (empty after delete_inputs)."""
        record = self.get(job_id)
        input_dir = self.job_dir(job_id) / INPUT_DIR
        return [input_dir / f.stored_as for f in record.input_files if (input_dir / f.stored_as).exists()]

    def has_inputs(self, job_id: str) -> bool:
        return (self.job_dir(job_id) / INPUT_DIR).exists()

    def delete_inputs(self, job_id: str) -> bool:
        """Delete input/; True when something was deleted."""
        with self.lock(job_id):
            input_dir = self.job_dir(job_id) / INPUT_DIR
            if not input_dir.exists():
                return False
            shutil.rmtree(input_dir)
            fsync_dir(input_dir.parent)
        logger.info("job inputs deleted job_id=%s", job_id)
        return True

    def delete_job(self, job_id: str) -> None:
        """Delete the whole job directory (job.json, inputs, result, report). JobNotFound if missing.

        The directory is first renamed to .deleting-…, so the job disappears at once and a
        half-deleted directory (crash during rmtree) is never read as a job.
        """
        with self.lock(job_id):
            directory = self.job_dir(job_id)
            if not (directory / JOB_FILE).exists():
                raise JobNotFound(job_id)
            trash = self.jobs_dir / f"{DELETING_PREFIX}{job_id}-{uuid.uuid4().hex[:8]}"
            os.replace(directory, trash)
            fsync_dir(self.jobs_dir)
        shutil.rmtree(trash, ignore_errors=True)
        logger.info("job deleted job_id=%s", job_id)


def _dump(record: JobRecord) -> bytes:
    return record.model_dump_json(indent=2).encode("utf-8")


def _changed(old: JobRecord, new: JobRecord) -> bool:
    exclude = {"updated_at"}
    return old.model_dump(exclude=exclude) != new.model_dump(exclude=exclude)
