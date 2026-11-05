"""RealBackend: jobs in JobStore (job.json in QGEN_DATA_DIR), queue, external_ref in Redis.

POST /v1/jobs (kap. 3 – Význam odpovede 202). The route already checked the API key,
options (incl. OCR), the files and the PDFs. `create` then does:

    1. external_ref already reserved and its job exists  → 200 replayed (nothing written)
    2. active jobs >= QGEN_MAX_ACTIVE_JOBS                → 429 too_many_jobs + Retry-After
    3. reserve external_ref (Redis SET NX)                → lost the race: 200 with the winner's job
    4. JobStore.create (PDFs + job.json, fsync)           → error: release reservation, 500
    5. queue.enqueue                                      → error: delete job, release, 500
    6. 202

So a 202 means the PDFs are on disk and the job is queued; after any error there is
neither a stored job nor a reservation. Step 1 comes before the limit so that a repeated
POST of an existing job (e.g. after a timeout in Laravel) gets 200 even at the limit.

A job of another owner behaves as if it did not exist (404, not 403). Webhooks are not
sent here (job.cancelled on DELETE comes in step 11).
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Callable, Optional

from qgen.contract import JobListItem, JobStatus, Report, UnitsResult
from service.backend import CreateJobRequest, CreateJobResult
from service.errors import ApiError
from service.jobs.convert import to_job_status, to_list_item
from service.jobs.models import JobRecord
from service.jobs.queue import JobQueue
from service.jobs.refs import ExternalRefs, delete_job_and_ref, release_quietly
from service.jobs.store import STAGING_PREFIX, InvalidTransition, JobNotFound, JobStore, NewInputFile
from service.settings import ServiceSettings

logger = logging.getLogger("qgen.api.jobs")

RETRY_AFTER_S = 60
RESERVE_ATTEMPTS = 2
PENDING_POLL_S = 0.05
PENDING_GRACE_S = 1.0  # a fresh reservation may not have its staging directory yet


class RealBackend:
    def __init__(
        self,
        settings: ServiceSettings,
        store: JobStore,
        queue: JobQueue,
        refs: ExternalRefs,
        *,
        on_stop: Optional[Callable[[], None]] = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._queue = queue
        self._refs = refs
        self._on_stop = on_stop

    # --- lifecycle ------------------------------------------------------------------------

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        if self._on_stop is not None:
            self._on_stop()

    def health(self) -> dict[str, str]:
        try:
            redis_state = "ok" if self._refs.ping() else "error"
        except Exception:  # noqa: BLE001
            redis_state = "error"
        try:
            worker_state = "ok" if self._queue.has_live_worker() else "error"
        except Exception:  # noqa: BLE001
            worker_state = "error"
        return {"api": "ok", "redis": redis_state, "worker": worker_state}

    # --- create ---------------------------------------------------------------------------

    def create(self, request: CreateJobRequest) -> CreateJobResult:
        owner, external_ref = request.owner, request.options.external_ref
        files = [NewInputFile(filename=f.filename, content=f.content, pages=_pages(f.pages)) for f in request.files]
        if external_ref:
            existing = self._existing_job(owner, external_ref)
            if existing is not None:
                return self._replayed(existing)
        self._check_capacity()

        job_id = uuid.uuid4().hex
        if external_ref:
            existing = self._reserve(owner, external_ref, job_id)
            if existing is not None:
                return self._replayed(existing)

        try:
            record = self._store.create(owner=owner, options=request.options, files=files, job_id=job_id)
        except Exception as exc:
            logger.error("job store failed job_id=%s external_ref=%s error=%s", job_id, external_ref, type(exc).__name__)
            if external_ref:
                self._release(owner, external_ref, job_id)
            raise _not_stored() from None

        try:
            self._queue.enqueue(job_id)
        except Exception as exc:
            logger.error("job enqueue failed job_id=%s external_ref=%s error=%s", job_id, external_ref, type(exc).__name__)
            self._rollback(record)
            raise _not_stored() from None

        logger.info("job created job_id=%s external_ref=%s files=%d", job_id, external_ref, len(files))
        return CreateJobResult(status=to_job_status(record), replayed=False)

    def _existing_job(self, owner: str, external_ref: str) -> Optional[JobRecord]:
        """Job that holds the reservation (waits for one being created); None if free or stale."""
        try:
            holder = self._refs.lookup(owner, external_ref)
        except Exception as exc:
            raise _redis_unavailable(exc) from None
        if holder is None:
            return None
        record = self._wait_for_job(holder, owner)
        if record is None:
            self._release(owner, external_ref, holder)  # stale: no job.json, no staging
            logger.warning("stale external_ref reservation removed job_id=%s external_ref=%s", holder, external_ref)
        return record

    def _reserve(self, owner: str, external_ref: str, job_id: str) -> Optional[JobRecord]:
        """Reserve for `job_id`. None = reserved; otherwise the job that already holds it."""
        for _ in range(RESERVE_ATTEMPTS):
            try:
                holder = self._refs.reserve(owner, external_ref, job_id)
            except Exception as exc:
                raise _redis_unavailable(exc) from None
            if holder is None:
                return self._repair_lost_reservation(owner, external_ref, job_id)
            record = self._wait_for_job(holder, owner)
            if record is not None:
                return record
            self._release(owner, external_ref, holder)
        raise _not_stored()

    def _repair_lost_reservation(self, owner: str, external_ref: str, job_id: str) -> Optional[JobRecord]:
        """We got the reservation, but job.json may say the ref exists (Redis lost the key)."""
        stored = self._store.list_by_external_ref(owner, external_ref)
        if not stored:
            return None
        newest = stored[0]
        try:
            self._refs.force(owner, external_ref, newest.job_id)
        except Exception as exc:
            raise _redis_unavailable(exc) from None
        logger.warning("external_ref reservation restored job_id=%s external_ref=%s", newest.job_id, external_ref)
        return newest

    def _wait_for_job(self, job_id: str, owner: str) -> Optional[JobRecord]:
        """The job, waiting while another request is still writing it. None = no such job."""
        started = time.monotonic()
        while True:
            try:
                record = self._store.get(job_id)
                return record if record.owner == owner else None
            except JobNotFound:
                pass
            waited = time.monotonic() - started
            being_written = (self._store.jobs_dir / f"{STAGING_PREFIX}{job_id}").exists()
            if waited >= self._store.lock_timeout_s or (not being_written and waited >= PENDING_GRACE_S):
                return None
            time.sleep(PENDING_POLL_S)

    def _rollback(self, record: JobRecord) -> None:
        try:
            self._store.delete_job(record.job_id)
        except Exception as exc:  # job stays queued on disk; worker recovery queues it
            logger.error("rollback delete failed job_id=%s error=%s", record.job_id, type(exc).__name__)
            return
        release_quietly(self._refs, record)

    def _release(self, owner: str, external_ref: str, job_id: str) -> None:
        try:
            self._refs.release(owner, external_ref, job_id)
        except Exception as exc:  # a stale reservation is removed by the next POST
            logger.error("external_ref release failed job_id=%s error=%s", job_id, type(exc).__name__)

    def _replayed(self, record: JobRecord) -> CreateJobResult:
        logger.info("job replayed job_id=%s external_ref=%s", record.job_id, record.external_ref)
        return CreateJobResult(status=to_job_status(record), replayed=True)

    def _check_capacity(self) -> None:
        limit = self._settings.max_active_jobs
        active = sum(1 for job in self._store.iter_jobs() if job.is_active)
        if active >= limit:
            raise ApiError(
                429,
                "too_many_jobs",
                f"Too many active jobs (limit {limit}).",
                {"max_active_jobs": limit},
                headers={"Retry-After": str(RETRY_AFTER_S)},
            )

    # --- read -----------------------------------------------------------------------------

    def get(self, job_id: str, owner: str) -> JobStatus:
        return to_job_status(self._find(job_id, owner))

    def list_by_external_ref(self, external_ref: str, owner: str) -> list[JobListItem]:
        return [to_list_item(job) for job in self._store.list_by_external_ref(owner, external_ref)]

    def result(self, job_id: str, owner: str) -> UnitsResult:
        self._require_completed(job_id, owner)
        return self._store.load_result(job_id)

    def report(self, job_id: str, owner: str) -> Report:
        self._require_completed(job_id, owner)
        return self._store.load_report(job_id)

    # --- DELETE / retry -------------------------------------------------------------------

    def cancel_or_delete(self, job_id: str, owner: str) -> Optional[JobStatus]:
        record = self._find(job_id, owner)
        if record.is_active:
            was_queued = record.status == "queued"
            try:
                record = self._store.update(job_id, _cancel)
            except InvalidTransition:
                # The worker finished it meanwhile: report the final state, delete nothing.
                return to_job_status(self._find(job_id, owner))
            except JobNotFound:
                raise _job_not_found(job_id) from None
            if was_queued:  # no worker reads these PDFs; a running job's PDFs are deleted by the worker
                self._store.delete_inputs(job_id)
            logger.info("job cancelled job_id=%s", job_id)
            return to_job_status(record)
        try:
            delete_job_and_ref(self._store, self._refs, record)
        except JobNotFound:
            raise _job_not_found(job_id) from None
        return None

    def retry(self, job_id: str, owner: str) -> JobStatus:
        record = self._find(job_id, owner)
        if record.status != "failed":
            raise ApiError(409, "job_not_retryable", f"Job is {record.status}; only a failed job can be retried.")
        now: datetime = self._store.clock()
        expired = record.retryable_until is not None and now >= record.retryable_until
        if expired or not self._store.input_paths(job_id):
            raise ApiError(409, "job_not_retryable", "Input PDF was already deleted; create a new job.")
        self._check_capacity()

        original_error = record.error
        try:
            queued = self._store.update(job_id, _requeue)
        except InvalidTransition:  # retried by a parallel request
            raise ApiError(409, "job_not_retryable", "Job is no longer failed.") from None
        try:
            self._queue.enqueue(job_id)
        except Exception as exc:
            logger.error("retry enqueue failed job_id=%s error=%s", job_id, type(exc).__name__)
            try:
                self._store.update(job_id, _back_to_failed(original_error))
            except Exception as revert_exc:  # stays queued; worker recovery queues it
                logger.error("retry revert failed job_id=%s error=%s", job_id, type(revert_exc).__name__)
            raise ApiError(500, "internal_error", "Job could not be queued; it stays failed.") from None
        logger.info("job retried job_id=%s", job_id)
        return to_job_status(queued)

    # --- internals ------------------------------------------------------------------------

    def _find(self, job_id: str, owner: str) -> JobRecord:
        try:
            record = self._store.get(job_id)
        except JobNotFound:
            raise _job_not_found(job_id) from None
        if record.owner != owner:
            raise _job_not_found(job_id)
        return record

    def _require_completed(self, job_id: str, owner: str) -> None:
        status = self._find(job_id, owner).status
        if status in ("queued", "running"):
            raise ApiError(409, "result_not_ready", f"Job is {status}; the result is not ready yet.")
        if status != "completed":
            raise ApiError(409, "job_not_completed", f"Job is {status}; it has no result.")


def _cancel(record: JobRecord) -> None:
    record.status = "cancelled"  # store sets cancel_requested and finished_at


def _requeue(record: JobRecord) -> None:
    if record.status != "failed":
        raise InvalidTransition(record.job_id, record.status, "queued", "no longer failed")
    record.status = "queued"  # store resets attempts, error, retryable_until


def _back_to_failed(error) -> Callable[[JobRecord], None]:
    def change(record: JobRecord) -> None:
        record.status = "failed"
        record.error = error

    return change


def _pages(pages: Optional[int]) -> int:
    if pages is None:
        raise ValueError("RealBackend needs files checked by intake.inspect_pdfs (pages missing)")
    return pages


def _job_not_found(job_id: str) -> ApiError:
    return ApiError(404, "job_not_found", "Job does not exist or was deleted.", {"job_id": job_id})


def _not_stored() -> ApiError:
    return ApiError(500, "internal_error", "Job could not be stored; nothing was created.")


def _redis_unavailable(exc: Exception) -> ApiError:
    logger.error("redis unavailable error=%s", type(exc).__name__)
    return ApiError(500, "internal_error", "Queue is not available; nothing was created.")
