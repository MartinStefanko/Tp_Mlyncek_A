"""Deletion by time (kap. 3 – Mazanie vstupov, Doba uchovania). Scheduled by the worker (service.jobs.tasks.run_maintenance).

- completed / cancelled: the PDFs are deleted at once (the worker also does it right after
  the final state; this catches the rest);
- failed: the PDFs stay until retryable_until (QGEN_FAILED_INPUT_TTL_HOURS), then deleted;
- every final job is deleted completely QGEN_RESULT_TTL_DAYS after finished_at, together
  with its external_ref reservation (when `refs` is given);
- active jobs are never deleted;
- housekeeping: leftovers of crashed operations (.staging-*, .deleting-*, *.tmp) and lock
  files without a job, older than `stale_after`.
"""

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional

from service.jobs.refs import ExternalRefs, delete_job_and_ref
from service.jobs.store import JobNotFound, JobStore, TMP_SUFFIX, JOB_ID_PATTERN

if TYPE_CHECKING:
    from qgen.config import Settings

logger = logging.getLogger("qgen.jobs.cleanup")

STALE_AFTER = timedelta(hours=1)


@dataclass
class CleanupResult:
    inputs_deleted: list[str] = field(default_factory=list)
    jobs_deleted: list[str] = field(default_factory=list)
    leftovers_removed: int = 0


def run_cleanup(
    store: JobStore,
    now: datetime,
    *,
    result_ttl: timedelta,
    stale_after: timedelta = STALE_AFTER,
    refs: Optional[ExternalRefs] = None,
) -> CleanupResult:
    result = CleanupResult()
    for job in store.iter_jobs():
        if not job.is_final:
            continue
        try:
            if job.finished_at is not None and now >= job.finished_at + result_ttl:
                delete_job_and_ref(store, refs, job)
                result.jobs_deleted.append(job.job_id)
                continue
            if _inputs_expired(job.status, job.retryable_until, now) and store.delete_inputs(job.job_id):
                result.inputs_deleted.append(job.job_id)
        except JobNotFound:
            continue
    result.leftovers_removed = _remove_leftovers(store, now, stale_after)
    if result.inputs_deleted or result.jobs_deleted or result.leftovers_removed:
        logger.info(
            "cleanup inputs=%d jobs=%d leftovers=%d",
            len(result.inputs_deleted),
            len(result.jobs_deleted),
            result.leftovers_removed,
        )
    return result


def run_cleanup_from_settings(
    store: JobStore, settings: "Settings", now: datetime, refs: Optional[ExternalRefs] = None
) -> CleanupResult:
    return run_cleanup(store, now, result_ttl=timedelta(days=settings.result_ttl_days), refs=refs)


def _inputs_expired(status: str, retryable_until, now: datetime) -> bool:
    if status in ("completed", "cancelled"):
        return True
    return status == "failed" and (retryable_until is None or now >= retryable_until)


def _remove_leftovers(store: JobStore, now: datetime, stale_after: timedelta) -> int:
    def is_stale(path) -> bool:
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except FileNotFoundError:
            return False
        return now - modified >= stale_after

    removed = 0
    for path in store.jobs_dir.iterdir():
        if path.name.startswith(".") and path.is_dir() and is_stale(path):
            shutil.rmtree(path, ignore_errors=True)  # .staging-* / .deleting-*
            removed += 1
        elif path.is_dir() and JOB_ID_PATTERN.match(path.name):
            for tmp in path.glob(f"*{TMP_SUFFIX}"):
                if is_stale(tmp):
                    tmp.unlink(missing_ok=True)
                    removed += 1
    for lock in store.locks_dir.glob("*.lock"):
        job_id = lock.stem
        if not (store.jobs_dir / job_id).exists() and is_stale(lock):
            try:
                lock.unlink()
                removed += 1
            except OSError:  # held by someone (Windows) – next run
                pass
    return removed
