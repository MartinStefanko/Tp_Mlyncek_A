"""Recovery at worker start (kap. 3 – Reštart počas spracovania).

Jobs in state queued or running that are not in the queue were lost (worker crash, Redis
lost the queue). They are queued again and start from the beginning. `attempts` counts the
runs started by a worker; a running job that already had `max_attempts` runs goes to failed
with error.code worker_lost. A queued job never ran, so it is only queued again.

job.json is written first, then the job is enqueued: a crash between the two is fixed by
the next start.
"""

import logging
from dataclasses import dataclass, field

from qgen.contract import JobError
from service.jobs.models import JobRecord
from service.jobs.queue import JobQueue
from service.jobs.store import InvalidTransition, JobNotFound, JobStore

logger = logging.getLogger("qgen.jobs.recovery")

MAX_ATTEMPTS = 2


@dataclass
class RecoveryResult:
    requeued: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # changed meanwhile (e.g. cancelled, deleted)


def recover_jobs(store: JobStore, queue: JobQueue, *, max_attempts: int = MAX_ATTEMPTS) -> RecoveryResult:
    result = RecoveryResult()
    for job in store.iter_jobs():
        if not job.is_active or queue.contains(job.job_id):
            continue
        try:
            if job.status == "running" and job.attempts >= max_attempts:
                store.update(job.job_id, _worker_lost)
                result.failed.append(job.job_id)
                logger.warning("job lost twice, failed job_id=%s attempts=%d", job.job_id, job.attempts)
                continue
            if job.status == "running":
                store.update(job.job_id, _back_to_queue)
            queue.enqueue(job.job_id)
            result.requeued.append(job.job_id)
            logger.info("job requeued job_id=%s attempts=%d", job.job_id, job.attempts)
        except (InvalidTransition, JobNotFound):
            result.skipped.append(job.job_id)
    return result


def _back_to_queue(record: JobRecord) -> None:
    if record.status != "running":  # changed since iter_jobs read it
        raise InvalidTransition(record.job_id, record.status, "queued", "no longer running")
    record.status = "queued"


def _worker_lost(record: JobRecord) -> None:
    if record.status != "running":
        raise InvalidTransition(record.job_id, record.status, "failed", "no longer running")
    record.status = "failed"
    record.error = JobError(
        code="worker_lost",
        message="The worker stopped during processing; the job was lost twice.",
        details={"attempts": record.attempts},
    )
