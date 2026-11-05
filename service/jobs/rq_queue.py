"""RqJobQueue: the job queue over RQ (queue "qgen"), used by the API and the worker.

The RQ job id is the job_id, the task is `service.jobs.tasks.run_job` given as a string,
so the API never imports the pipeline. job.json stays the source of truth; RQ keeps only
the queue entry (result_ttl=0, the RQ job is dropped after the run).

Time limit: RQ kills a run after QGEN_JOB_TIMEOUT_S + JOB_TIMEOUT_GRACE_S. The worker
itself stops at QGEN_JOB_TIMEOUT_S (service.jobs.runner) and writes failed; the grace
gives it time to do so before the hard kill.

`contains` is True only while the job waits in the queue (also in RQ's intermediate list
during dequeue) or runs on a LIVE worker. A worker is live while its Redis key exists and
has no death date: RQ extends the key on every heartbeat, a dead worker's key expires. A
job of a dead worker (left in StartedJobRegistry) does not count, so recovery brings it back.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

from rq import Queue, Worker
from rq.defaults import DEFAULT_WORKER_TTL
from rq.exceptions import NoSuchJobError
from rq.job import Job, JobStatus
from rq.registry import StartedJobRegistry

if TYPE_CHECKING:
    import redis

logger = logging.getLogger("qgen.jobs.queue")

QUEUE_NAME = "qgen"
RUN_JOB_TASK = "service.jobs.tasks.run_job"
JOB_TIMEOUT_GRACE_S = 120
FAILURE_TTL_S = 24 * 3600


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value or "")


class RqJobQueue:
    def __init__(self, connection: "redis.Redis", *, job_timeout_s: float, name: str = QUEUE_NAME) -> None:
        self.connection = connection
        self.queue = Queue(name, connection=connection)
        self.job_timeout_s = job_timeout_s

    @property
    def rq_timeout_s(self) -> int:
        """Hard limit of RQ: the job's own limit plus the grace for writing failed."""
        return int(self.job_timeout_s) + JOB_TIMEOUT_GRACE_S

    # --- JobQueue ------------------------------------------------------------------------

    def enqueue(self, job_id: str) -> None:
        if self.contains(job_id):
            logger.info("job already in queue job_id=%s", job_id)
            return
        self._drop_stale(job_id)
        self.queue.enqueue(
            RUN_JOB_TASK,
            job_id,
            job_id=job_id,
            job_timeout=self.rq_timeout_s,
            result_ttl=0,
            failure_ttl=FAILURE_TTL_S,
        )

    def contains(self, job_id: str) -> bool:
        if self.waits(job_id):
            return True
        job = self._fetch(job_id)
        if job is None or job.get_status(refresh=False) != JobStatus.STARTED:
            return False
        return bool(job.worker_name) and self.worker_alive(job.worker_name)

    def has_live_worker(self) -> bool:
        return queue_has_live_worker(self.queue)

    # --- helpers -------------------------------------------------------------------------

    def waits(self, job_id: str) -> bool:
        """The id is in the queue list or in RQ's intermediate list (being dequeued)."""
        for key in (self.queue.key, self.queue.intermediate_queue_key):
            if self.connection.lpos(key, job_id) is not None:
                return True
        return False

    def worker_alive(self, name: str) -> bool:
        return worker_alive(self.connection, name)

    def _fetch(self, job_id: str) -> Optional[Job]:
        try:
            return Job.fetch(job_id, connection=self.connection)
        except NoSuchJobError:
            return None

    def _drop_stale(self, job_id: str) -> None:
        """Delete an old RQ job with this id (finished, failed, or of a dead worker).

        Without this, StartedJobRegistry.cleanup could later take the new run for the
        abandoned old one and mark it failed.
        """
        job = self._fetch(job_id)
        if job is None:
            return
        StartedJobRegistry(queue=self.queue).remove_executions(job)
        job.delete(remove_from_queue=True)
        logger.info("stale rq job removed job_id=%s", job_id)


def rq_worker_options(maintenance_interval_s: float) -> dict[str, int]:
    """Options of rq.Worker for both workers, from QGEN_MAINTENANCE_INTERVAL_S.

    After a worker is killed, its scheduler lock (61 s) blocks the scheduler of the new
    worker; RQ takes the lock over only at its own maintenance, which runs between two
    dequeues. An idle worker waits in a dequeue for worker_ttl - 15 s (405 s by default).
    So both RQ's maintenance interval and that wait follow our interval; with the default
    600 s the values are RQ's defaults (maintenance 600 s, worker_ttl 420 s).
    """
    interval = max(1, int(maintenance_interval_s))
    return {"maintenance_interval": interval, "worker_ttl": min(DEFAULT_WORKER_TTL, interval + 15)}


def worker_alive(connection: "redis.Redis", name: str) -> bool:
    """The worker's Redis key exists (extended on every heartbeat) and has no death date."""
    key = Worker.redis_worker_namespace_prefix + name
    if not connection.exists(key):
        return False
    return not _text(connection.hget(key, "death"))


def queue_has_live_worker(queue: Queue) -> bool:
    """At least one live worker serves `queue` (also used for the webhook queue)."""
    return any(worker_alive(queue.connection, worker.name) for worker in Worker.all(queue=queue))
