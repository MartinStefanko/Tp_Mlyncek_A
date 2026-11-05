"""RQ tasks of the worker: run_job and the periodic maintenance.

RQ calls these functions by name ("service.jobs.tasks.run_job"). They use the
WorkerContext set by service.worker at start (`configure`); without one (e.g. a plain
`rq worker`), the context is built from the environment on first use. Tests call
`configure` with fake clients.

Maintenance (every QGEN_MAINTENANCE_INTERVAL_S, default 600 s; shorter only in development,
e.g. docker-compose.replay.yml, so a killed worker's job comes back within about two
minutes; scheduled by RQ `enqueue_in`, run by the
scheduler built into the worker, `--with-scheduler`):
    - run_cleanup (PDF deletion, deletion after QGEN_RESULT_TTL_DAYS incl. external_ref)
    - recover_jobs (jobs lost while the worker kept running: a killed work horse, or a
      restart while the old worker's key had not expired yet)
A Redis lock makes sure it never runs twice at the same time. The next run gets the RQ id
`qgen-maintenance-{slot}`, so more workers schedule the same entry and there is one chain.
"""

import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Callable, Optional

from service.jobs.cleanup import CleanupResult, run_cleanup_from_settings
from service.jobs.events import JobEvents
from service.jobs.recovery import RecoveryResult, recover_jobs
from service.jobs.redis_lock import RedisLock
from service.jobs.refs import ExternalRefs
from service.jobs.rq_queue import RqJobQueue
from service.jobs.runner import JobRunner
from service.jobs.store import JobStore

if TYPE_CHECKING:
    import redis

    from service.settings import ServiceSettings

logger = logging.getLogger("qgen.worker")

MAINTENANCE_TASK = "service.jobs.tasks.run_maintenance"
MAINTENANCE_INTERVAL_S = 600  # default of QGEN_MAINTENANCE_INTERVAL_S
MAINTENANCE_ID_PREFIX = "qgen-maintenance-"
MAINTENANCE_LOCK_KEY = "qgen:lock:maintenance"
RECOVERY_LOCK_KEY = "qgen:lock:recovery"
LOCK_TTL_S = 300


@dataclass
class WorkerContext:
    settings: "ServiceSettings"
    redis: "redis.Redis"
    store: JobStore
    queue: RqJobQueue
    refs: ExternalRefs
    runner: JobRunner
    events: Optional[JobEvents] = None
    wall_clock: Callable[[], float] = time.time


_context: Optional[WorkerContext] = None


def build_context(settings: "ServiceSettings", client: "redis.Redis", **runner_kwargs) -> WorkerContext:
    from service.webhooks.rq_delivery import RqWebhookNotifier

    store = JobStore.from_settings(settings)
    events = RqWebhookNotifier(client, settings)
    return WorkerContext(
        settings=settings,
        redis=client,
        store=store,
        queue=RqJobQueue(client, job_timeout_s=settings.job_timeout_s),
        refs=ExternalRefs(client),
        runner=JobRunner(store, settings, events=events, **runner_kwargs),
        events=events,
    )


def configure(context: Optional[WorkerContext]) -> None:
    global _context
    _context = context


def get_context() -> WorkerContext:
    global _context
    if _context is None:
        import redis

        from service.settings import ServiceSettings

        settings = ServiceSettings()
        _context = build_context(settings, redis.Redis.from_url(settings.redis_url))
    return _context


# --- RQ tasks -------------------------------------------------------------------------------


def run_job(job_id: str) -> Optional[str]:
    return get_context().runner.run(job_id)


def run_maintenance() -> Optional[CleanupResult]:
    context = get_context()
    try:
        with RedisLock(context.redis, MAINTENANCE_LOCK_KEY, ttl_s=LOCK_TTL_S) as lock:
            if not lock.acquired:
                logger.info("maintenance already running on another worker, skipped")
                return None
            result = run_cleanup_from_settings(context.store, context.settings, context.store.clock(), context.refs)
            recovery = recover_jobs(context.store, context.queue, events=context.events)
            if recovery.requeued or recovery.failed:
                logger.info("maintenance recovery requeued=%d failed=%d", len(recovery.requeued), len(recovery.failed))
            return result
    finally:
        schedule_maintenance(context)


# --- start of the worker --------------------------------------------------------------------


def recover_at_start(context: WorkerContext) -> Optional[RecoveryResult]:
    with RedisLock(context.redis, RECOVERY_LOCK_KEY, ttl_s=LOCK_TTL_S) as lock:
        if not lock.acquired:
            logger.info("recovery already running on another worker, skipped")
            return None
        result = recover_jobs(context.store, context.queue, events=context.events)
    logger.info(
        "recovery requeued=%d failed=%d skipped=%d", len(result.requeued), len(result.failed), len(result.skipped)
    )
    return result


def maintenance_job_id(now_s: float, interval_s: float = MAINTENANCE_INTERVAL_S) -> str:
    slot = int(now_s // interval_s) + 1
    return f"{MAINTENANCE_ID_PREFIX}{slot}"


def schedule_maintenance(context: WorkerContext) -> str:
    """Schedule the next maintenance run in the next slot (same id for every worker)."""
    now_s = context.wall_clock()
    interval_s = context.settings.maintenance_interval_s
    job_id = maintenance_job_id(now_s, interval_s)
    slot_start = (int(now_s // interval_s) + 1) * interval_s
    delay = max(1.0, slot_start - now_s)
    context.queue.queue.enqueue_in(
        timedelta(seconds=delay),
        MAINTENANCE_TASK,
        job_id=job_id,
        job_timeout=LOCK_TTL_S,
        result_ttl=0,
        failure_ttl=3600,
    )
    return job_id
