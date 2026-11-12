"""Worker: processes jobs from the RQ queue "qgen".

    python -m service.worker

Start (kap. 3):
    1. check_config – a missing key / base_url stops the start (exit code 1);
       check_service_config – QGEN_ENV=production without QGEN_API_KEYS (or with webhooks
       on and without QGEN_WEBHOOK_SECRET) stops the start too (service.startup)
    2. recovery: queued / running jobs that are not in the queue are queued again
    3. the periodic maintenance (cleanup + recovery) is scheduled
    4. RQ worker with the built-in scheduler (--with-scheduler: maintenance); RQ's own
       maintenance interval = QGEN_MAINTENANCE_INTERVAL_S

Webhooks are only queued here (queue "qgen-webhooks"); the webhook worker
(service.webhook_worker, its own container) delivers them and plans their retries.

RQ forks a work horse per job, so the worker runs only on Linux (Docker); tests use
SimpleWorker.
"""

import logging
import sys
from typing import Optional

from qgen.llm.errors import LLMConfigError
from qgen.llm.factory import check_config
from service.jobs import tasks
from service.jobs.rq_queue import rq_worker_options
from service.settings import ServiceSettings
from service.startup import StartupConfigError, check_service_config

logger = logging.getLogger("qgen.worker")


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = ServiceSettings()
    try:
        check_config(settings)
    except LLMConfigError as exc:
        logger.error("worker not started, configuration error: %s", exc)
        return 1
    try:
        check_service_config(settings, "worker")
    except StartupConfigError:
        return 1

    import redis
    from rq import Worker

    client = redis.Redis.from_url(settings.redis_url)
    context = tasks.build_context(settings, client)
    tasks.configure(context)
    tasks.recover_at_start(context)
    tasks.schedule_maintenance(context)
    logger.info("worker started queue=%s job_timeout_s=%g", context.queue.queue.name, settings.job_timeout_s)
    options = rq_worker_options(settings.maintenance_interval_s)  # scheduler lock of a killed worker
    Worker([context.queue.queue], connection=client, **options).work(with_scheduler=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
