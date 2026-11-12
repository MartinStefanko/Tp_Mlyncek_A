"""Webhook worker: delivers webhooks from the RQ queue "qgen-webhooks".

    python -m service.webhook_worker

Its own container (same image as the API and the worker). It serves only the webhook queue,
so a delivery never waits for a running job of the pipeline. The built-in scheduler
(--with-scheduler) moves the retries (enqueue_in 10 / 60 / 300 s) back to the queue.
No LLM configuration, no recovery and no maintenance here (that is service.worker).
"""

import logging
import sys
from typing import Optional

from service.jobs.rq_queue import rq_worker_options
from service.settings import ServiceSettings
from service.startup import StartupConfigError, check_service_config
from service.webhooks import rq_delivery

logger = logging.getLogger("qgen.webhooks")


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = ServiceSettings()
    try:
        check_service_config(settings, "webhook worker")  # empty secret with webhooks on
    except StartupConfigError:
        return 1
    if not settings.allowed_webhook_hosts:
        logger.info("QGEN_WEBHOOK_ALLOWED_HOSTS is empty: webhooks are off, nothing to deliver")

    import redis
    from rq import Worker

    client = redis.Redis.from_url(settings.redis_url)
    context = rq_delivery.build_context(settings, client)
    rq_delivery.configure(context)
    logger.info("webhook worker started queue=%s timeout_s=%g", context.queue.name, settings.webhook_timeout_s)
    # Same as the job worker: after a crash the dead worker's scheduler lock is taken over
    # soon, then the planned retries go on (service.jobs.rq_queue.rq_worker_options).
    options = rq_worker_options(settings.maintenance_interval_s)
    Worker([context.queue], connection=client, **options).work(with_scheduler=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
