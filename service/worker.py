"""Worker: processes jobs from the RQ queue "qgen".

    python -m service.worker

Start (kap. 3):
    1. check_config – a missing key / base_url stops the start (exit code 1)
    2. recovery: queued / running jobs that are not in the queue are queued again
    3. the periodic maintenance (cleanup + recovery) is scheduled
    4. RQ worker with the built-in scheduler (--with-scheduler: maintenance, later the
       webhook retries of step 11)

RQ forks a work horse per job, so the worker runs only on Linux (Docker); tests use
SimpleWorker.
"""

import logging
import sys
from typing import Optional

from qgen.llm.errors import LLMConfigError
from qgen.llm.factory import check_config
from service.jobs import tasks
from service.settings import ServiceSettings

logger = logging.getLogger("qgen.worker")


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = ServiceSettings()
    try:
        check_config(settings)
    except LLMConfigError as exc:
        logger.error("worker not started, configuration error: %s", exc)
        return 1

    import redis
    from rq import Worker

    client = redis.Redis.from_url(settings.redis_url)
    context = tasks.build_context(settings, client)
    tasks.configure(context)
    tasks.recover_at_start(context)
    tasks.schedule_maintenance(context)
    logger.info("worker started queue=%s job_timeout_s=%g", context.queue.queue.name, settings.job_timeout_s)
    Worker([context.queue.queue], connection=client).work(with_scheduler=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
