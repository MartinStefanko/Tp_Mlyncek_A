"""Webhooks of the real service over RQ: queue "qgen-webhooks", served by the webhook worker.

    API / worker:  job.json written → RqWebhookNotifier.notify → enqueue deliver(…, attempt=1)
    webhook worker (python -m service.webhook_worker, --with-scheduler):
        deliver: host allowed? → POST (new timestamp + signature) → 2xx: done
                 otherwise enqueue_in(10 | 60 | 300 s, attempt + 1); after the 4th attempt a
                 log line (job_id, event, status) and nothing else – the job state never changes.

Why a queue of its own: a job of the pipeline runs for minutes on queue "qgen". A webhook on
the same queue would wait until the job ends. The webhook worker serves only this queue, so
the first attempt goes out at once and its built-in scheduler moves the retries back to it.

Per delivery the body (incl. occurred_at) and X-QGen-Delivery are fixed when the event is
queued; X-QGen-Timestamp and the signature are computed for every attempt (delivery.py).
"""

import logging
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Callable, Optional

import httpx
from rq import Queue

from qgen.contract import WebhookEvent, WebhookPayload
from service.jobs.models import JobRecord
from service.jobs.rq_queue import queue_has_live_worker
from service.webhooks.delivery import RETRY_DELAYS_S, WebhookDelivery, new_delivery, next_retry_delay, post_once
from service.webhooks.hosts import callback_host_allowed

if TYPE_CHECKING:
    import redis

    from service.settings import ServiceSettings

logger = logging.getLogger("qgen.webhooks")

WEBHOOK_QUEUE_NAME = "qgen-webhooks"
DELIVER_TASK = "service.webhooks.rq_delivery.deliver"
JOB_TIMEOUT_EXTRA_S = 30  # RQ kills a stuck attempt after the HTTP timeout + this
FAILURE_TTL_S = 24 * 3600


def payload_from_record(event: WebhookEvent, record: JobRecord) -> WebhookPayload:
    """Body of the webhook: the state of the job (as in GET /v1/jobs/{id}), never the result."""
    return WebhookPayload(
        event=event,
        job_id=record.job_id,
        external_ref=record.external_ref,
        status=record.status,
        stage=record.stage,
        progress=record.progress,
        error=record.error,
        occurred_at=record.updated_at,
    )


def rq_job_timeout(settings: "ServiceSettings") -> int:
    return int(settings.webhook_timeout_s) + JOB_TIMEOUT_EXTRA_S


# --- producer (API and worker) ----------------------------------------------------------------


class RqWebhookNotifier:
    """JobEvents over the RQ queue "qgen-webhooks"."""

    def __init__(self, connection: "redis.Redis", settings: "ServiceSettings", *, name: str = WEBHOOK_QUEUE_NAME) -> None:
        self.queue = Queue(name, connection=connection)
        self.job_timeout_s = rq_job_timeout(settings)

    def notify(self, event: WebhookEvent, record: JobRecord) -> None:
        url = record.options.callback_url
        if url is None:
            return
        delivery = new_delivery(str(url), payload_from_record(event, record))
        try:
            enqueue_attempt(self.queue, delivery, 1, job_timeout_s=self.job_timeout_s)
        except Exception as exc:  # noqa: BLE001 – a delivery problem never changes the job
            logger.error(
                "webhook %s job_id=%s not queued error=%s", event, record.job_id, type(exc).__name__
            )

    def has_live_worker(self) -> bool:
        return queue_has_live_worker(self.queue)


def enqueue_attempt(
    queue: Queue, delivery: WebhookDelivery, attempt: int, *, job_timeout_s: int, delay_s: Optional[float] = None
) -> None:
    kwargs: dict[str, Any] = {"job_timeout": job_timeout_s, "result_ttl": 0, "failure_ttl": FAILURE_TTL_S}
    args = (asdict(delivery), attempt)
    if delay_s is None:
        queue.enqueue(DELIVER_TASK, *args, **kwargs)
    else:
        queue.enqueue_in(timedelta(seconds=delay_s), DELIVER_TASK, *args, **kwargs)


# --- consumer (webhook worker) ----------------------------------------------------------------


@dataclass
class DeliveryContext:
    settings: "ServiceSettings"
    queue: Queue
    client: httpx.Client
    unix_time: Callable[[], float] = time.time
    delays: tuple[float, ...] = RETRY_DELAYS_S


_context: Optional[DeliveryContext] = None


def build_context(settings: "ServiceSettings", connection: "redis.Redis", **kwargs: Any) -> DeliveryContext:
    transport = kwargs.pop("transport", None)
    return DeliveryContext(
        settings=settings,
        queue=Queue(WEBHOOK_QUEUE_NAME, connection=connection),
        client=httpx.Client(timeout=settings.webhook_timeout_s, transport=transport),
        **kwargs,
    )


def configure(context: Optional[DeliveryContext]) -> None:
    global _context
    _context = context


def get_context() -> DeliveryContext:
    global _context
    if _context is None:
        import redis

        from service.settings import ServiceSettings

        settings = ServiceSettings()
        _context = build_context(settings, redis.Redis.from_url(settings.redis_url))
    return _context


def deliver(fields: dict[str, Any], attempt: int) -> bool:
    """RQ task: one attempt of one delivery. True when delivered."""
    context = get_context()
    delivery = WebhookDelivery(**fields)
    settings = context.settings
    if not callback_host_allowed(delivery.url, settings.allowed_webhook_hosts):
        logger.warning(
            "webhook %s job_id=%s delivery=%s not sent: host is not in QGEN_WEBHOOK_ALLOWED_HOSTS",
            delivery.event,
            delivery.job_id,
            delivery.delivery_id,
        )
        return False
    timestamp = int(context.unix_time())
    result = post_once(context.client, delivery, settings.webhook_secret.get_secret_value(), timestamp, attempt)
    if result.ok:
        return True
    delay = next_retry_delay(attempt, context.delays)
    if delay is None:
        logger.warning(
            "webhook gave up job_id=%s event=%s delivery=%s attempts=%d last=%s",
            delivery.job_id,
            delivery.event,
            delivery.delivery_id,
            attempt,
            result.outcome,
        )
        return False
    enqueue_attempt(context.queue, delivery, attempt + 1, job_timeout_s=rq_job_timeout(settings), delay_s=delay)
    return False
