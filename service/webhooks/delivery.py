"""One webhook delivery: body, headers, one attempt, retry plan.

This module does not schedule anything. The mock retries with asyncio
(service/webhooks/asyncio_dispatcher.py); the real service's webhook worker calls
post_once() from an RQ job and plans the next attempt with enqueue_in(next_retry_delay(...))
(service/webhooks/rq_delivery.py).

Per delivery the body, X-QGen-Delivery and occurred_at stay the same for every
attempt; X-QGen-Timestamp and the signature are computed again for each attempt.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Optional, Sequence

import httpx

from qgen.contract import WebhookPayload
from service.webhooks.signing import HEADER_DELIVERY, HEADER_EVENT, HEADER_SIGNATURE, HEADER_TIMESTAMP, sign

logger = logging.getLogger("qgen.webhooks")

# Waits before the 2nd, 3rd and 4th attempt (Príloha A – Opakovanie). Then give up.
RETRY_DELAYS_S: tuple[float, ...] = (10, 60, 300)


@dataclass(frozen=True)
class WebhookDelivery:
    url: str
    event: str
    job_id: str
    delivery_id: str
    body: bytes


def new_delivery(url: str, payload: WebhookPayload) -> WebhookDelivery:
    return WebhookDelivery(
        url=url,
        event=payload.event,
        job_id=payload.job_id,
        delivery_id=str(uuid.uuid4()),
        body=payload.model_dump_json().encode("utf-8"),
    )


def build_headers(delivery: WebhookDelivery, secret: str, timestamp: int) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        HEADER_EVENT: delivery.event,
        HEADER_DELIVERY: delivery.delivery_id,
        HEADER_TIMESTAMP: str(timestamp),
        HEADER_SIGNATURE: sign(secret, timestamp, delivery.body),
    }


def next_retry_delay(attempts_done: int, delays: Sequence[float] = RETRY_DELAYS_S) -> Optional[float]:
    """Delay before the next attempt after `attempts_done` failed attempts, or None to give up."""
    if attempts_done < 1 or attempts_done > len(delays):
        return None
    return delays[attempts_done - 1]


def _log_result(delivery: WebhookDelivery, attempt: int, outcome: str) -> None:
    logger.info(
        "webhook %s job_id=%s delivery=%s attempt=%d: %s",
        delivery.event,
        delivery.job_id,
        delivery.delivery_id,
        attempt,
        outcome,
    )


@dataclass(frozen=True)
class AttemptResult:
    """Outcome of one attempt: the HTTP status, or the type of the transport error."""

    ok: bool
    status_code: Optional[int] = None
    error_type: Optional[str] = None

    @property
    def outcome(self) -> str:
        return f"HTTP {self.status_code}" if self.status_code is not None else str(self.error_type)


def post_once(
    client: httpx.Client, delivery: WebhookDelivery, secret: str, timestamp: int, attempt: int = 1
) -> AttemptResult:
    """One synchronous attempt (the real service's webhook worker). The timeout is set on the client."""
    try:
        response = client.post(delivery.url, content=delivery.body, headers=build_headers(delivery, secret, timestamp))
    except httpx.HTTPError as exc:
        result = AttemptResult(ok=False, error_type=type(exc).__name__)
    else:
        result = AttemptResult(ok=response.is_success, status_code=response.status_code)
    _log_result(delivery, attempt, result.outcome)
    return result


def send_once(client: httpx.Client, delivery: WebhookDelivery, secret: str, timestamp: int, attempt: int = 1) -> bool:
    """One synchronous attempt; True on 2xx. The timeout is set on the client."""
    return post_once(client, delivery, secret, timestamp, attempt).ok


async def send_once_async(
    client: httpx.AsyncClient, delivery: WebhookDelivery, secret: str, timestamp: int, attempt: int = 1
) -> bool:
    """One asynchronous attempt; True on 2xx. The timeout is set on the client."""
    try:
        response = await client.post(delivery.url, content=delivery.body, headers=build_headers(delivery, secret, timestamp))
    except httpx.HTTPError as exc:
        _log_result(delivery, attempt, type(exc).__name__)
        return False
    _log_result(delivery, attempt, f"HTTP {response.status_code}")
    return response.is_success
