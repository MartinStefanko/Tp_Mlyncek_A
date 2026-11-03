"""Webhook sending with retries on the API's asyncio loop (mock only).

The real service replaces this with RQ (send_once + enqueue_in). The delivery logic
(body, headers, signature, retry plan) is in delivery.py and is shared.
"""

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional, Sequence

import httpx

from qgen.contract import WebhookPayload
from service.webhooks.delivery import (
    RETRY_DELAYS_S,
    WebhookDelivery,
    new_delivery,
    next_retry_delay,
    send_once_async,
)

logger = logging.getLogger("qgen.webhooks")


class AsyncioWebhookDispatcher:
    """submit() may be called from any thread; deliveries run on the loop given to start()."""

    def __init__(
        self,
        secret: str,
        timeout_s: float,
        *,
        delays: Sequence[float] = RETRY_DELAYS_S,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        unix_time: Callable[[], float] = time.time,
    ) -> None:
        self._secret = secret
        self._timeout_s = timeout_s
        self._delays = tuple(delays)
        self._transport = transport
        self._sleep = sleep
        self._unix_time = unix_time
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._client = httpx.AsyncClient(timeout=self._timeout_s, transport=self._transport)

    async def stop(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._client is not None:
            await self._client.aclose()
        self._client = None
        self._loop = None

    def submit(self, url: str, payload: WebhookPayload) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            logger.warning("webhook %s job_id=%s dropped: dispatcher not running", payload.event, payload.job_id)
            return
        delivery = new_delivery(url, payload)
        loop.call_soon_threadsafe(self._spawn, delivery)

    def _spawn(self, delivery: WebhookDelivery) -> None:
        task = asyncio.ensure_future(self.deliver(delivery))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def deliver(self, delivery: WebhookDelivery) -> bool:
        """First attempt now, then retries after the configured delays. True when delivered."""
        assert self._client is not None, "dispatcher not started"
        attempt = 1
        while True:
            timestamp = int(self._unix_time())
            if await send_once_async(self._client, delivery, self._secret, timestamp, attempt):
                return True
            delay = next_retry_delay(attempt, self._delays)
            if delay is None:
                logger.warning(
                    "webhook %s job_id=%s delivery=%s: giving up after %d attempts",
                    delivery.event,
                    delivery.job_id,
                    delivery.delivery_id,
                    attempt,
                )
                return False
            await self._sleep(delay)
            attempt += 1

    async def drain(self) -> None:
        """Wait until all running deliveries finish (tests)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
