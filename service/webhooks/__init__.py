"""Webhooks service → backend ECEH (Príloha A – Webhook)."""

from service.webhooks.delivery import (
    RETRY_DELAYS_S,
    AttemptResult,
    WebhookDelivery,
    build_headers,
    new_delivery,
    next_retry_delay,
    post_once,
    send_once,
    send_once_async,
)
from service.webhooks.hosts import callback_host_allowed
from service.webhooks.signing import sign, verify

__all__ = [
    "RETRY_DELAYS_S",
    "AttemptResult",
    "WebhookDelivery",
    "build_headers",
    "callback_host_allowed",
    "new_delivery",
    "next_retry_delay",
    "post_once",
    "send_once",
    "send_once_async",
    "sign",
    "verify",
]
