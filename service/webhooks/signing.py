"""Webhook signature (Príloha A – Webhook, Podpis).

    X-QGen-Signature: sha256=<hex HMAC-SHA256(QGEN_WEBHOOK_SECRET, f"{timestamp}.{body}")>

`timestamp` is Unix time in seconds (X-QGen-Timestamp), `body` the exact bytes sent.
"""

import hashlib
import hmac

HEADER_EVENT = "X-QGen-Event"
HEADER_DELIVERY = "X-QGen-Delivery"
HEADER_TIMESTAMP = "X-QGen-Timestamp"
HEADER_SIGNATURE = "X-QGen-Signature"

SIGNATURE_PREFIX = "sha256="


def sign(secret: str, timestamp: int, body: bytes) -> str:
    message = str(timestamp).encode("ascii") + b"." + body
    digest = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return SIGNATURE_PREFIX + digest


def verify(secret: str, timestamp: str, body: bytes, signature: str, *, now: int, max_age_s: int = 300) -> bool:
    """Receiver-side check (what Laravel does); used by tests and tools."""
    try:
        ts = int(timestamp)
    except ValueError:
        return False
    if abs(now - ts) >= max_age_s:
        return False
    return hmac.compare_digest(sign(secret, ts, body), signature)
