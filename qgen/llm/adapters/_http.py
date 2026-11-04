"""HTTP error translation shared by the openai_compatible and anthropic adapters.

Same rules as the Gemini adapter (see qgen.llm.errors):
    429             -> RateLimitError, retry_after from the response headers
    408, 5xx        -> ProviderError(retryable=True)      (also Anthropic 529 "overloaded")
    other 4xx       -> ProviderError(retryable=False)     e.g. 400, 401, 403, 404
The message has only the provider name and the status, never the response body (it may
quote the prompt).
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Optional

from qgen.llm.errors import ProviderError, RateLimitError


def _header(headers: Any, name: str) -> Optional[str]:
    if headers is None:
        return None
    try:
        value = headers.get(name)
    except Exception:  # unusual header container
        return None
    return None if value is None else str(value).strip()


def parse_retry_after(headers: Any, now: Optional[Callable[[], datetime]] = None) -> Optional[float]:
    """Seconds to wait from `retry-after-ms` (OpenAI) or `Retry-After` (seconds or HTTP date)."""
    millis = _header(headers, "retry-after-ms")
    if millis:
        try:
            return max(0.0, float(millis) / 1000.0)
        except ValueError:
            pass
    value = _header(headers, "retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        moment = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    current = now() if now else datetime.now(timezone.utc)
    return max(0.0, (moment - current).total_seconds())


def status_error(provider: str, status: int, headers: Any = None) -> ProviderError:
    message = f"{provider} API error {status}"
    if status == 429:
        return RateLimitError(message, retry_after=parse_retry_after(headers))
    retryable = status == 408 or status >= 500
    return ProviderError(message, retryable=retryable, status=status)
