"""Errors of the LLM layer.

Adapters translate vendor exceptions into `ProviderError` (or a subclass). Its `retryable`
flag tells `qgen.llm.retry` whether to try again:
    retried:      HTTP 429 (RateLimitError), 408 and 5xx, timeout (LLMTimeoutError),
                  network errors
    not retried:  HTTP 400, 401, 403, 404 and every other 4xx; any exception that is not a
                  ProviderError (also not one of the built-in TimeoutError / ConnectionError)

`LLMError` is what the pipeline sees: a call failed after all attempts, failed with an
error that is not retried, or returned invalid JSON also after the repeated call. It must
not be caught in the pipeline; the service turns it into a failed job with code
`llm_error`. Its message never contains the prompt or the response text.
"""

from typing import Optional


class ProviderError(Exception):
    """Error of one provider call, raised by an adapter."""

    def __init__(self, message: str, *, retryable: bool, status: Optional[int] = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class RateLimitError(ProviderError):
    """HTTP 429. `retry_after` in seconds, when the provider said how long to wait."""

    def __init__(self, message: str, *, retry_after: Optional[float] = None):
        super().__init__(message, retryable=True, status=429)
        self.retry_after = retry_after


class LLMTimeoutError(ProviderError):
    """The client-side timeout (QGEN_LLM_TIMEOUT_S) of one call ran out."""

    def __init__(self, message: str = "LLM call timed out"):
        super().__init__(message, retryable=True, status=None)


class LLMError(Exception):
    """An LLM or embedding call failed for good. Job error code: llm_error."""

    code = "llm_error"

    def __init__(
        self,
        message: str,
        *,
        role: str,
        model: str,
        cause: Optional[BaseException] = None,
        invalid_json: bool = False,
        attempts: int = 0,
    ):
        super().__init__(f"{message} (role={role}, model={model}, attempts={attempts})")
        self.role = role
        self.model = model
        self.cause = cause
        self.invalid_json = invalid_json
        self.attempts = attempts


def is_retryable(error: BaseException) -> bool:
    if isinstance(error, ProviderError):
        return error.retryable
    # Built-in timeout / network errors (e.g. from a fake or a plain socket client).
    return isinstance(error, (TimeoutError, ConnectionError))
