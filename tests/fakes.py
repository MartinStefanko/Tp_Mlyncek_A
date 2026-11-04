"""Scripted fakes of the LLM interface for unit tests. No network, no vendor library.

FakeLLM / FakeEmbedder take a list of steps, used one per call in order; when the list is
used up, `default` answers every further call. A step is:
    str                 FakeLLM: the answer text (may be invalid JSON)
    list of vectors     FakeEmbedder: the answer
    BaseException       raised (see the helpers below)
    callable            called with (prompt_or_texts, model); returns one of the above

Helpers for the errors the retry layer must handle:
    server_error(503)          retryable ProviderError
    client_error(404)          not retryable ProviderError (400, 401, 403, 404, ...)
    rate_limited(retry_after)  RateLimitError (HTTP 429)
    timed_out()                LLMTimeoutError (no real waiting)
"""

from __future__ import annotations

import hashlib
from collections import deque
from typing import Any, Iterable, Optional

from qgen.llm.base import LLMResponse
from qgen.llm.errors import LLMTimeoutError, ProviderError, RateLimitError

INVALID_JSON = "Sorry, here is no JSON {"


def server_error(status: int = 503) -> ProviderError:
    return ProviderError(f"fake server error {status}", retryable=True, status=status)


def client_error(status: int = 400) -> ProviderError:
    return ProviderError(f"fake client error {status}", retryable=False, status=status)


def rate_limited(retry_after: Optional[float] = None) -> RateLimitError:
    return RateLimitError("fake 429", retry_after=retry_after)


def timed_out() -> LLMTimeoutError:
    return LLMTimeoutError("fake timeout")


def _resolve(step: Any, arg: Any, model: str) -> Any:
    if callable(step) and not isinstance(step, BaseException):
        step = step(arg, model)
    if isinstance(step, BaseException):
        raise step
    return step


class FakeLLM:
    """LLMClient with scripted answers. `calls` lists (model, prompt) of every call."""

    def __init__(self, steps: Iterable[Any] = (), *, default: Any = "[]"):
        self.steps = deque(steps)
        self.default = default
        self.calls: list[tuple[str, str]] = []

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool = True,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        self.calls.append((model, prompt))
        step = self.steps.popleft() if self.steps else self.default
        text = _resolve(step, prompt, model)
        return LLMResponse(text=text, model=model, usage={"input_tokens": len(prompt)})


def hash_vector(text: str, dim: int = 8) -> list[float]:
    """Deterministic pseudo embedding of a text."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [b / 255.0 for b in digest[:dim]]


class FakeEmbedder:
    """EmbeddingClient with scripted answers; default: hash vectors. `calls` lists the inputs."""

    def __init__(self, steps: Iterable[Any] = (), *, default: Any = None):
        self.steps = deque(steps)
        self.default = default
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.steps:
            step = self.steps.popleft()
        elif self.default is not None:
            step = self.default
        else:
            return [hash_vector(t) for t in texts]
        return _resolve(step, texts, model)
