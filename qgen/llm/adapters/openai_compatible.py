"""Adapter for servers with the OpenAI API (official `openai` package).

Covers OpenAI, Mistral, Groq, OpenRouter and local servers (Ollama, vLLM, LM Studio); only
base_url changes:
    generate  POST {base_url}/chat/completions  (QGEN_LLM_BASE_URL, e.g. http://llm:8001/v1)
    embed     POST {base_url}/embeddings        (QGEN_EMBED_BASE_URL)
Keys: QGEN_LLM_API_KEY for generate, QGEN_EMBED_API_KEY for embed (empty = QGEN_LLM_API_KEY).
A local server often needs no key, but the SDK does not accept an empty one, so
PLACEHOLDER_API_KEY is sent instead.

generate sends:
    messages         the prompt as one user message
    temperature      always (QGEN_LLM_TEMPERATURE)
    max_tokens       only when the caller gives it (question generation)
    response_format  {"type": "json_object"} when json_mode=True. Since step 13 every prompt
                     asks for a JSON object ({"items": [...]} or the language object), and
                     every prompt contains the word "JSON", as OpenAI requires.
    stream          True with QGEN_LLM_STREAM (default), with stream_options include_usage
safe_load_json and one repeated call stay in qgen.llm.retry. An answer that ended at the
token limit (finish_reason "length") raises OutputTruncatedError (not retried).

Degenerated answer (qgen.llm.degenerate): a local model in JSON mode can get stuck and write
only whitespace until the token limit (Gemma 4 on 01.pdf: 32 768 tokens, minutes per call).
With streaming the answer is read chunk by chunk and the call is stopped as soon as
QGEN_LLM_MAX_WHITESPACE_RUN whitespace characters came in a row; without streaming an answer
cut off at the limit that ends in such a run is recognised afterwards. Both raise
DegenerateOutputError, which qgen.llm.retry retries.

Timeout: QGEN_LLM_TIMEOUT_S (seconds) as the client timeout. With streaming the SDK timeout
applies to each read, so the adapter also stops a stream that runs longer than
QGEN_LLM_TIMEOUT_S in total (LLMTimeoutError). The SDK's own retries are off
(max_retries=0); qgen.llm.retry retries.

Error translation (see qgen.llm.adapters._http):
    APIStatusError 429        -> RateLimitError (retry-after-ms / Retry-After header)
    APIStatusError 408, 5xx   -> ProviderError(retryable=True)
    APIStatusError other 4xx  -> ProviderError(retryable=False)
    APITimeoutError           -> LLMTimeoutError
    APIConnectionError        -> ProviderError(retryable=True)   network
    httpx(2) errors while a stream is read: TimeoutException -> LLMTimeoutError,
                              TransportError -> ProviderError(retryable=True)
    APIError in a stream      -> ProviderError(retryable=True)   error event from the server
"""

from __future__ import annotations

import time
from typing import Any, Optional

import openai

from qgen.llm.adapters._http import status_error
from qgen.llm.base import LLMResponse
from qgen.llm.degenerate import DEFAULT_MAX_WHITESPACE_RUN, WhitespaceRunGuard, cut_off_error
from qgen.llm.errors import DegenerateOutputError, LLMConfigError, LLMTimeoutError, ProviderError

try:  # the openai SDK uses httpx2 (a fork with its own exception classes) when it is installed
    import httpx2 as _httpx
except ImportError:  # pragma: no cover
    import httpx as _httpx

PROVIDER = "OpenAI-compatible"
PLACEHOLDER_API_KEY = "no-key"  # sent when the key is empty (local server)


def build_client(base_url: Optional[str], api_key: str, timeout_s: Optional[float]) -> Any:
    if not base_url:
        raise LLMConfigError("openai_compatible needs a base_url (QGEN_LLM_BASE_URL / QGEN_EMBED_BASE_URL).")
    kwargs: dict[str, Any] = {"base_url": base_url, "api_key": api_key or PLACEHOLDER_API_KEY, "max_retries": 0}
    if timeout_s:
        kwargs["timeout"] = float(timeout_s)
    return openai.OpenAI(**kwargs)


def translate_error(error: Exception) -> Exception:
    """SDK exception -> ProviderError. Anything unknown is returned unchanged."""
    if isinstance(error, ProviderError):
        return error
    if isinstance(error, openai.APIStatusError):
        return status_error(PROVIDER, error.status_code, getattr(error.response, "headers", None))
    if isinstance(error, (openai.APITimeoutError, TimeoutError)):  # APITimeoutError before APIConnectionError
        return LLMTimeoutError(f"{PROVIDER} call timed out ({type(error).__name__})")
    if isinstance(error, (openai.APIConnectionError, ConnectionError)):
        return ProviderError(f"{PROVIDER} network error ({type(error).__name__})", retryable=True)
    if isinstance(error, _httpx.TimeoutException):  # raised while a stream is read
        return LLMTimeoutError(f"{PROVIDER} stream timed out ({type(error).__name__})")
    if isinstance(error, _httpx.TransportError):
        return ProviderError(f"{PROVIDER} network error in stream ({type(error).__name__})", retryable=True)
    if isinstance(error, openai.APIError):  # error event inside a stream (no HTTP status)
        return ProviderError(f"{PROVIDER} error in stream ({type(error).__name__})", retryable=True)
    return error


def _call(fn, **kwargs):
    try:
        return fn(**kwargs)
    except Exception as error:
        translated = translate_error(error)
        if translated is error:
            raise
        raise translated from error


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    return getattr(message, "content", None) or ""


def _truncated(response: Any) -> bool:
    choices = getattr(response, "choices", None) or []
    return bool(choices) and getattr(choices[0], "finish_reason", None) == "length"


def _usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    result = {}
    for key, attr in (
        ("input_tokens", "prompt_tokens"),
        ("output_tokens", "completion_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        value = getattr(usage, attr, None)
        if isinstance(value, int):
            result[key] = value
    return result


class OpenAICompatLLM:
    """LLMClient over /v1/chat/completions. One HTTP call per `generate`.

    stream=True: the answer comes as server-sent events and a degenerated answer is stopped
    early (the factory passes QGEN_LLM_STREAM; the default here is False for a plain client).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: str = "",
        timeout_s: Optional[float] = None,
        client: Any = None,
        stream: bool = False,
        max_whitespace_run: int = DEFAULT_MAX_WHITESPACE_RUN,
    ):
        self._client = client if client is not None else build_client(base_url, api_key, timeout_s)
        self._stream = stream
        self._timeout_s = float(timeout_s) if timeout_s else None
        self._max_whitespace_run = max_whitespace_run

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool = True,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self._stream:
            return self._generate_stream(kwargs, model)
        response = _call(self._client.chat.completions.create, **kwargs)
        text = _response_text(response)
        if _truncated(response):
            raise cut_off_error(PROVIDER, "finish_reason length", text, self._max_whitespace_run)
        answered_by = getattr(response, "model", None)
        return LLMResponse(
            text=text,
            model=answered_by if isinstance(answered_by, str) and answered_by else model,
            usage=_usage(response),
        )

    def _generate_stream(self, kwargs: dict[str, Any], model: str) -> LLMResponse:
        deadline = time.monotonic() + self._timeout_s if self._timeout_s else None
        guard = WhitespaceRunGuard(self._max_whitespace_run)
        parts: list[str] = []
        finish_reason = None
        answered_by = None
        usage: dict[str, int] = {}
        stream = _call(
            self._client.chat.completions.create, stream=True, stream_options={"include_usage": True}, **kwargs
        )
        try:
            iterator = iter(stream)
            while True:
                try:
                    chunk = next(iterator)
                except StopIteration:
                    break
                except Exception as error:
                    translated = translate_error(error)
                    if translated is error:
                        raise
                    raise translated from error
                answered_by = answered_by or getattr(chunk, "model", None)
                if getattr(chunk, "usage", None) is not None:
                    usage = _usage(chunk)
                for choice in getattr(chunk, "choices", None) or []:
                    delta = getattr(choice, "delta", None)
                    text = getattr(delta, "content", None) or ""
                    if text:
                        parts.append(text)
                        if guard.feed(text):
                            raise DegenerateOutputError(
                                f"{PROVIDER} answer degenerated into whitespace "
                                f"({guard.run} characters in a row, stream stopped)"
                            )
                    if getattr(choice, "finish_reason", None):
                        finish_reason = choice.finish_reason
                if deadline is not None and time.monotonic() > deadline:
                    raise LLMTimeoutError(f"{PROVIDER} stream ran longer than {self._timeout_s:g} s")
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        text = "".join(parts)
        if finish_reason == "length":
            raise cut_off_error(PROVIDER, "finish_reason length", text, self._max_whitespace_run)
        return LLMResponse(
            text=text,
            model=answered_by if isinstance(answered_by, str) and answered_by else model,
            usage=usage,
        )


class OpenAICompatEmbedder:
    """EmbeddingClient over /v1/embeddings. One HTTP call per `embed`."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: str = "",
        timeout_s: Optional[float] = None,
        client: Any = None,
    ):
        self._client = client if client is not None else build_client(base_url, api_key, timeout_s)

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        if not texts:
            return []
        # encoding_format="float": some local servers do not support the SDK's base64 default.
        response = _call(self._client.embeddings.create, model=model, input=list(texts), encoding_format="float")
        data = list(getattr(response, "data", None) or [])
        if len(data) != len(texts):
            raise ProviderError("Number of embeddings does not match the number of inputs.", retryable=False)
        data.sort(key=lambda item: getattr(item, "index", 0))
        vectors = []
        for item in data:
            values = getattr(item, "embedding", None)
            if values is None:
                raise ProviderError("Embedding without values.", retryable=False)
            vectors.append([float(v) for v in values])
        return vectors
