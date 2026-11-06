"""Anthropic (Claude) adapter over the Messages API (official `anthropic` package).

Generation only. Anthropic has no embedding API; qgen.llm.factory rejects
QGEN_EMBED_PROVIDER=anthropic at start, so embeddings come from another provider
(e.g. QGEN_EMBED_PROVIDER=gemini).

generate sends:
    messages    the prompt as one user message
    max_tokens  required by the Messages API: the caller's value, else QGEN_LLM_MAX_TOKENS
    system      JSON_SYSTEM_INSTRUCTION when json_mode=True. The Messages API has no JSON
                mode like OpenAI's response_format; the answer is still parsed by
                safe_load_json, with one repeated call in qgen.llm.retry.
    temperature NEVER sent (decision of step 13): newer Claude models may reject sampling
                parameters, and the installed SDK (anthropic 2.4) has no temperature
                parameter in messages.create anyway. QGEN_LLM_TEMPERATURE does not apply
                here; it is also not sent through extra_body.
The answer text is the concatenation of all text blocks; usage from response.usage.
An answer that ended at the token limit (stop_reason "max_tokens") raises
OutputTruncatedError (not retried).

Timeout: QGEN_LLM_TIMEOUT_S (seconds) as the client timeout. The SDK's own retries are off
(max_retries=0); qgen.llm.retry retries.

Error translation (see qgen.llm.adapters._http):
    APIStatusError 429        -> RateLimitError (Retry-After header)
    APIStatusError 408, 5xx   -> ProviderError(retryable=True)    incl. 529 overloaded
    APIStatusError other 4xx  -> ProviderError(retryable=False)
    APITimeoutError           -> LLMTimeoutError
    APIConnectionError        -> ProviderError(retryable=True)    network
"""

from __future__ import annotations

from typing import Any, Optional

import anthropic

from qgen.llm.adapters._http import status_error
from qgen.llm.base import LLMResponse
from qgen.llm.errors import LLMConfigError, LLMTimeoutError, OutputTruncatedError, ProviderError

PROVIDER = "Anthropic"
DEFAULT_MAX_TOKENS = 8192  # same as the default of QGEN_LLM_MAX_TOKENS
JSON_SYSTEM_INSTRUCTION = (
    "Respond only with valid JSON. Do not add any text, explanation or Markdown code fence "
    "before or after the JSON."
)


def build_client(api_key: str, timeout_s: Optional[float]) -> Any:
    if not api_key:
        raise LLMConfigError("ANTHROPIC_API_KEY is not set.")
    kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 0}
    if timeout_s:
        kwargs["timeout"] = float(timeout_s)
    return anthropic.Anthropic(**kwargs)


def translate_error(error: Exception) -> Exception:
    """SDK exception -> ProviderError. Anything unknown is returned unchanged."""
    if isinstance(error, ProviderError):
        return error
    if isinstance(error, anthropic.APIStatusError):
        return status_error(PROVIDER, error.status_code, getattr(error.response, "headers", None))
    if isinstance(error, (anthropic.APITimeoutError, TimeoutError)):  # before APIConnectionError
        return LLMTimeoutError(f"{PROVIDER} call timed out ({type(error).__name__})")
    if isinstance(error, (anthropic.APIConnectionError, ConnectionError)):
        return ProviderError(f"{PROVIDER} network error ({type(error).__name__})", retryable=True)
    return error


def _response_text(response: Any) -> str:
    parts = []
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts)


def _usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    result = {}
    for key in ("input_tokens", "output_tokens"):
        value = getattr(usage, key, None)
        if isinstance(value, int):
            result[key] = value
    if "input_tokens" in result and "output_tokens" in result:
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result


class AnthropicLLM:
    """LLMClient over the Messages API. One HTTP call per `generate`."""

    def __init__(
        self,
        api_key: str = "",
        timeout_s: Optional[float] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Any = None,
    ):
        self._client = client if client is not None else build_client(api_key, timeout_s)
        self._max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool = True,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        # temperature is accepted by the interface but intentionally not sent (see above).
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_mode:
            kwargs["system"] = JSON_SYSTEM_INSTRUCTION
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as error:
            translated = translate_error(error)
            if translated is error:
                raise
            raise translated from error
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise OutputTruncatedError(f"{PROVIDER} answer cut off (stop_reason max_tokens)")
        answered_by = getattr(response, "model", None)
        return LLMResponse(
            text=_response_text(response),
            model=answered_by if isinstance(answered_by, str) and answered_by else model,
            usage=_usage(response),
        )
