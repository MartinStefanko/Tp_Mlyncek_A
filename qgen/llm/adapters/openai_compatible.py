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
    temperature      always (there is no baseline recorded with this adapter)
    max_tokens       only when the caller gives it
    response_format  NOT sent yet (JSON_MODE_ENABLED = False, decision of step 7).
safe_load_json and one repeated call stay in qgen.llm.retry.

Open point (step 7), same as the Gemini adapter: json_mode is not sent. OpenAI's
{"type": "json_object"} allows only a JSON object at the top level, but 13 of the 14
pipeline prompts ask for a JSON array (only language detection asks for an object), and
qgen.llm.retry always calls with json_mode=True. With the mode on, the pipeline would get
{"items": [...]} instead of a list. The mapping json_mode -> response_format is ready
below; switch JSON_MODE_ENABLED on together with prompts that return an object (and a new
baseline). OpenAI also requires the word "JSON" in the messages; every prompt has it.

Timeout: QGEN_LLM_TIMEOUT_S (seconds) as the client timeout. The SDK's own retries are off
(max_retries=0); qgen.llm.retry retries.

Error translation (see qgen.llm.adapters._http):
    APIStatusError 429        -> RateLimitError (retry-after-ms / Retry-After header)
    APIStatusError 408, 5xx   -> ProviderError(retryable=True)
    APIStatusError other 4xx  -> ProviderError(retryable=False)
    APITimeoutError           -> LLMTimeoutError
    APIConnectionError        -> ProviderError(retryable=True)   network
"""

from __future__ import annotations

from typing import Any, Optional

import openai

from qgen.llm.adapters._http import status_error
from qgen.llm.base import LLMResponse
from qgen.llm.errors import LLMConfigError, LLMTimeoutError, ProviderError

PROVIDER = "OpenAI-compatible"
PLACEHOLDER_API_KEY = "no-key"  # sent when the key is empty (local server)
JSON_MODE_ENABLED = False  # see the open point in the module docstring


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
    """LLMClient over /v1/chat/completions. One HTTP call per `generate`."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: str = "",
        timeout_s: Optional[float] = None,
        client: Any = None,
    ):
        self._client = client if client is not None else build_client(base_url, api_key, timeout_s)

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
        if json_mode and JSON_MODE_ENABLED:
            kwargs["response_format"] = {"type": "json_object"}
        response = _call(self._client.chat.completions.create, **kwargs)
        answered_by = getattr(response, "model", None)
        return LLMResponse(
            text=_response_text(response),
            model=answered_by if isinstance(answered_by, str) and answered_by else model,
            usage=_usage(response),
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
