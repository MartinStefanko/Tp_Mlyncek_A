"""Google Gemini adapter (google-genai). The only module in the project that imports google.genai.

The calls are the same as before step 6, so the baseline recording stays valid:
    client.models.generate_content(model=..., contents=prompt)
    client.models.embed_content(model=..., contents=texts)
No `config` is passed to either call.

Open point (step 6): `json_mode`, `temperature` and `max_tokens` of `generate` are NOT sent
to the API yet (no GenerateContentConfig, so no response_mime_type="application/json",
no temperature). The baseline was recorded without them; sending them changes the model
output and needs a new baseline. The pipeline still parses the answer with safe_load_json.

Timeout: set once on the client with HttpOptions(timeout=...). google-genai takes it in
MILLISECONDS (HttpOptions.timeout: "Timeout for the request in milliseconds"); the setting
QGEN_LLM_TIMEOUT_S is in seconds. The timeout is not a parameter of the calls above.
The SDK's own retries stay off (HttpOptions.retry_options is None); qgen.llm.retry retries.

Error translation (see qgen.llm.errors):
    APIError 429                 -> RateLimitError (Retry-After header or RetryInfo.retryDelay)
    APIError 408, 5xx            -> ProviderError(retryable=True)
    APIError other 4xx           -> ProviderError(retryable=False)   e.g. 400, 401, 403, 404
    httpx(2).TimeoutException    -> LLMTimeoutError
    httpx(2).TransportError      -> ProviderError(retryable=True)    network
"""

from __future__ import annotations

import re
from typing import Any, Optional

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from qgen.llm.base import LLMResponse
from qgen.llm.errors import LLMTimeoutError, ProviderError, RateLimitError

try:  # google-genai uses httpx2 (a fork with its own exception classes) when it is installed
    import httpx2
except ImportError:  # pragma: no cover - depends on the environment
    httpx2 = None

_TIMEOUT_ERRORS = (httpx.TimeoutException, TimeoutError) + ((httpx2.TimeoutException,) if httpx2 else ())
_NETWORK_ERRORS = (httpx.TransportError, ConnectionError) + ((httpx2.TransportError,) if httpx2 else ())

_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)s\s*$")


def build_client(api_key: str, timeout_s: Optional[float]) -> Any:
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    http_options = None
    if timeout_s:
        http_options = genai_types.HttpOptions(timeout=int(round(timeout_s * 1000)))
    return genai.Client(api_key=api_key, http_options=http_options)


def _parse_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = _DURATION.match(text)
    if match:
        return float(match.group(1))
    try:
        return float(text)
    except ValueError:
        return None  # HTTP-date form of Retry-After is not used by Gemini


def _find_retry_delay(details: Any) -> Optional[float]:
    """RetryInfo.retryDelay (e.g. "17s") anywhere in the error body."""
    if isinstance(details, dict):
        if "retryDelay" in details:
            return _parse_seconds(details["retryDelay"])
        for value in details.values():
            found = _find_retry_delay(value)
            if found is not None:
                return found
    elif isinstance(details, list):
        for value in details:
            found = _find_retry_delay(value)
            if found is not None:
                return found
    return None


def _retry_after(error: genai_errors.APIError) -> Optional[float]:
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers is not None:
        try:
            seconds = _parse_seconds(headers.get("retry-after"))
        except Exception:  # unusual header container; fall back to the body
            seconds = None
        if seconds is not None:
            return seconds
    return _find_retry_delay(getattr(error, "details", None))


def translate_error(error: Exception) -> Exception:
    """Vendor exception -> ProviderError. Anything unknown is returned unchanged."""
    if isinstance(error, ProviderError):
        return error
    if isinstance(error, genai_errors.APIError):
        code = error.code
        message = f"Gemini API error {code} {error.status or ''}".strip()
        if code == 429:
            return RateLimitError(message, retry_after=_retry_after(error))
        retryable = code == 408 or (isinstance(code, int) and code >= 500)
        return ProviderError(message, retryable=retryable, status=code)
    if isinstance(error, _TIMEOUT_ERRORS):
        return LLMTimeoutError(f"Gemini call timed out ({type(error).__name__})")
    if isinstance(error, _NETWORK_ERRORS):
        return ProviderError(f"Gemini network error ({type(error).__name__})", retryable=True)
    return error


def _response_text(response: Any) -> str:
    try:
        text = response.text
    except ValueError:  # older SDKs raise on a blocked / empty candidate
        return ""
    return text or ""


def _usage(response: Any) -> dict[str, int]:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {}
    usage = {}
    for key, attr in (
        ("input_tokens", "prompt_token_count"),
        ("output_tokens", "candidates_token_count"),
        ("total_tokens", "total_token_count"),
    ):
        value = getattr(meta, attr, None)
        if isinstance(value, int):
            usage[key] = value
    return usage


def _vectors(response: Any, expected: int) -> list[list[float]]:
    """Same parsing as the former qgen.llm.client.embed_batch."""
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None:
        single = getattr(response, "embedding", None)
        if single is not None:
            embeddings = [single]
        elif isinstance(response, dict) and "embedding" in response:
            embeddings = [response["embedding"]]
    if embeddings is None:
        raise ProviderError("API returned no embeddings.", retryable=False)
    if len(embeddings) != expected:
        raise ProviderError("Number of embeddings does not match the number of inputs.", retryable=False)
    vectors = []
    for emb in embeddings:
        values = getattr(emb, "values", None) or (emb.get("values") if isinstance(emb, dict) else None)
        if values is None:
            raise ProviderError("Embedding without 'values'.", retryable=False)
        vectors.append(list(values))
    return vectors


class GeminiLLM:
    """LLMClient for Gemini. See the module docstring for the open point on json_mode."""

    def __init__(self, api_key: str = "", timeout_s: Optional[float] = None, client: Any = None):
        self._client = client if client is not None else build_client(api_key, timeout_s)

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool = True,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        # json_mode / temperature / max_tokens are intentionally not sent (open point above).
        try:
            response = self._client.models.generate_content(model=model, contents=prompt)
        except Exception as error:
            translated = translate_error(error)
            if translated is error:
                raise
            raise translated from error
        return LLMResponse(text=_response_text(response), model=model, usage=_usage(response))


class GeminiEmbedder:
    """EmbeddingClient for Gemini (gemini-embedding-001). One call per `embed`."""

    def __init__(self, api_key: str = "", timeout_s: Optional[float] = None, client: Any = None):
        self._client = client if client is not None else build_client(api_key, timeout_s)

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.models.embed_content(model=model, contents=texts)
        except Exception as error:
            translated = translate_error(error)
            if translated is error:
                raise
            raise translated from error
        return _vectors(response, len(texts))
