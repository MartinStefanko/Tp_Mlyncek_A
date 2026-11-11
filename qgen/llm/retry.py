"""Retries, waiting and JSON parsing over the LLM interface (replaces generate_with_retry).

    generate_text  one prompt -> LLMResponse
    generate_json  generate_text + safe_load_json; invalid JSON -> one more call
                   items_key="items": the answer must be {"items": [...]}; returns the list
    embed_texts    texts -> vectors, split into batches of QGEN_EMBED_MAX_BATCH

Every call gets up to QGEN_LLM_RETRIES attempts (incl. the first). Which errors are retried
is decided by qgen.llm.errors.is_retryable: HTTP 429, 408, 5xx, timeout and network errors.
Waiting between attempts:
    HTTP 429 with Retry-After  the provider's value
    HTTP 429 without it        QGEN_LLM_RETRY_DELAY_S * 2 ** (attempt - 1)
    other retried errors       QGEN_LLM_RETRY_DELAY_S
The timeout of one call (QGEN_LLM_TIMEOUT_S) is set in the client by the adapter, which
raises LLMTimeoutError; here it is one more failed attempt. No thread runs a call in the
background, so two attempts of the same call never run at the same time.

All calls go through one semaphore sized by QGEN_LLM_MAX_CONCURRENCY (Príloha B), shared by
the whole process: it limits the sum of the calls of all batches that run at the same time
(qgen.llm.parallel, krok 13b), LLM and embedding calls together. Only the call itself holds
it; the wait between attempts does not.

On final failure the functions raise LLMError. The pipeline must not catch it.
`llm` / `embedder` = None: the client comes from qgen.llm.factory for the role.
`settings` is required: the Settings object the pipeline got (never get_settings() here).
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Optional, TypeVar

from qgen.config import Settings
from qgen.llm import factory
from qgen.llm.base import EmbeddingClient, LLMClient, LLMResponse
from qgen.llm.errors import LLMError, ProviderError, RateLimitError, is_retryable
from qgen.llm.json_load import safe_load_json

T = TypeVar("T")

JSON_ATTEMPTS = 2  # first answer + one repeated call when the JSON is invalid

# Replaced in tests, so retries do not really wait.
_sleep: Callable[[float], None] = time.sleep

_semaphore_lock = threading.Lock()
_semaphore: Optional[threading.BoundedSemaphore] = None
_semaphore_size = 0


def _concurrency_limit(size: int) -> threading.BoundedSemaphore:
    global _semaphore, _semaphore_size
    with _semaphore_lock:
        if _semaphore is None or _semaphore_size != size:
            _semaphore = threading.BoundedSemaphore(size)
            _semaphore_size = size
        return _semaphore


def _require_settings(settings: Any) -> None:
    if settings is None:
        raise TypeError("settings is required: pass the Settings object the pipeline got.")


def _describe(error: BaseException) -> str:
    """Short error text for logs. Only our own ProviderError messages are printed in full;
    for other exceptions only the type, so no foreign text (possibly with the prompt) gets
    into the log. The full exception stays in LLMError.cause."""
    if isinstance(error, ProviderError):
        return f"{type(error).__name__}: {error}"
    return type(error).__name__


def _wait_seconds(error: BaseException, attempt: int, settings: Settings) -> float:
    delay = settings.llm_retry_delay_s
    if isinstance(error, RateLimitError):
        if error.retry_after is not None:
            return max(0.0, float(error.retry_after))
        return delay * (2 ** (attempt - 1))
    return delay


def _with_retry(call: Callable[[], T], *, role: str, model: str, settings: Settings, verbose: bool) -> T:
    attempts = settings.llm_retries
    limit = _concurrency_limit(settings.llm_max_concurrency)
    for attempt in range(1, attempts + 1):
        try:
            with limit:
                return call()
        except Exception as error:
            if not is_retryable(error):
                raise LLMError(
                    f"LLM call failed with an error that is not retried: {_describe(error)}",
                    role=role, model=model, cause=error, attempts=attempt,
                ) from error
            if attempt == attempts:
                raise LLMError(
                    f"LLM call failed after all attempts: {_describe(error)}",
                    role=role, model=model, cause=error, attempts=attempt,
                ) from error
            wait = _wait_seconds(error, attempt, settings)
            if verbose:
                print(f"LLM zlyhalo (pokus {attempt}/{attempts}), cakam {wait:g}s a opakujem. Chyba: {_describe(error)}")
            _sleep(wait)
    raise AssertionError("unreachable")  # pragma: no cover


def generate_text(
    llm: Optional[LLMClient],
    prompt: str,
    *,
    role: str,
    settings: Settings,
    model: Optional[str] = None,
    verbose: bool = False,
    json_mode: bool = True,
    max_tokens: Optional[int] = None,
) -> LLMResponse:
    """One prompt with retries. JSON mode is on by default (every prompt asks for JSON);
    temperature is QGEN_LLM_TEMPERATURE. An answer cut off at the token limit
    (OutputTruncatedError) is not retried: LLMError with `truncated` = True."""
    _require_settings(settings)
    model = model or factory.model_for_role(role, settings)
    client = llm if llm is not None else factory.get_llm(role, settings)
    temperature = settings.llm_temperature
    return _with_retry(
        lambda: client.generate(
            prompt, model=model, json_mode=json_mode, temperature=temperature, max_tokens=max_tokens
        ),
        role=role, model=model, settings=settings, verbose=verbose,
    )


def generate_json(
    llm: Optional[LLMClient],
    prompt: str,
    *,
    role: str,
    settings: Settings,
    model: Optional[str] = None,
    verbose: bool = False,
    items_key: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Any:
    """Parsed JSON answer. Invalid JSON twice in a row -> LLMError(invalid_json=True).

    items_key="items": the prompt asks for an object {"items": [...]} (JSON mode of every
    provider returns an object). The answer must be a JSON object whose `items_key` is an
    array; that array is returned. Anything else (an array, an object without the key, a
    key that is not an array) counts as invalid JSON: same repeated call, then
    LLMError(invalid_json=True). It never becomes an empty result.
    """
    _require_settings(settings)
    model = model or factory.model_for_role(role, settings)
    last_error: Optional[json.JSONDecodeError] = None
    wrong_shape: Optional[str] = None
    for attempt in range(1, JSON_ATTEMPTS + 1):
        response = generate_text(
            llm, prompt, role=role, model=model, settings=settings, verbose=verbose, max_tokens=max_tokens
        )
        try:
            parsed = safe_load_json(response.text or "")
        except json.JSONDecodeError as error:
            last_error, wrong_shape = error, None
            if verbose and attempt < JSON_ATTEMPTS:
                print(f"LLM vratilo neplatny JSON (role {role}), opakujem volanie.")
            continue
        if items_key is not None:
            shape = _items_shape_error(parsed, items_key)
            if shape is not None:
                last_error, wrong_shape = None, shape
                if verbose and attempt < JSON_ATTEMPTS:
                    print(f"LLM vratilo JSON zleho tvaru ({shape}, role {role}), opakujem volanie.")
                continue
            return parsed[items_key]
        return parsed
    if wrong_shape is not None:
        message = (
            f"LLM returned JSON of the wrong shape ({wrong_shape}, expected an object with an array "
            f"'{items_key}') also after the repeated call"
        )
    else:
        message = "LLM returned invalid JSON also after the repeated call"
    raise LLMError(
        message, role=role, model=model, cause=last_error, invalid_json=True, attempts=JSON_ATTEMPTS,
    )


def _items_shape_error(parsed: Any, items_key: str) -> Optional[str]:
    """None when `parsed` is {items_key: [...]}, else a short description of the shape."""
    if not isinstance(parsed, dict):
        return type(parsed).__name__
    if items_key not in parsed:
        return f"object without '{items_key}'"
    if not isinstance(parsed[items_key], list):
        return f"'{items_key}' is {type(parsed[items_key]).__name__}"
    return None


def embed_texts(
    embedder: Optional[EmbeddingClient],
    texts: list[str],
    *,
    settings: Settings,
    model: Optional[str] = None,
    verbose: bool = False,
) -> list[list[float]]:
    """Vectors for `texts`, in order. One embedding call per QGEN_EMBED_MAX_BATCH texts."""
    _require_settings(settings)
    if not texts:
        return []
    model = model or settings.embed_model
    client = embedder if embedder is not None else factory.get_embedder(settings)
    batch = settings.embed_max_batch
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch):
        chunk = list(texts[start:start + batch])
        chunk_vectors = _with_retry(
            lambda: client.embed(chunk, model=model),
            role=factory.EMBED_ROLE, model=model, settings=settings, verbose=verbose,
        )
        if len(chunk_vectors) != len(chunk):
            raise LLMError(
                "Number of embeddings does not match the number of inputs",
                role=factory.EMBED_ROLE, model=model, attempts=1,
            )
        vectors.extend(list(v) for v in chunk_vectors)
    return vectors
