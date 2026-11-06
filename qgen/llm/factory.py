"""Creates LLM and embedding clients from the configuration (Príloha B).

Roles and the model setting each one uses:
    lo         QGEN_LLM_MODEL_LO         language detection (LO pipeline), LO generation,
                                         filling missing sources
    prereq     QGEN_LLM_MODEL_LO         prerequisites
    questions  QGEN_LLM_MODEL_QUESTIONS  language detection (question pipeline), question generation
    eval       QGEN_LLM_MODEL_EVAL       question quality score (1–5)
    check      QGEN_LLM_MODEL            LO / question faithfulness, answerability
    topics     QGEN_LLM_MODEL            document topic extraction
    split      QGEN_LLM_MODEL            splitting into lessons (later step)
QGEN_LLM_MODEL_LO / _QUESTIONS / _EVAL inherit QGEN_LLM_MODEL when not set (qgen.config).

Providers:
    QGEN_LLM_PROVIDER    gemini | openai_compatible | anthropic | replay  -> get_llm(role, settings)
    QGEN_EMBED_PROVIDER  gemini | openai_compatible | replay           -> get_embedder(settings)
The two are independent, e.g. vLLM for generation and Gemini for embeddings.

What each provider needs (checked by check_config at start, and again by get_llm /
get_embedder before a client is built):
    gemini             GEMINI_API_KEY
    openai_compatible  QGEN_LLM_BASE_URL (LLM) / QGEN_EMBED_BASE_URL (embeddings);
                       keys optional (a local server needs none): QGEN_LLM_API_KEY for the
                       LLM, QGEN_EMBED_API_KEY for embeddings (empty = QGEN_LLM_API_KEY)
    anthropic          ANTHROPIC_API_KEY; LLM only (no embedding API)
    replay             QGEN_LLM_REPLAY_PATH (an existing recording, e.g. baseline_v4/llm_calls.jsonl);
                       development only: answers from the recording (qgen.llm.adapters.replay).
                       Never refused, but check_config logs a loud warning every time. One
                       client serves both roles when LLM and embeddings are replay.
A provider other than gemini with a model whose name starts with GEMINI_MODEL_PREFIX is
rejected too: the usual mistake after switching QGEN_LLM_PROVIDER without changing the
models. The same holds for QGEN_EMBED_PROVIDER and QGEN_EMBED_MODEL.

Every function takes the `settings` the caller (pipeline, CLI, worker) got; nothing here
reads the configuration itself. A client is built once per provider and settings (cache
key below) and shared by all roles and calls.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable

from qgen.config import ROLE_MODEL_FIELDS, Settings
from qgen.llm.base import EmbeddingClient, LLMClient
from qgen.llm.errors import LLMConfigError

ROLE_MODEL_SETTING = {
    "lo": "llm_model_lo",
    "prereq": "llm_model_lo",
    "questions": "llm_model_questions",
    "eval": "llm_model_eval",
    "check": "llm_model",
    "topics": "llm_model",
    "split": "llm_model",
}
ROLES = tuple(ROLE_MODEL_SETTING)
EMBED_ROLE = "embed"  # role name in LLMError for embedding calls
GEMINI_MODEL_PREFIX = "gemini-"
REPLAY = "replay"

logger = logging.getLogger("qgen.llm")

_lock = threading.Lock()
_llm_cache: dict[tuple, LLMClient] = {}
_embedder_cache: dict[tuple, EmbeddingClient] = {}
_replay_cache: dict[tuple, Any] = {}


def _env_name(field: str) -> str:
    return "QGEN_" + field.upper()


def _require_settings(settings: Any) -> Settings:
    if settings is None:
        raise TypeError("settings is required: pass the Settings object the pipeline got.")
    return settings


def _check_role(role: str) -> None:
    if role not in ROLE_MODEL_SETTING:
        raise ValueError(f"Unknown LLM role {role!r} (allowed: {', '.join(ROLES)}).")


def model_for_role(role: str, settings: Settings) -> str:
    _check_role(role)
    return getattr(_require_settings(settings), ROLE_MODEL_SETTING[role])


# --- configuration checks -------------------------------------------------------------

def llm_config_problems(settings: Settings) -> list[str]:
    """What is wrong with the LLM part of the configuration (empty list = fine)."""
    provider = settings.llm_provider
    problems = []
    if provider == "gemini":
        if not settings.gemini_api_key.get_secret_value():
            problems.append("QGEN_LLM_PROVIDER=gemini needs GEMINI_API_KEY.")
    elif provider == "openai_compatible":
        if not settings.llm_base_url:
            problems.append(
                "QGEN_LLM_PROVIDER=openai_compatible needs QGEN_LLM_BASE_URL "
                "(e.g. https://api.openai.com/v1 or http://llm:8001/v1)."
            )
    elif provider == "anthropic":
        if not settings.anthropic_api_key.get_secret_value():
            problems.append("QGEN_LLM_PROVIDER=anthropic needs ANTHROPIC_API_KEY.")
    elif provider == REPLAY:
        problems.extend(_replay_problems(settings, "QGEN_LLM_PROVIDER"))
    else:  # pragma: no cover - Settings validates the Literal
        problems.append(f"Unknown QGEN_LLM_PROVIDER={provider!r}.")

    if provider not in ("gemini", REPLAY):  # a recording of Gemini has Gemini models
        base = settings.llm_model
        if base.startswith(GEMINI_MODEL_PREFIX):
            problems.append(
                f"QGEN_LLM_MODEL={base} is a Gemini model, but QGEN_LLM_PROVIDER={provider}. "
                f"Set QGEN_LLM_MODEL to a model of {provider}."
            )
        for field in ROLE_MODEL_FIELDS:
            value = getattr(settings, field)
            if value != base and value.startswith(GEMINI_MODEL_PREFIX):
                problems.append(
                    f"{_env_name(field)}={value} is a Gemini model, but QGEN_LLM_PROVIDER={provider}. "
                    f"Set it to a model of {provider} or remove it (it then inherits QGEN_LLM_MODEL)."
                )
    return problems


def embed_config_problems(settings: Settings) -> list[str]:
    """What is wrong with the embedding part of the configuration (empty list = fine)."""
    provider = settings.embed_provider
    problems = []
    if provider == "gemini":
        if not settings.gemini_api_key.get_secret_value():
            problems.append("QGEN_EMBED_PROVIDER=gemini needs GEMINI_API_KEY.")
    elif provider == "openai_compatible":
        if not settings.embed_base_url:
            problems.append(
                "QGEN_EMBED_PROVIDER=openai_compatible needs QGEN_EMBED_BASE_URL "
                "(e.g. https://api.openai.com/v1 or http://llm:8001/v1)."
            )
    elif provider == "anthropic":
        problems.append(
            "QGEN_EMBED_PROVIDER=anthropic is not possible: Anthropic has no embedding API. "
            "Use QGEN_EMBED_PROVIDER=gemini (GEMINI_API_KEY) or openai_compatible (QGEN_EMBED_BASE_URL)."
        )
    elif provider == REPLAY:
        problems.extend(_replay_problems(settings, "QGEN_EMBED_PROVIDER"))
    else:  # pragma: no cover - Settings validates the Literal
        problems.append(f"Unknown QGEN_EMBED_PROVIDER={provider!r}.")

    if provider not in ("gemini", "anthropic", REPLAY) and settings.embed_model.startswith(GEMINI_MODEL_PREFIX):
        problems.append(
            f"QGEN_EMBED_MODEL={settings.embed_model} is a Gemini model, but QGEN_EMBED_PROVIDER={provider}. "
            f"Set QGEN_EMBED_MODEL to an embedding model of the server."
        )
    return problems


def _replay_problems(settings: Settings, variable: str) -> list[str]:
    path = settings.llm_replay_path
    if path is None or not str(path):
        return [f"{variable}=replay needs QGEN_LLM_REPLAY_PATH (a recording llm_calls.jsonl)."]
    if not Path(path).is_file():
        return [f"{variable}=replay: recording QGEN_LLM_REPLAY_PATH={path} does not exist."]
    return []


def replay_in_use(settings: Settings, *, llm: bool = True, embedder: bool = True) -> list[str]:
    """Names of the variables that select replay (empty list = no replay)."""
    used = []
    if llm and settings.llm_provider == REPLAY:
        used.append("QGEN_LLM_PROVIDER")
    if embedder and settings.embed_provider == REPLAY:
        used.append("QGEN_EMBED_PROVIDER")
    return used


def warn_replay(settings: Settings, *, llm: bool = True, embedder: bool = True) -> bool:
    """Loud warning when replay is selected: it is for development only. True if warned."""
    used = replay_in_use(settings, llm=llm, embedder=embedder)
    if not used:
        return False
    bar = "!" * 78
    logger.warning(
        "\n%s\n!!! REPLAY: %s=replay. Answers come from the recording %s, not from an LLM.\n"
        "!!! Development and tests only. NEVER in production. /v1/info shows the providers.\n"
        "!!! delay_s=%g fail_questions=%s\n%s",
        bar, "=replay, ".join(used), settings.llm_replay_path, settings.llm_replay_delay_s,
        settings.llm_replay_fail, bar,
    )
    return True


def _raise_if(problems: list[str]) -> None:
    if problems:
        raise LLMConfigError("Invalid LLM configuration:\n- " + "\n- ".join(problems))


def check_config(settings: Settings, *, llm: bool = True, embedder: bool = True) -> None:
    """Start-up check of the LLM / embedding configuration; LLMConfigError lists every problem.

    Called by the entry points (CLI, worker) before any work, so a missing key or base_url
    fails at once and not at the first call. Also builds the clients (no network call),
    which are then reused. `llm` / `embedder` = False skips a part the caller replaces
    (e.g. a test client).
    """
    settings = _require_settings(settings)
    problems = (llm_config_problems(settings) if llm else []) + (
        embed_config_problems(settings) if embedder else []
    )
    _raise_if(problems)
    warn_replay(settings, llm=llm, embedder=embedder)
    if llm:
        get_llm(ROLES[0], settings)
    if embedder:
        get_embedder(settings)


# --- clients --------------------------------------------------------------------------

def _cached(cache: dict, key: tuple, build: Callable[[], Any]) -> Any:
    with _lock:
        client = cache.get(key)
        if client is None:
            client = cache[key] = build()
        return client


def get_llm(role: str, settings: Settings) -> LLMClient:
    """LLM client for a role. Every role uses QGEN_LLM_PROVIDER; the model comes per call."""
    _check_role(role)
    settings = _require_settings(settings)
    _raise_if(llm_config_problems(settings))
    provider = settings.llm_provider
    timeout = settings.llm_timeout_s

    if provider == "gemini":
        key = settings.gemini_api_key.get_secret_value()

        def build():
            from qgen.llm.adapters.gemini import GeminiLLM

            return GeminiLLM(api_key=key, timeout_s=timeout)

        return _cached(_llm_cache, (provider, key, timeout), build)

    if provider == "openai_compatible":
        base_url, key = settings.llm_base_url, settings.llm_api_key.get_secret_value()

        def build():
            from qgen.llm.adapters.openai_compatible import OpenAICompatLLM

            return OpenAICompatLLM(base_url=base_url, api_key=key, timeout_s=timeout)

        return _cached(_llm_cache, (provider, base_url, key, timeout), build)

    if provider == "anthropic":
        key, max_tokens = settings.anthropic_api_key.get_secret_value(), settings.llm_max_tokens

        def build():
            from qgen.llm.adapters.anthropic import AnthropicLLM

            return AnthropicLLM(api_key=key, timeout_s=timeout, max_tokens=max_tokens)

        return _cached(_llm_cache, (provider, key, timeout, max_tokens), build)

    if provider == REPLAY:
        return _replay_client(settings)

    raise LLMConfigError(f"Unknown QGEN_LLM_PROVIDER={provider!r}.")  # pragma: no cover


def get_embedder(settings: Settings) -> EmbeddingClient:
    """Embedding client from QGEN_EMBED_PROVIDER, independent of the LLM provider."""
    settings = _require_settings(settings)
    _raise_if(embed_config_problems(settings))
    provider = settings.embed_provider
    timeout = settings.llm_timeout_s

    if provider == "gemini":
        key = settings.gemini_api_key.get_secret_value()

        def build():
            from qgen.llm.adapters.gemini import GeminiEmbedder

            return GeminiEmbedder(api_key=key, timeout_s=timeout)

        return _cached(_embedder_cache, (provider, key, timeout), build)

    if provider == "openai_compatible":
        # QGEN_EMBED_API_KEY, else QGEN_LLM_API_KEY (backward compatible).
        base_url, key = settings.embed_base_url, settings.effective_embed_api_key

        def build():
            from qgen.llm.adapters.openai_compatible import OpenAICompatEmbedder

            return OpenAICompatEmbedder(base_url=base_url, api_key=key, timeout_s=timeout)

        return _cached(_embedder_cache, (provider, base_url, key, timeout), build)

    if provider == REPLAY:
        return _replay_client(settings)

    raise LLMConfigError(f"Unknown QGEN_EMBED_PROVIDER={provider!r}.")  # pragma: no cover


def _replay_client(settings: Settings) -> Any:
    """One ReplayLLM per recording and options, shared by the LLM and the embedding role."""
    path, delay, fail = str(settings.llm_replay_path), settings.llm_replay_delay_s, settings.llm_replay_fail

    def build():
        from qgen.llm.adapters.replay import ReplayLLM

        return ReplayLLM(path, delay_s=delay, fail_questions=fail)

    return _cached(_replay_cache, (path, delay, fail), build)


def clear_cache() -> None:
    with _lock:
        _llm_cache.clear()
        _embedder_cache.clear()
        _replay_cache.clear()
