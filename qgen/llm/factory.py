"""Creates LLM and embedding clients from the configuration (Príloha B).

Roles and the model setting each one uses (same models as before step 6):
    lo         QGEN_LLM_MODEL_LO         language detection (LO pipeline), LO generation,
                                         filling missing sources
    prereq     QGEN_LLM_MODEL_LO         prerequisites
    questions  QGEN_LLM_MODEL_QUESTIONS  language detection (question pipeline), question generation
    eval       QGEN_LLM_MODEL_EVAL       question quality score (1–5)
    check      QGEN_LLM_MODEL            LO / question faithfulness, answerability
    topics     QGEN_LLM_MODEL            document topic extraction
    split      QGEN_LLM_MODEL            splitting into lessons (later step)

The embedding provider is configured separately from the LLM provider
(QGEN_EMBED_PROVIDER, QGEN_EMBED_MODEL).

Only the gemini provider exists in step 6; openai_compatible and anthropic come in step 7.
Clients are cached per (provider, key, timeout), so the pipeline does not build a new SDK
client for each call.
"""

from __future__ import annotations

from typing import Optional

from qgen.config import Settings, get_settings
from qgen.llm.base import EmbeddingClient, LLMClient

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

_llm_cache: dict[tuple, LLMClient] = {}
_embedder_cache: dict[tuple, EmbeddingClient] = {}


def _check_role(role: str) -> None:
    if role not in ROLE_MODEL_SETTING:
        raise ValueError(f"Unknown LLM role {role!r} (allowed: {', '.join(ROLES)}).")


def model_for_role(role: str, settings: Optional[Settings] = None) -> str:
    _check_role(role)
    return getattr(settings or get_settings(), ROLE_MODEL_SETTING[role])


def get_llm(role: str, settings: Optional[Settings] = None) -> LLMClient:
    """LLM client for a role. Today every role uses QGEN_LLM_PROVIDER."""
    _check_role(role)
    settings = settings or get_settings()
    provider = settings.llm_provider
    if provider == "gemini":
        key = settings.gemini_api_key.get_secret_value()
        cache_key = (provider, key, settings.llm_timeout_s)
        if cache_key not in _llm_cache:
            from qgen.llm.adapters.gemini import GeminiLLM

            _llm_cache[cache_key] = GeminiLLM(api_key=key, timeout_s=settings.llm_timeout_s)
        return _llm_cache[cache_key]
    raise NotImplementedError(f"LLM provider {provider!r} is not implemented yet (step 7).")


def get_embedder(settings: Optional[Settings] = None) -> EmbeddingClient:
    """Embedding client from QGEN_EMBED_PROVIDER, independent of the LLM provider."""
    settings = settings or get_settings()
    provider = settings.embed_provider
    if provider == "gemini":
        key = settings.gemini_api_key.get_secret_value()
        cache_key = (provider, key, settings.llm_timeout_s)
        if cache_key not in _embedder_cache:
            from qgen.llm.adapters.gemini import GeminiEmbedder

            _embedder_cache[cache_key] = GeminiEmbedder(api_key=key, timeout_s=settings.llm_timeout_s)
        return _embedder_cache[cache_key]
    raise NotImplementedError(f"Embedding provider {provider!r} is not implemented yet (step 7).")


def clear_cache() -> None:
    _llm_cache.clear()
    _embedder_cache.clear()
