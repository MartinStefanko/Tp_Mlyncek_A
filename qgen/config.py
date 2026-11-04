"""Central configuration (Príloha D – Konfigurácia) from the environment and `.env`.

Every variable has the prefix QGEN_, except the vendor keys GEMINI_API_KEY and
ANTHROPIC_API_KEY. This is the only module that reads the environment.

The pipeline gets a `Settings` object as a parameter and passes it on to every module,
also to qgen.llm.retry and qgen.llm.factory. Only the entry points (qgen/cli.py, service/,
tools/) call `get_settings()` (one cached instance per process); a guard test checks it.

Models: QGEN_LLM_MODEL is the default model (default gemini-3.1-flash-lite).
QGEN_LLM_MODEL_LO and _QUESTIONS are optional; when not set (or empty) they inherit
QGEN_LLM_MODEL. QGEN_LLM_MODEL_EVAL has its own default gemini-3.5-flash (the question
quality score uses a stronger model); set it empty to inherit QGEN_LLM_MODEL. With another
provider than gemini, QGEN_LLM_MODEL_EVAL must be set (qgen.llm.factory.check_config).
The gemini-2.5 models of the old baseline (baseline/) return 404 to new users; the replay
tests pass them explicitly (tests/baseline_data.py). The reference recording on the
current models will be baseline_v3.

Model roles (role names of qgen.llm.factory in brackets):
    QGEN_LLM_MODEL            checks: topic extraction [topics], LO faithfulness, question
                              faithfulness and answerability [check]; lesson split [split]
    QGEN_LLM_MODEL_LO         language detection in the LO pipeline, LO generation [lo],
                              prerequisites [prereq]
    QGEN_LLM_MODEL_QUESTIONS  language detection in the question pipeline, question
                              generation [questions]
    QGEN_LLM_MODEL_EVAL       question quality score (1–5) [eval]
"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from qgen.contract.options import OptionDefaults

LlmProvider = Literal["gemini", "openai_compatible", "anthropic"]
# "anthropic" is accepted here only so that qgen.llm.factory can reject it with a clear
# message (Anthropic has no embedding API) instead of a generic validation error.
EmbedProvider = Literal["gemini", "openai_compatible", "anthropic"]
ROLE_MODEL_FIELDS = ("llm_model_lo", "llm_model_questions", "llm_model_eval")


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QGEN_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    # --- LLM -----------------------------------------------------------------------
    llm_provider: LlmProvider = "gemini"
    llm_base_url: Optional[str] = Field(default=None, description="openai_compatible: server address.")
    llm_api_key: SecretStr = Field(default=SecretStr(""), description="openai_compatible: key.")
    gemini_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias=AliasChoices("GEMINI_API_KEY", "gemini_api_key")
    )
    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias=AliasChoices("ANTHROPIC_API_KEY", "anthropic_api_key")
    )
    llm_model: str = Field(default="gemini-3.1-flash-lite", description="Checks (faithfulness, answerability, topics).")
    # Role models: None or "" = inherit llm_model (filled in model_post_init).
    llm_model_lo: Optional[str] = Field(default=None, description="LO generation, prerequisites.")
    llm_model_questions: Optional[str] = Field(default=None, description="Question generation.")
    llm_model_eval: Optional[str] = Field(
        default="gemini-3.5-flash", description="Question quality score; empty = inherit llm_model."
    )
    llm_max_tokens: int = Field(
        default=8192, ge=1, description="Anthropic: max_tokens of a call (required by the Messages API)."
    )
    llm_max_concurrency: int = Field(default=4, ge=1, description="Parallel LLM calls of one worker.")
    llm_retries: int = Field(default=5, ge=1, description="Attempts of one LLM / embedding call (incl. the first).")
    llm_retry_delay_s: float = Field(
        default=3.0, ge=0, description="Wait between attempts; base of the exponential wait on HTTP 429."
    )
    llm_timeout_s: float = Field(default=120, gt=0, description="Client timeout of one LLM / embedding call.")

    # --- Embeddings ------------------------------------------------------------------
    embed_provider: EmbedProvider = "gemini"
    embed_base_url: Optional[str] = None
    embed_api_key: SecretStr = Field(
        default=SecretStr(""), description="openai_compatible embeddings: key; empty = QGEN_LLM_API_KEY."
    )
    embed_model: str = "gemini-embedding-001"
    embed_max_batch: int = Field(default=100, ge=1, description="Texts per embedding call.")

    # --- Pipeline thresholds -----------------------------------------------------------
    cluster_sim_threshold: float = Field(default=0.82, ge=0, le=1, description="LO clustering (cosine).")
    cluster_require_source_overlap: bool = Field(default=False, description="Cluster only LO with a shared page.")
    topic_match_threshold: float = Field(default=0.72, ge=0, le=1, description="Topic covered by an LO (cosine).")
    lo_min_faithfulness_score: int = Field(default=3, ge=1, le=5, description="LO filter.")
    question_min_score: int = Field(default=3, ge=1, le=5, description="Question filter: quality score.")
    question_min_faithfulness_score: int = Field(default=3, ge=1, le=5, description="Question filter.")
    question_min_answerability_score: int = Field(default=3, ge=1, le=5, description="Question filter.")
    question_faithful_score: int = Field(default=4, ge=1, le=5, description="Report only: 'faithful' flag.")
    question_answerable_score: int = Field(default=4, ge=1, le=5, description="Report only: 'answerable' flag.")

    # --- Pipeline batches, attempts and text limits --------------------------------------
    lo_batch_size: int = Field(default=10, ge=1, description="Pages per LO generation call; LO per faithfulness call.")
    question_batch_size: int = Field(default=10, ge=1, description="LO per question generation call.")
    question_eval_batch_size: int = Field(default=20, ge=1, description="Questions per faithfulness/answerability call.")
    question_generation_attempts: int = Field(default=3, ge=1)
    question_eval_attempts: int = Field(default=2, ge=1)
    check_batch_attempts: int = Field(default=2, ge=1, description="Faithfulness / answerability batch attempts.")
    context_max_chars: int = Field(default=8000, ge=1, description="Source text for one LO or question.")
    language_sample_max_chars: int = Field(default=12000, ge=1, description="Text for language detection.")
    topic_source_max_chars: int = Field(default=20000, ge=1, description="Text for topic extraction.")

    # --- Option defaults ------------------------------------------------------------------
    ocr_enabled: bool = False
    default_lesson_minutes: int = 45

    # --- Access and input limits --------------------------------------------------------
    api_keys: SecretStr = Field(default=SecretStr(""), description="Comma-separated keys; empty = no check.")
    max_files: int = Field(default=10, ge=1)
    max_upload_mb: float = Field(default=50, gt=0)
    max_pages: int = Field(default=300, ge=1)
    max_active_jobs: int = Field(default=10, ge=1)

    # --- Retention ---------------------------------------------------------------------
    result_ttl_days: float = Field(default=7, gt=0)
    failed_input_ttl_hours: float = Field(default=24, gt=0)

    # --- Webhooks ----------------------------------------------------------------------
    webhook_secret: SecretStr = SecretStr("")
    webhook_allowed_hosts: str = Field(default="", description="Comma-separated hosts; empty = webhooks off.")
    webhook_timeout_s: float = Field(default=5, gt=0)

    # --- Deployment (read by docker compose; here for /v1/info and completeness) -----------
    workers: int = Field(default=1, ge=1)
    local_model: Optional[str] = None

    def model_post_init(self, __context) -> None:
        """Role models that are not set (None or "") inherit QGEN_LLM_MODEL.

        Runs also for `Settings.model_construct(...)` (used by the replay tests), so a
        constructed object has the same role models as a validated one.
        """
        for name in ROLE_MODEL_FIELDS:
            if not getattr(self, name, None):
                setattr(self, name, self.llm_model)

    @property
    def api_key_list(self) -> list[str]:
        return _split_list(self.api_keys.get_secret_value())

    @property
    def allowed_webhook_hosts(self) -> set[str]:
        return {host.lower() for host in _split_list(self.webhook_allowed_hosts)}

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def effective_embed_api_key(self) -> str:
        """QGEN_EMBED_API_KEY, or QGEN_LLM_API_KEY when it is empty (backward compatible)."""
        return self.embed_api_key.get_secret_value() or self.llm_api_key.get_secret_value()

    def option_defaults(self) -> OptionDefaults:
        return OptionDefaults(ocr=self.ocr_enabled, lesson_minutes=self.default_lesson_minutes)


def setting_or_default(value, name: str, settings: Optional["Settings"] = None):
    """`value`, or the setting `name` when value is None.

    The setting comes from `settings` when given, else from get_settings(). The settings
    are read only when needed, so a caller that passes every value never touches the
    environment.
    """
    if value is not None:
        return value
    return getattr(settings if settings is not None else get_settings(), name)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings. Tests call `get_settings.cache_clear()` after changing env."""
    return Settings()
