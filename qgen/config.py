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
current models is baseline_v4 (step 13; its models and question settings are in its
metrics.json and are passed explicitly by the replay tests).

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
from pathlib import Path
from typing import Literal, Optional

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from qgen.contract.options import OptionDefaults

# "replay" answers from a recording (qgen.llm.adapters.replay); development and tests only.
LlmProvider = Literal["gemini", "openai_compatible", "anthropic", "replay"]
# "anthropic" is accepted here only so that qgen.llm.factory can reject it with a clear
# message (Anthropic has no embedding API) instead of a generic validation error.
EmbedProvider = Literal["gemini", "openai_compatible", "anthropic", "replay"]
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
    llm_max_output_tokens: int = Field(
        default=32768, ge=1, description="Output token limit of question generation calls (all providers)."
    )
    llm_temperature: float = Field(default=0.2, ge=0, le=2, description="Sent with every call (not to Anthropic).")
    llm_max_concurrency: int = Field(default=4, ge=1, description="Parallel LLM calls of one worker.")
    llm_retries: int = Field(default=5, ge=1, description="Attempts of one LLM / embedding call (incl. the first).")
    llm_retry_delay_s: float = Field(
        default=3.0, ge=0, description="Wait between attempts; base of the exponential wait on HTTP 429."
    )
    llm_timeout_s: float = Field(default=120, gt=0, description="Client timeout of one LLM / embedding call.")

    # --- Replay (development only: QGEN_LLM_PROVIDER / QGEN_EMBED_PROVIDER = replay) -----
    llm_replay_path: Optional[Path] = Field(default=None, description="Recording llm_calls.jsonl.")
    llm_replay_delay_s: float = Field(default=0, ge=0, description="Sleep before every replayed call.")
    llm_replay_fail: bool = Field(default=False, description="Question generation fails (llm_error).")

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
    question_generation_batch_size: int = Field(
        default=3, ge=1, description="Sections per question generation call (also extra rounds and coverage)."
    )
    question_eval_batch_size: int = Field(default=20, ge=1, description="Questions per faithfulness/answerability call.")
    question_generation_attempts: int = Field(default=3, ge=1)
    question_eval_attempts: int = Field(default=2, ge=1)
    check_batch_attempts: int = Field(default=2, ge=1, description="Faithfulness / answerability batch attempts.")
    context_max_chars: int = Field(default=8000, ge=1, description="Source text for one LO or question.")
    language_sample_max_chars: int = Field(default=12000, ge=1, description="Text for language detection.")
    topic_source_max_chars: int = Field(default=20000, ge=1, description="Text for topic extraction.")

    # --- Question generation (krok 13) -------------------------------------------------------
    question_extra_rounds: int = Field(
        default=1, ge=0, le=5, description="Extra generation rounds per batch; 0 = off."
    )
    question_dedup_threshold: float = Field(
        default=0.94, ge=0, le=1, description="Cosine similarity at which a question of the same section is a duplicate."
    )
    question_choice_min_share: float = Field(
        default=0.6, ge=0, le=1, description="Share of choice questions per section asked in the prompt."
    )
    question_practical_bloom: str = Field(
        default="apply,analyze,evaluate,create", description="Section Bloom levels that also get a practical task."
    )
    question_no_answer_bloom: str = Field(
        default="evaluate,create", description="Question Bloom levels allowed for no_answer."
    )

    # --- Question hints (krok 13c): one hint with the source, made by code, not by the LLM ---
    # {file} = original name of the uploaded PDF, {pages} = e.g. "14", "14 – 15" or "14, 16".
    hint_name_sk: str = Field(default="Zdroj", min_length=1, description="Hint name, Slovak document.")
    hint_name_en: str = Field(default="Source", min_length=1, description="Hint name, English document.")
    hint_page_sk: str = Field(default="{file}, strana {pages}", description="One page, Slovak document.")
    hint_pages_sk: str = Field(default="{file}, strany {pages}", description="More pages, Slovak document.")
    hint_page_en: str = Field(default="{file}, page {pages}", description="One page, English document.")
    hint_pages_en: str = Field(default="{file}, pages {pages}", description="More pages, English document.")

    @field_validator("hint_page_sk", "hint_pages_sk", "hint_page_en", "hint_pages_en")
    @classmethod
    def _hint_template(cls, value: str) -> str:
        if "{file}" not in value or "{pages}" not in value:
            raise ValueError("hint template must contain {file} and {pages}")
        return value

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
    data_dir: Path = Field(default=Path("/data"), description="Job storage: jobs/{id}/ and locks/.")
    result_ttl_days: float = Field(default=7, gt=0)
    failed_input_ttl_hours: float = Field(default=24, gt=0)

    # --- Queue -------------------------------------------------------------------------
    redis_url: str = Field(default="redis://redis:6379/0", description="Queue, progress cache, external_ref reservations.")
    maintenance_interval_s: float = Field(
        default=600, ge=5, description="Worker maintenance (cleanup + recovery); shorter only for development."
    )
    job_timeout_s: float = Field(
        default=3600, gt=0, description="Time limit of one job run in the worker (eko.pdf takes about 290 s)."
    )

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
    def practical_bloom_levels(self) -> tuple[str, ...]:
        return tuple(level.lower() for level in _split_list(self.question_practical_bloom))

    @property
    def no_answer_bloom_levels(self) -> tuple[str, ...]:
        return tuple(level.lower() for level in _split_list(self.question_no_answer_bloom))

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
