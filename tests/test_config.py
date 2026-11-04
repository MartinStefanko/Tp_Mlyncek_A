"""qgen/config.py: defaults, loading from the environment and .env, one source of truth."""

import json
import os
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from qgen.config import Settings, get_settings, setting_or_default
from qgen.contract import OptionDefaults
from service.settings import ServiceSettings

REPO_ROOT = Path(__file__).resolve().parent.parent
QGEN_DIR = REPO_ROOT / "qgen"
VENDOR_KEYS = ("GEMINI_API_KEY", "ANTHROPIC_API_KEY")


@pytest.fixture
def clean_env(monkeypatch):
    """No QGEN_* or vendor key variables from the developer's shell."""
    for name in list(os.environ):
        if name.upper().startswith("QGEN_") or name.upper() in VENDOR_KEYS:
            monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


def test_defaults(clean_env):
    s = Settings(_env_file=None)
    # Models: today's models (baseline), not gemini-4-argon from the specification.
    assert s.llm_provider == "gemini"
    assert s.llm_model == "gemini-2.5-flash-lite"
    assert s.llm_model_lo == "gemini-2.5-flash-lite"
    assert s.llm_model_questions == "gemini-2.5-flash-lite"
    assert s.llm_model_eval == "gemini-2.5-flash"
    assert s.embed_provider == "gemini"
    assert s.embed_model == "gemini-embedding-001"
    assert s.llm_base_url is None and s.embed_base_url is None
    assert s.llm_max_concurrency == 4
    # Thresholds and filters
    assert s.cluster_sim_threshold == 0.82
    assert s.cluster_require_source_overlap is False
    assert s.topic_match_threshold == 0.72
    assert s.lo_min_faithfulness_score == 3
    assert s.question_min_score == 3
    assert s.question_min_faithfulness_score == 3
    assert s.question_min_answerability_score == 3
    assert s.question_faithful_score == 4
    assert s.question_answerable_score == 4
    # Batches, attempts, text limits
    assert (s.lo_batch_size, s.question_batch_size, s.question_eval_batch_size) == (10, 10, 20)
    assert (s.question_generation_attempts, s.question_eval_attempts, s.check_batch_attempts) == (3, 2, 2)
    assert (s.context_max_chars, s.language_sample_max_chars, s.topic_source_max_chars) == (8000, 12000, 20000)
    # Príloha D
    assert s.ocr_enabled is False
    assert s.default_lesson_minutes == 45
    assert s.api_key_list == []
    assert (s.max_files, s.max_upload_mb, s.max_pages, s.max_active_jobs) == (10, 50, 300, 10)
    assert (s.result_ttl_days, s.failed_input_ttl_hours) == (7, 24)
    assert s.webhook_secret.get_secret_value() == ""
    assert s.allowed_webhook_hosts == set()
    assert s.webhook_timeout_s == 5
    assert s.workers == 1
    assert s.local_model is None
    assert s.gemini_api_key.get_secret_value() == ""


def test_model_construct_equals_validated_defaults(clean_env):
    """Replay tests use Settings.model_construct() to ignore .env; it must equal the defaults."""
    assert Settings.model_construct().model_dump() == Settings(_env_file=None).model_dump()


def test_defaults_match_baseline_models(clean_env):
    metrics_path = REPO_ROOT / "baseline" / "metrics.json"
    if not metrics_path.is_file():
        pytest.skip("baseline/metrics.json is missing")
    models = json.loads(metrics_path.read_text(encoding="utf-8"))["models"]
    s = Settings(_env_file=None)
    assert s.llm_model_lo == models["lo_generation"] == models["lo_prerequisites"]
    assert s.llm_model_questions == models["item_generation"]
    assert s.llm_model_eval == models["item_evaluation"]
    assert s.lo_batch_size == models["lo_batch_size"]
    assert s.question_batch_size == models["item_batch_size"]
    assert s.question_eval_batch_size == models["item_evaluation_batch_size"]


def test_values_from_environment(clean_env):
    clean_env.setenv("QGEN_LLM_MODEL", "model-a")
    clean_env.setenv("QGEN_LLM_MODEL_EVAL", "model-b")
    clean_env.setenv("QGEN_CLUSTER_SIM_THRESHOLD", "0.9")
    clean_env.setenv("QGEN_TOPIC_MATCH_THRESHOLD", "0.5")
    clean_env.setenv("QGEN_QUESTION_MIN_SCORE", "4")
    clean_env.setenv("QGEN_OCR_ENABLED", "true")
    clean_env.setenv("QGEN_WEBHOOK_ALLOWED_HOSTS", "Eceh-Backend, other:8080")
    clean_env.setenv("GEMINI_API_KEY", "secret-key")
    s = Settings(_env_file=None)
    assert s.llm_model == "model-a"
    assert s.llm_model_eval == "model-b"
    assert s.llm_model_lo == "gemini-2.5-flash-lite"  # other roles keep their default
    assert s.cluster_sim_threshold == 0.9
    assert s.topic_match_threshold == 0.5
    assert s.question_min_score == 4
    assert s.ocr_enabled is True
    assert s.allowed_webhook_hosts == {"eceh-backend", "other:8080"}
    assert s.gemini_api_key.get_secret_value() == "secret-key"
    assert "secret-key" not in repr(s)
    assert s.option_defaults().ocr is True


def test_values_from_env_file(clean_env, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("QGEN_MAX_FILES=3\nQGEN_LLM_MODEL_LO=from-file\nGEMINI_API_KEY=file-key\n", encoding="utf-8")
    s = Settings(_env_file=env_file)
    assert s.max_files == 3
    assert s.llm_model_lo == "from-file"
    assert s.gemini_api_key.get_secret_value() == "file-key"

    clean_env.setenv("QGEN_MAX_FILES", "7")  # a real variable wins over the file
    assert Settings(_env_file=env_file).max_files == 7


def test_invalid_value_is_rejected(clean_env):
    clean_env.setenv("QGEN_CLUSTER_SIM_THRESHOLD", "1.5")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_get_settings_is_cached_until_cleared(clean_env):
    clean_env.setenv("QGEN_LO_BATCH_SIZE", "5")
    first = get_settings()
    assert first.lo_batch_size == 5
    clean_env.setenv("QGEN_LO_BATCH_SIZE", "6")
    assert get_settings() is first
    get_settings.cache_clear()
    assert get_settings().lo_batch_size == 6


def test_setting_or_default(clean_env):
    clean_env.setenv("QGEN_EMBED_MODEL", "embed-x")
    assert setting_or_default("given", "embed_model") == "given"
    assert setting_or_default(0.0, "cluster_sim_threshold") == 0.0  # only None means "not given"
    assert setting_or_default(None, "embed_model") == "embed-x"


def test_service_settings_share_the_definitions(clean_env):
    clean_env.setenv("QGEN_MAX_FILES", "4")
    clean_env.setenv("QGEN_DEFAULT_LESSON_MINUTES", "60")
    service = ServiceSettings(_env_file=None)
    core = Settings(_env_file=None)
    assert isinstance(service, Settings)
    shared = set(Settings.model_fields)
    assert {k: v for k, v in service.model_dump().items() if k in shared} == core.model_dump()
    assert set(ServiceSettings.model_fields) - shared == {"backend", "mock_duration_s"}
    assert service.max_files == 4
    assert service.option_defaults() == OptionDefaults(ocr=False, lesson_minutes=60)


# --- source guards -----------------------------------------------------------------


def _python_sources(exclude=()):
    for path in QGEN_DIR.rglob("*.py"):
        rel = path.relative_to(QGEN_DIR).as_posix()
        if rel not in exclude:
            yield rel, path.read_text(encoding="utf-8")


def test_environment_is_read_only_in_config():
    pattern = re.compile(r"os\.environ|os\.getenv|getenv\(|load_dotenv|dotenv")
    offenders = [rel for rel, text in _python_sources(exclude={"config.py"}) if pattern.search(text)]
    assert offenders == []


def test_no_hardcoded_model_names_in_qgen():
    pattern = re.compile(r"[\"'](gemini|gpt|claude)-[\w.-]+[\"']")
    offenders = [
        (rel, m.group(0))
        for rel, text in _python_sources(exclude={"config.py"})
        for m in pattern.finditer(text)
    ]
    assert offenders == []


def test_no_removed_threshold_constants_in_qgen():
    removed = re.compile(
        r"\b(SIM_THRESHOLD|TOPIC_MATCH_THRESHOLD|EMBEDDING_MODEL|MAX_SOURCE_CHARS|REQUIRE_SOURCE_OVERLAP|"
        r"LO_MIN_FAITHFULNESS_SCORE|ITEM_MIN_\w+|FAITHFULNESS_MODEL|ANSWERABILITY_MODEL|TOPIC_EXTRACTION_MODEL)\b"
    )
    offenders = [(rel, m.group(0)) for rel, text in _python_sources() for m in removed.finditer(text)]
    assert offenders == []
