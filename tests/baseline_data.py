"""Shared access to the reference recording baseline_v5/ for replay tests.

baseline_v5/ is the recording of tools/baseline.py on eko.pdf with gemini-3.1-flash-lite
and gemini-3.5-flash (eval), made with commit dff7c00 (metrics.json "run": git_dirty
false; hints built by code, question filter with each threshold on its own). The replay
tests run the pipeline with the replay client on its segments.json and compare the result
with its outputs; the expected values come only from these files.

Only metrics.json is in git; the recording (llm_calls.jsonl, segments.json, outputs) has
document text and stays local. Tests that need it call `require_baseline()`, which skips
the test when a file is missing.

The former baseline/ (gemini-2.5), baseline_v3/ (old question format, before step 13) and
baseline_v4/ (step 13, prompts with hints) are historical records only; no test uses them.
"""

import json
from pathlib import Path

import pytest

from qgen.config import Settings
from qgen.llm.adapters.replay import ReplayClient

BASELINE_DIR = Path(__file__).resolve().parent.parent / "baseline_v5"
BASELINE_NAME = BASELINE_DIR.name
REQUIRED_FILES = (
    "llm_calls.jsonl",
    "segments.json",
    "metrics.json",
    "learning_objects.json",
    "learning_objects_all.json",
    "questions.json",
    "questions_all.json",
)


# Reason why the replay tests are skipped; None = run them on BASELINE_DIR.
AWAITING_BASELINE = None


def require_baseline() -> None:
    if AWAITING_BASELINE:
        pytest.skip(AWAITING_BASELINE)
    missing = [name for name in REQUIRED_FILES if not (BASELINE_DIR / name).is_file()]
    if missing:
        pytest.skip(f"Recording {BASELINE_NAME} is missing locally: {', '.join(missing)}")


def load_json(name):
    with open(BASELINE_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def replay_client() -> ReplayClient:
    """Replays baseline_v5/llm_calls.jsonl as recorded (no transforms: same prompts)."""
    return ReplayClient(BASELINE_DIR / "llm_calls.jsonl")


def load_metrics() -> dict:
    return json.loads((BASELINE_DIR / "metrics.json").read_text(encoding="utf-8"))


def baseline_models() -> dict[str, str]:
    """Every model of the recorded run, by Settings field, from metrics.json "run".
    Passed explicitly, so the replay does not depend on the code defaults, on the model
    inheritance or on a local .env."""
    roles = load_metrics()["run"]["models_by_role"]
    assert roles["lo"] == roles["prereq"]
    assert roles["check"] == roles["topics"] == roles["split"]
    return {
        "llm_model": roles["check"],
        "llm_model_lo": roles["lo"],
        "llm_model_questions": roles["questions"],
        "llm_model_eval": roles["eval"],
        "embed_model": roles["embed"],
    }


# Settings of the question pipeline recorded in metrics.json "questions" -> "settings" (step 13).
QUESTION_SETTING_FIELDS = (
    "question_generation_batch_size",
    "question_extra_rounds",
    "question_dedup_threshold",
    "llm_temperature",
    "llm_max_output_tokens",
)


def baseline_question_settings() -> dict:
    """The question settings of the recorded run, so the replay does not depend on the
    code defaults (e.g. a later change of the dedup threshold)."""
    recorded = load_metrics()["questions"]["settings"]
    return {name: recorded[name] for name in QUESTION_SETTING_FIELDS}


def baseline_concurrency() -> dict:
    """llm_max_concurrency of the recorded run (metrics.json "run"). The results do not
    depend on it (ids and order follow the sequential run), it is taken over so the replay
    runs like the recording."""
    return {"llm_max_concurrency": load_metrics()["run"]["llm_max_concurrency"]}


def baseline_settings() -> Settings:
    """Code defaults plus the recorded models, question settings and concurrency; a local
    .env or QGEN_* variables must not change a replay run."""
    return Settings.model_construct(**baseline_models(), **baseline_question_settings(),
                                    **baseline_concurrency())
