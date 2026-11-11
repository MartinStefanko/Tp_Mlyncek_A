"""Shared access to the reference recording baseline_v4/ for replay tests.

baseline_v4/ is the recording of tools/baseline.py on eko.pdf with gemini-3.1-flash-lite
and gemini-3.5-flash (eval), made with the code of step 13 before its commit (metrics.json
"run" says git_commit 9e1eddf, git_dirty true; the code is commit c9cb618). The replay
tests run the pipeline with the replay client on its segments.json and compare the result
with its outputs; the expected values come only from these files.

Only metrics.json is in git; the recording (llm_calls.jsonl, segments.json, outputs) has
document text and stays local. Tests that need it call `require_baseline()`, which skips
the test when a file is missing.

The former baseline/ (gemini-2.5) and baseline_v3/ (old question format, before step 13)
are historical records only; no test uses them.

Step 13c changed the prompts (no hints), so the prompts of baseline_v4 no longer match:
every replay test is skipped ("čaká na baseline_v5") until the new recording baseline_v5
exists. Then point BASELINE_DIR to it and set AWAITING_BASELINE = None.
"""

import json
from pathlib import Path

import pytest

from qgen.config import Settings
from qgen.llm.adapters.replay import ReplayClient

BASELINE_DIR = Path(__file__).resolve().parent.parent / "baseline_v4"
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
AWAITING_BASELINE = "čaká na baseline_v5"


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
    """Replays baseline_v4/llm_calls.jsonl as recorded (no transforms: same prompts)."""
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
    """The question settings of the recorded run (e.g. dedup threshold 0.95, while the
    code default is 0.94), so the replay does not depend on the code defaults."""
    recorded = load_metrics()["questions"]["settings"]
    return {name: recorded[name] for name in QUESTION_SETTING_FIELDS}


def baseline_settings() -> Settings:
    """Code defaults plus the recorded models and question settings; a local .env or
    QGEN_* variables must not change a replay run."""
    return Settings.model_construct(**baseline_models(), **baseline_question_settings())
