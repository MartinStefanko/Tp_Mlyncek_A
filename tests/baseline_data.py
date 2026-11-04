"""Shared access to the baseline recording (baseline/) for replay tests.

The recording (llm_calls.jsonl, segments.json, ...) is not in git. Tests that need it
call `require_baseline()`, which skips the test when a file is missing.
"""

import json
from pathlib import Path

import pytest

from qgen.config import Settings
from tests.recording_transforms import REMOVED_ITEM_FIELDS, step4_transforms
from tests.replay_client import ReplayClient

BASELINE_DIR = Path(__file__).resolve().parent.parent / "baseline"
REQUIRED_FILES = (
    "llm_calls.jsonl",
    "segments.json",
    "metrics.json",
    "learning_objects.json",
    "learning_objects_all.json",
    "questions.json",
    "questions_all.json",
)


def require_baseline() -> None:
    missing = [name for name in REQUIRED_FILES if not (BASELINE_DIR / name).is_file()]
    if missing:
        pytest.skip(f"Baseline recording is missing locally: {', '.join(missing)}")


def load_json(name):
    with open(BASELINE_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def replay_client() -> ReplayClient:
    return ReplayClient(BASELINE_DIR / "llm_calls.jsonl", transforms=step4_transforms())


# Model of the checks (topics, faithfulness, answerability) in the baseline run. It is not in
# metrics.json; the baseline used the former constants, all gemini-2.5-flash-lite.
BASELINE_CHECK_MODEL = "gemini-2.5-flash-lite"


def baseline_models() -> dict[str, str]:
    """Every model of the baseline run, by Settings field. Passed explicitly, so the replay
    does not depend on the code defaults or on the model inheritance (step 7)."""
    models = json.loads((BASELINE_DIR / "metrics.json").read_text(encoding="utf-8"))["models"]
    assert models["lo_generation"] == models["lo_prerequisites"]
    return {
        "llm_model": BASELINE_CHECK_MODEL,
        "llm_model_lo": models["lo_generation"],
        "llm_model_questions": models["item_generation"],
        "llm_model_eval": models["item_evaluation"],
    }


def baseline_settings() -> Settings:
    """Code defaults plus the baseline models; a local .env or QGEN_* variables must not
    change a replay run."""
    return Settings.model_construct(**baseline_models())


def without_removed_fields(items):
    return [{k: v for k, v in item.items() if k not in REMOVED_ITEM_FIELDS} for item in items]
