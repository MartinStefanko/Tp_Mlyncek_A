"""Shared access to the reference recording baseline_v3/ for replay tests.

baseline_v3/ is the recording of tools/baseline.py on eko.pdf with gemini-3.1-flash-lite
and gemini-3.5-flash (eval), made at commit 42a7cd1 (see metrics.json "run"). The replay
tests run the pipeline with the replay client on its segments.json and compare the result
with its outputs; the expected values come only from these files.

Only metrics.json is in git; the recording (llm_calls.jsonl, segments.json, outputs) has
document text and stays local. Tests that need it call `require_baseline()`, which skips
the test when a file is missing.

The former baseline/ (gemini-2.5, before the generation prompt changes) is a historical
record only; no test uses it.
"""

import json
from pathlib import Path

import pytest

from qgen.config import Settings
from qgen.llm.adapters.replay import ReplayClient

BASELINE_DIR = Path(__file__).resolve().parent.parent / "baseline_v3"
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
        pytest.skip(f"Recording baseline_v3 is missing locally: {', '.join(missing)}")


def load_json(name):
    with open(BASELINE_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def replay_client() -> ReplayClient:
    """Replays baseline_v3/llm_calls.jsonl as recorded (no transforms: same prompts)."""
    return ReplayClient(BASELINE_DIR / "llm_calls.jsonl")


def baseline_models() -> dict[str, str]:
    """Every model of the recorded run, by Settings field, from metrics.json "run".
    Passed explicitly, so the replay does not depend on the code defaults, on the model
    inheritance or on a local .env."""
    roles = json.loads((BASELINE_DIR / "metrics.json").read_text(encoding="utf-8"))["run"]["models_by_role"]
    assert roles["lo"] == roles["prereq"]
    assert roles["check"] == roles["topics"] == roles["split"]
    return {
        "llm_model": roles["check"],
        "llm_model_lo": roles["lo"],
        "llm_model_questions": roles["questions"],
        "llm_model_eval": roles["eval"],
        "embed_model": roles["embed"],
    }


def baseline_settings() -> Settings:
    """Code defaults plus the recorded models; a local .env or QGEN_* variables must not
    change a replay run."""
    return Settings.model_construct(**baseline_models())
