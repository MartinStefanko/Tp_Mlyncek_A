"""Shared access to the reference recording baseline_v6/ for replay tests.

baseline_v6/ is the recording of tools/baseline.py on eko.pdf with gemini-3.1-flash-lite
and gemini-3.5-flash (eval), with coverage rounds (QGEN_QUESTION_COVERAGE_ROUNDS) and the
drop reasons in the coverage prompt. The replay tests run the pipeline with the replay
client on its segments.json and compare the result with its outputs; the expected values
come only from these files. Until the recording exists locally the replay tests are
skipped ("čaká na baseline_v6").

Only metrics.json is in git; the recording (llm_calls.jsonl, segments.json, outputs) has
document text and stays local. Tests that need it call `require_baseline()`, which skips
the test when a file is missing.

The former baseline/ (gemini-2.5), baseline_v3/ (old question format, before step 13) and
baseline_v4/ (step 13, prompts with hints) and baseline_v5/ with baseline_v5_* (one coverage
call, commit dff7c00) are historical records only; no test uses them.
"""

import json
from pathlib import Path
from typing import Optional

import pytest

from qgen.config import Settings
from qgen.llm.adapters.replay import ReplayClient

BASELINE_DIR = Path(__file__).resolve().parent.parent / "baseline_v6"
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
        pytest.skip(f"čaká na {BASELINE_NAME} (chýba lokálne: {', '.join(missing)})")


def load_json(name):
    with open(BASELINE_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def replay_client() -> ReplayClient:
    """Replays baseline_v6/llm_calls.jsonl as recorded (no transforms: same prompts)."""
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
    "question_coverage_rounds",
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


# --- split into lessons (step 16): short recordings on top of baseline_v6 -------------------
# tools/baseline.py --base baseline_v6 --split JSON records only the calls of the split and
# output steps (lesson titles, syllabus plan, syllabus embeddings). Replay = baseline_v6 +
# the short recording. Only metrics.json of each is in git.

SPLIT_RECORDINGS = ("baseline_v6_single", "baseline_v6_topics", "baseline_v6_lessons3", "baseline_v6_syllabus_text")
SPLIT_REQUIRED_FILES = ("llm_calls.jsonl", "units.json", "report.json", "metrics.json")


def split_dir(name: str) -> Path:
    return BASELINE_DIR.parent / name


def require_split_recording(name: str) -> None:
    require_baseline()
    missing = [f for f in SPLIT_REQUIRED_FILES if not (split_dir(name) / f).is_file()]
    if missing:
        pytest.skip(f"čaká na {name} (chýba lokálne: {', '.join(missing)})")


def load_split_json(name: str, file_name: str):
    return json.loads((split_dir(name) / file_name).read_text(encoding="utf-8"))


def recorded_split(name: str) -> Optional[dict]:
    """"split" part of the metrics.json of the split recording `name` (lessons,
    sections_per_lesson, warnings, ...); None when it is missing locally. Tests that only
    replay the pipeline compare with it when it exists, else they check properties only."""
    path = split_dir(name) / "metrics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["split"]


def recorded_sections_per_lesson(split: dict) -> list[list[int]]:
    """Section ids with an accepted question per lesson ("S-12" -> 12), in recorded order."""
    return [[int(section_id[2:]) for section_id in lesson] for lesson in split["sections_per_lesson"]]


def split_replay_client(name: str) -> ReplayClient:
    """baseline_v6 and the split recording `name` joined (recording index 1 = the split one)."""
    return ReplayClient([BASELINE_DIR / "llm_calls.jsonl", split_dir(name) / "llm_calls.jsonl"])
