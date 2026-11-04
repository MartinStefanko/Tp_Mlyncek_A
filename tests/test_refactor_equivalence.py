"""Step 4: moving Hlavna_cast/ into qgen/ must not change the pipeline results.

Runs the LO pipeline and the question pipeline on baseline/segments.json (no PDF, no OCR)
with the replay client and compares the results with the baseline outputs. The only
allowed differences are listed in tests/recording_transforms.py.

The recording is not in git. When it is missing, the test is skipped.
"""

import json
from pathlib import Path

import pytest

from qgen.lo.pipeline import generate_lo_pipeline
from qgen.outputs import normalize_lo_for_export
from qgen.questions.pipeline import generate_all_items
from tests.recording_transforms import REMOVED_CALL_MARKERS, REMOVED_ITEM_FIELDS, step4_transforms
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


def _load_json(name):
    with open(BASELINE_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def _as_json(value):
    """Round trip through JSON, so the result has the same types as the saved baseline."""
    return json.loads(json.dumps(value, ensure_ascii=False))


def _without_removed_fields(items):
    return [{k: v for k, v in item.items() if k not in REMOVED_ITEM_FIELDS} for item in items]


@pytest.fixture(scope="module")
def replay_run(tmp_path_factory):
    missing = [name for name in REQUIRED_FILES if not (BASELINE_DIR / name).is_file()]
    if missing:
        pytest.skip(f"Baseline recording is missing locally: {', '.join(missing)}")

    models = _load_json("metrics.json")["models"]
    segments = _load_json("segments.json")
    client = ReplayClient(BASELINE_DIR / "llm_calls.jsonl", transforms=step4_transforms())
    reports_dir = tmp_path_factory.mktemp("reports")

    los, lo_report = generate_lo_pipeline(
        segments,
        batch_size=models["lo_batch_size"],
        generation_model=models["lo_generation"],
        prerequisites_model=models["lo_prerequisites"],
        output_dir=str(reports_dir),
        client=client,
        verbose=False,
        return_metrics=True,
    )
    items, item_report = [], {"all_items": []}
    if los:
        items, item_report = generate_all_items(
            los,
            segments,
            batch_size=models["item_batch_size"],
            evaluation_batch_size=models["item_evaluation_batch_size"],
            generation_model=models["item_generation"],
            evaluation_model=models["item_evaluation"],
            output_dir=str(reports_dir),
            client=client,
            verbose=False,
            return_metrics=True,
        )
    return {
        "client": client,
        "los": los,
        "lo_report": lo_report,
        "items": items,
        "item_report": item_report,
    }


def test_learning_objects_match_baseline(replay_run):
    accepted = _as_json([normalize_lo_for_export(lo) for lo in replay_run["los"]])
    all_los = _as_json([normalize_lo_for_export(lo) for lo in replay_run["lo_report"]["all_los"]])
    assert accepted == _load_json("learning_objects.json")
    assert all_los == _load_json("learning_objects_all.json")


def test_questions_match_baseline(replay_run):
    assert _as_json(replay_run["items"]) == _without_removed_fields(_load_json("questions.json"))
    assert _as_json(replay_run["item_report"]["all_items"]) == _without_removed_fields(
        _load_json("questions_all.json")
    )


def test_questions_have_no_removed_fields(replay_run):
    for item in replay_run["items"] + replay_run["item_report"]["all_items"]:
        assert not set(REMOVED_ITEM_FIELDS) & set(item), item.get("id")


def test_only_removed_calls_are_unused(replay_run):
    unused = replay_run["client"].unused_records()
    unexpected = [
        r for r in unused
        if not (isinstance(r["contents"], str) and any(m in r["contents"] for m in REMOVED_CALL_MARKERS))
    ]
    assert unexpected == [], [(r["seq"], r["method"], r["model"]) for r in unexpected]
    assert unused, "the removed Python classification call should be left unused"


def test_every_transform_was_applied(replay_run):
    for transform in replay_run["client"].transforms:
        assert transform.hits > 0, f"transform {transform.name} changed nothing in the recording"
