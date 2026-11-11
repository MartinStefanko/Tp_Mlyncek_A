"""Replay of the reference recording baseline_v5/: the pipeline must give the same results.

Runs the LO pipeline and the question pipeline on baseline_v5/segments.json (no PDF, no
OCR) with the replay client, with the same arguments as tools/baseline.py, and compares the
results with the recorded outputs. Every prompt must be byte-identical to the recording: a
changed prompt is a call that is not in the recording (UnrecordedCallError), and every
recorded call must be used.

The recording is not in git. When it is missing, the test is skipped.
"""

import json

import pytest

from qgen.cli import normalize_lo_for_export
from qgen.lo.pipeline import generate_lo_pipeline
from qgen.questions.pipeline import generate_all_items
from tests.baseline_data import baseline_settings, load_json, replay_client, require_baseline


def _as_json(value):
    """Round trip through JSON, so the result has the same types as the saved baseline."""
    return json.loads(json.dumps(value, ensure_ascii=False))


@pytest.fixture(scope="module")
def replay_run():
    require_baseline()
    models = load_json("metrics.json")["models"]
    segments = load_json("segments.json")
    client = replay_client()
    # Code defaults plus the recorded models (explicit); a local .env or QGEN_* variables
    # must not change the replay.
    settings = baseline_settings()
    assert settings.llm_model_eval == models["item_evaluation"]

    lo_result = generate_lo_pipeline(
        segments,
        batch_size=models["lo_batch_size"],
        generation_model=models["lo_generation"],
        prerequisites_model=models["lo_prerequisites"],
        llm=client,
        embedder=client,
        verbose=False,
        settings=settings,
    )
    items, all_items, item_reports = [], [], {}
    if lo_result.learning_objects:
        item_result = generate_all_items(
            lo_result.learning_objects,
            segments,
            batch_size=models["item_batch_size"],
            evaluation_batch_size=models["item_evaluation_batch_size"],
            generation_model=models["item_generation"],
            evaluation_model=models["item_evaluation"],
            llm=client,
            embedder=client,
            verbose=False,
            settings=settings,
        )
        items, all_items, item_reports = item_result.questions, item_result.all_questions, item_result.reports
    return {
        "client": client,
        "los": lo_result.learning_objects,
        "all_los": lo_result.all_learning_objects,
        "items": items,
        "all_items": all_items,
        "reports": {**lo_result.reports, **item_reports},
    }


def test_learning_objects_match_baseline(replay_run):
    accepted = _as_json([normalize_lo_for_export(lo) for lo in replay_run["los"]])
    all_los = _as_json([normalize_lo_for_export(lo) for lo in replay_run["all_los"]])
    assert accepted == load_json("learning_objects.json")
    assert all_los == load_json("learning_objects_all.json")


def test_questions_match_baseline(replay_run):
    assert _as_json(replay_run["items"]) == load_json("questions.json")
    assert _as_json(replay_run["all_items"]) == load_json("questions_all.json")


def test_counts_match_metrics(replay_run):
    counts = load_json("metrics.json")["counts"]
    assert len(replay_run["all_los"]) == counts["los_all"]
    assert len(replay_run["los"]) == counts["los_accepted"]
    assert len(replay_run["all_items"]) == counts["items_all"]
    assert len(replay_run["items"]) == counts["items_accepted"]


def test_diagnostics_match_metrics(replay_run):
    metrics = load_json("metrics.json")
    assert _as_json(replay_run["reports"]["section_diagnostics"]) == metrics["section_diagnostics"]
    question_diagnostics = {
        k: v for k, v in replay_run["reports"]["question_diagnostics"].items() if k != "sections_without_content"
    }
    assert _as_json(question_diagnostics) == metrics["question_diagnostics"]


def test_every_recorded_call_is_used(replay_run):
    unused = replay_run["client"].unused_records()
    assert unused == [], [(r["seq"], r["method"], r["model"]) for r in unused]
