"""progress_callback(stage, done, total): stage names from the contract, start and end of every step."""

import hashlib
import re
from itertools import groupby
from pathlib import Path
from typing import get_args

import pytest

from qgen.cli import normalize_lo_for_export
from qgen.config import Settings
from qgen.contract.jobs import JobStage
from qgen.extraction.document_language import resolve_document_language
from qgen.extraction.text_extraction import pdfs_to_text
from qgen.pipeline import run_pipeline
from qgen.progress import PIPELINE_STAGES, SubProgress, batch_count
from tests.baseline_data import (
    baseline_settings,
    load_json,
    replay_client,
    require_baseline,
    without_removed_fields,
)

EKO_PDF = Path(__file__).resolve().parent / "fixtures" / "eko.pdf"


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, stage, done, total):
        self.calls.append((stage, done, total))

    def stages(self):
        return [stage for stage, _ in groupby(stage for stage, _, _ in self.calls)]

    def segments(self):
        """Consecutive calls grouped by stage: [(stage, [(done, total), ...]), ...]."""
        return [(stage, [(d, t) for _, d, t in group]) for stage, group in groupby(self.calls, key=lambda c: c[0])]


def assert_well_formed(recorder):
    for stage, calls in recorder.segments():
        assert calls[0][0] == 0, (stage, calls[:3])
        assert calls[-1][0] == calls[-1][1], (stage, calls[-3:])
        dones = [d for d, _ in calls]
        assert dones == sorted(dones), (stage, calls)
        assert all(0 <= d <= t for d, t in calls), (stage, calls)


def test_pipeline_stages_are_job_stages():
    assert set(PIPELINE_STAGES) <= set(get_args(JobStage))
    assert list(PIPELINE_STAGES) == list(get_args(JobStage))[: len(PIPELINE_STAGES)]


@pytest.fixture(scope="module")
def replay_progress():
    require_baseline()
    recorder = Recorder()
    client = replay_client()
    result = run_pipeline(
        load_json("segments.json"),
        llm=client,
        embedder=client,
        progress_callback=recorder,
        settings=baseline_settings(),
    )
    return recorder, result


def test_pipeline_reports_progress_with_stage_names(replay_progress):
    recorder, _ = replay_progress
    assert recorder.calls
    assert {stage for stage, _, _ in recorder.calls} <= set(get_args(JobStage))
    # Both pipelines detect the language (as the former app.py did).
    assert recorder.stages() == [
        "language",
        "learning_objects",
        "lo_evaluation",
        "language",
        "questions",
        "q_evaluation",
    ]
    assert_well_formed(recorder)


def test_progress_counts_batches(replay_progress):
    recorder, result = replay_progress
    settings = baseline_settings()
    segments = dict(recorder.segments())
    pages = len(load_json("segments.json"))

    lo_total = batch_count(pages, settings.lo_batch_size) + 2  # + clustering + prerequisites
    assert segments["learning_objects"][-1] == (lo_total, lo_total)
    assert len(segments["learning_objects"]) > 3  # reported after batches, not only start and end

    lo_count = len(result.all_learning_objects)
    question_batches = batch_count(len(result.learning_objects), settings.question_batch_size)
    assert segments["questions"][0] == (0, question_batches)
    assert segments["questions"][-1] == (question_batches, question_batches)
    assert [d for d, _ in segments["questions"]] == list(range(question_batches + 1))

    assert segments["lo_evaluation"][-1][1] <= 2 + batch_count(lo_count, settings.lo_batch_size)
    assert segments["q_evaluation"][-1][1] >= 1


def test_callback_does_not_change_results(replay_progress):
    _, result = replay_progress
    assert [normalize_lo_for_export(lo) for lo in result.learning_objects] == load_json("learning_objects.json")
    assert result.questions == without_removed_fields(load_json("questions.json"))


def test_extraction_reports_pages():
    recorder = Recorder()
    segments = pdfs_to_text([str(EKO_PDF)], ocr=False, progress_callback=recorder)
    assert recorder.calls[0] == ("extract", 0, len(segments))
    assert recorder.calls[-1] == ("extract", len(segments), len(segments))
    assert len(recorder.calls) == len(segments) + 1
    assert_well_formed(recorder)


def test_extraction_without_ocr_keeps_the_page_text():
    """Without OCR the pages are the baseline pages minus the OCR blocks (baseline ran with OCR)."""
    require_baseline()
    metrics = load_json("metrics.json")
    if metrics["inputs"][0]["sha256"] != hashlib.sha256(EKO_PDF.read_bytes()).hexdigest():
        pytest.skip("tests/fixtures/eko.pdf is not the baseline input")
    baseline = load_json("segments.json")
    segments = pdfs_to_text([{"path": str(EKO_PDF), "name": "eko.pdf"}], ocr=False)
    ocr_block = re.compile(r"\n\n\[OCR OBRÁZOK \d+\]\n.*", re.DOTALL)
    assert len(segments) == len(baseline)
    for page, expected in zip(segments, baseline):
        assert page == {**expected, "text": ocr_block.sub("", expected["text"])}


def test_language_option_skips_detection():
    recorder = Recorder()
    info = resolve_document_language(
        [{"page": 1, "text": "text"}],
        "en",
        llm=None,  # a detection call would need an LLM
        model="unused",
        settings=Settings.model_construct(),
        verbose=False,
        progress_callback=recorder,
    )
    assert info["language"] == "en"
    assert recorder.calls == [("language", 0, 1), ("language", 1, 1)]
    with pytest.raises(ValueError):
        resolve_document_language([], "de", None, "unused", Settings.model_construct(), False, None)


def test_sub_progress_shifts_into_the_parent_step():
    recorder = Recorder()
    sub = SubProgress(recorder, before=2, after=3)
    sub("lo_evaluation", 0, 4)
    sub("lo_evaluation", 4, 4)
    assert recorder.calls == [("lo_evaluation", 2, 9), ("lo_evaluation", 6, 9)]
    assert sub.sub_total == 4
    SubProgress(None)("questions", 1, 1)  # no callback: no error
