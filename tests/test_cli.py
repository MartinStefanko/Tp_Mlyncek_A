"""python -m qgen.cli: writes results and reports to a folder, prints progress."""

import json
import sys
from pathlib import Path

import pytest

import qgen.cli as cli
from qgen.config import Settings
from qgen.progress import STAGE_EXTRACT
from tests.baseline_data import (
    BASELINE_DIR,
    baseline_settings,
    load_json,
    replay_client,
    require_baseline,
)

EKO_PDF = Path(__file__).resolve().parent / "fixtures" / "eko.pdf"

# Reports whose content does not depend on time; they must match the baseline files.
STABLE_REPORTS = (
    "document_topics.txt",
    "topic_coverage_report.txt",
    "learning_objects_validation_report.txt",
    "lo_relevance_to_segment_report.txt",
    "lo_faithfulness_report.txt",
    "item_relevance_to_lo_report.txt",
    "item_faithfulness_report.txt",
    "item_answerability_report.txt",
    "section_diagnostics.json",
    "question_diagnostics.json",
)
ALL_REPORTS = STABLE_REPORTS + (
    "item_validation_report.txt",
    "lo_processing_time_report.txt",
    "item_processing_time_report.txt",
)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture
def baseline_segments_extraction(monkeypatch):
    """Replace PDF extraction by baseline_v3/segments.json (the recording starts after extraction)."""
    calls = []

    def fake_pdfs_to_text(pdf_inputs, ocr=None, progress_callback=None, settings=None):
        calls.append({"inputs": pdf_inputs, "ocr": ocr})
        segments = load_json("segments.json")
        if progress_callback:
            progress_callback(STAGE_EXTRACT, 0, len(segments))
            progress_callback(STAGE_EXTRACT, len(segments), len(segments))
        return segments

    monkeypatch.setattr(cli, "pdfs_to_text", fake_pdfs_to_text)
    return calls


def test_cli_writes_results_and_reports(tmp_path, capsys, baseline_segments_extraction):
    require_baseline()
    out = tmp_path / "out"
    replay = replay_client()
    code = cli.main([str(EKO_PDF), "--out", str(out)], llm=replay, embedder=replay, settings=baseline_settings())
    assert code == 0

    assert _read_json(out / "segments.json") == load_json("segments.json")
    assert _read_json(out / "learning_objects.json") == load_json("learning_objects.json")
    assert _read_json(out / "learning_objects_all.json") == load_json("learning_objects_all.json")
    assert _read_json(out / "questions.json") == load_json("questions.json")
    assert _read_json(out / "questions_all.json") == load_json("questions_all.json")

    written = sorted(p.name for p in (out / "reports").iterdir())
    assert written == sorted(ALL_REPORTS)
    for name in STABLE_REPORTS:
        baseline_file = BASELINE_DIR / "reports" / name
        if baseline_file.is_file():
            assert (out / "reports" / name).read_text(encoding="utf-8") == baseline_file.read_text(encoding="utf-8"), name

    assert baseline_segments_extraction[0]["ocr"] is None  # no --ocr: QGEN_OCR_ENABLED decides
    printed = capsys.readouterr().out
    for stage in ("extract", "language", "learning_objects", "lo_evaluation", "questions", "q_evaluation"):
        assert f"[{stage:<16}]" in printed, stage
    assert "Done:" in printed


def test_cli_passes_ocr_and_language(tmp_path, monkeypatch, baseline_segments_extraction):
    require_baseline()  # the fake extraction reads baseline_v3/segments.json
    captured = {}

    def fake_run_pipeline(segments, **kwargs):
        captured.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    with pytest.raises(SystemExit):
        cli.main(
            [str(EKO_PDF), "--out", str(tmp_path), "--ocr", "--language", "en"],
            llm=object(), embedder=object(),
            settings=Settings.model_construct(),
        )
    assert baseline_segments_extraction[0]["ocr"] is True
    assert captured["language"] == "en"


def test_cli_reports_missing_easyocr(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "easyocr", None)
    code = cli.main([str(EKO_PDF), "--out", str(tmp_path), "--ocr"], llm=object(), embedder=object(), settings=Settings.model_construct())
    assert code == 2
    assert "easyocr is not installed" in capsys.readouterr().err


def test_cli_rejects_bad_arguments(tmp_path):
    with pytest.raises(SystemExit) as missing_file:
        cli.main([str(tmp_path / "missing.pdf"), "--out", str(tmp_path)], llm=object(), embedder=object())
    assert missing_file.value.code == 2
    with pytest.raises(SystemExit):
        cli.main([str(EKO_PDF), "--out", str(tmp_path), "--language", "de"], llm=object(), embedder=object())
    with pytest.raises(SystemExit):
        cli.main([str(EKO_PDF)], llm=object(), embedder=object())  # --out is required
