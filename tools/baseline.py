"""Baseline run of the current pipeline in Hlavna_cast/ without Streamlit.

Repeats exactly what app.py does on "Spustiť extrakciu textu":
    pdfs_to_text -> generate_lo_pipeline -> generate_all_items
with the same models and parameters, but passes a RecordingClient through the
existing `client` parameter so every LLM / embedding call is stored in JSONL.

Usage (from the repo root):
    venv/Scripts/python.exe tools/baseline.py tests/fixtures/eko.pdf [more.pdf ...] [--out baseline]

Outputs in --out (default: baseline/):
    llm_calls.jsonl                    recording of all LLM and embedding calls
    segments.json                      extracted pages (pipeline input after extraction)
    learning_objects.json / _all.json  final / all LO (same export as app.py)
    questions.json / _all.json         final / all items (same export as app.py)
    reports/                           the pipeline's own reports (output_dir)
    metrics.json                       summary metrics taken from those reports
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "Hlavna_cast"
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import item_pipeline  # noqa: E402
import llm_client  # noqa: E402
import lo_pipeline  # noqa: E402
from item_pipeline import generate_all_items  # noqa: E402
from lo_pipeline import generate_lo_pipeline  # noqa: E402
from outputs import save_learning_objects_json_txt, save_questions_json_txt  # noqa: E402
from recording_client import RecordingClient  # noqa: E402
from text_extraction import pdfs_to_text  # noqa: E402

# Same values as Hlavna_cast/app.py
LO_GENERATION_MODEL = "gemini-2.5-flash-lite"
LO_PREREQ_MODEL = "gemini-2.5-flash-lite"
ITEM_GENERATION_MODEL = "gemini-2.5-flash-lite"
ITEM_EVALUATION_MODEL = "gemini-2.5-flash"
ITEM_EVALUATION_BATCH_SIZE = 20
LO_BATCH_SIZE = 10
ITEM_BATCH_SIZE = 10

# Report save functions whose input dicts we capture (module -> function names).
CAPTURED_REPORTS = {
    lo_pipeline: [
        "save_topic_coverage_report",
        "save_lo_relevance_to_segment_report",
        "save_lo_faithfulness_report",
        "save_lo_validation_report",
    ],
    item_pipeline: [
        "save_item_validation_report",
        "save_item_relevance_to_lo_report",
        "save_item_faithfulness_report",
        "save_item_answerability_report",
        "save_python_code_syntax_report",
        "save_python_code_runtime_report",
        "save_python_code_correctness_report",
    ],
}


def install_report_capture(captured: dict) -> list:
    """Wrap the pipeline's report save functions so their report dicts are kept.

    The original function is still called, so the pipeline writes its reports as usual.
    Returns a list of (module, name, original) for restoring.
    """
    restore = []
    for module, names in CAPTURED_REPORTS.items():
        for name in names:
            original = getattr(module, name)

            def wrapper(report, output_dir, *args, _orig=original, _name=name, **kwargs):
                captured[_name] = report
                return _orig(report, output_dir, *args, **kwargs)

            setattr(module, name, wrapper)
            restore.append((module, name, original))
    return restore


class UnexpectedClientCreation(RuntimeError):
    pass


def install_client_guard() -> tuple:
    """Fail loudly if any module builds its own client instead of using the injected one."""
    genai_module = llm_client.genai  # the vendor module, reached without importing it here
    original = genai_module.Client

    def guarded(*args, **kwargs):
        raise UnexpectedClientCreation(
            "A pipeline module tried to create its own LLM client; "
            "the recording client would be bypassed."
        )

    genai_module.Client = guarded
    return genai_module, original


def stats_of(captured: dict, name: str) -> dict:
    return (captured.get(name) or {}).get("stats", {}) or {}


def build_metrics(args, pdf_paths, captured, lo_report, item_report, los, items, timings, recorder):
    coverage = stats_of(captured, "save_topic_coverage_report")
    lo_faith = stats_of(captured, "save_lo_faithfulness_report")
    lo_rel = stats_of(captured, "save_lo_relevance_to_segment_report")
    item_faith = stats_of(captured, "save_item_faithfulness_report")
    item_ans = stats_of(captured, "save_item_answerability_report")
    item_rel = stats_of(captured, "save_item_relevance_to_lo_report")
    lo_details = (lo_report or {}).get("details", {})
    item_details = (item_report or {}).get("details", {})

    inputs = []
    for p in pdf_paths:
        inputs.append({
            "name": os.path.basename(p),
            "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest(),
        })

    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": inputs,
        "models": {
            "lo_generation": LO_GENERATION_MODEL,
            "lo_prerequisites": LO_PREREQ_MODEL,
            "item_generation": ITEM_GENERATION_MODEL,
            "item_evaluation": ITEM_EVALUATION_MODEL,
            "item_evaluation_batch_size": ITEM_EVALUATION_BATCH_SIZE,
            "lo_batch_size": LO_BATCH_SIZE,
            "item_batch_size": ITEM_BATCH_SIZE,
        },
        "document_language": lo_details.get("document_language"),
        "is_python_document": item_details.get("is_python_document"),
        "topic_coverage": {
            "coverage_percent": coverage.get("coverage_percent"),
            "topics_total": coverage.get("topics_total"),
            "topics_covered": coverage.get("topics_covered"),
            "similarity_threshold": coverage.get("similarity_threshold"),
        },
        "lo_faithfulness": {
            "average_faithfulness_score": lo_faith.get("average_faithfulness_score"),
            "los_evaluated": lo_faith.get("los_evaluated"),
        },
        "lo_relevance_to_segment": {
            "average_similarity": lo_rel.get("average_similarity"),
        },
        "item_faithfulness": {
            "average_faithfulness_score": item_faith.get("average_faithfulness_score"),
            "items_evaluated": item_faith.get("items_evaluated"),
            "faithful_items_percent": item_faith.get("faithful_items_percent"),
        },
        "item_answerability": {
            "average_answerability_score": item_ans.get("average_answerability_score"),
            "items_evaluated": item_ans.get("items_evaluated"),
            "answerable_items_percent": item_ans.get("answerable_items_percent"),
        },
        "item_relevance_to_lo": {
            "average_similarity": item_rel.get("average_similarity"),
        },
        "counts": {
            "pages": timings.get("pages"),
            "los_all": lo_details.get("los_count_all", lo_details.get("los_count")),
            "los_accepted": len(los),
            "items_all": item_details.get("items_count_all"),
            "items_accepted": len(items),
        },
        "timings_seconds": {
            "extraction": timings.get("extraction"),
            "lo_pipeline_wall": timings.get("lo_pipeline_wall"),
            "lo_generation": (lo_report or {}).get("generation_seconds"),
            "lo_evaluation": (lo_report or {}).get("evaluation_seconds"),
            "item_pipeline_wall": timings.get("item_pipeline_wall"),
            "item_generation": (item_report or {}).get("generation_seconds"),
            "item_evaluation": (item_report or {}).get("evaluation_seconds"),
            "total_wall": timings.get("total_wall"),
        },
        "llm_calls": {
            "by_method": dict(recorder.counts),
            "failed": recorder.errors,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdfs", nargs="+", help="Input PDF file(s), e.g. tests/fixtures/eko.pdf")
    parser.add_argument("--out", default=str(REPO_ROOT / "baseline"), help="Output directory (default: baseline/)")
    args = parser.parse_args(argv)

    pdf_paths = [str(Path(p).resolve()) for p in args.pdfs]
    for p in pdf_paths:
        if not Path(p).is_file():
            parser.error(f"File not found: {p}")

    out_dir = Path(args.out).resolve()
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Real client via the pipeline's own factory (loads GEMINI_API_KEY from .env).
    real_client = llm_client.get_client()
    client = RecordingClient(real_client, out_dir / "llm_calls.jsonl")

    captured: dict = {}
    restore = install_report_capture(captured)
    genai_module, original_client_cls = install_client_guard()

    timings: dict = {}
    total_start = time.perf_counter()
    try:
        # 1) Extraction – same input shape as app.py ({"path", "name"}).
        pdf_inputs = [{"path": p, "name": os.path.basename(p)} for p in pdf_paths]
        t0 = time.perf_counter()
        segments = pdfs_to_text(pdf_inputs)
        timings["extraction"] = round(time.perf_counter() - t0, 4)
        timings["pages"] = len(segments)
        with open(out_dir / "segments.json", "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        # 2) LO pipeline – same arguments as app.py, plus the recording client.
        t0 = time.perf_counter()
        los, lo_report = generate_lo_pipeline(
            segments,
            batch_size=LO_BATCH_SIZE,
            generation_model=LO_GENERATION_MODEL,
            prerequisites_model=LO_PREREQ_MODEL,
            output_dir=str(reports_dir),
            client=client,
            verbose=True,
            return_metrics=True,
        )
        timings["lo_pipeline_wall"] = round(time.perf_counter() - t0, 4)
        save_learning_objects_json_txt(los, str(out_dir), all_los=lo_report.get("all_los"))

        # 3) Item pipeline – app.py runs it only when some LO passed the filter.
        items, item_report = [], None
        if los:
            t0 = time.perf_counter()
            items, item_report = generate_all_items(
                los,
                segments,
                batch_size=ITEM_BATCH_SIZE,
                evaluation_batch_size=ITEM_EVALUATION_BATCH_SIZE,
                generation_model=ITEM_GENERATION_MODEL,
                evaluation_model=ITEM_EVALUATION_MODEL,
                output_dir=str(reports_dir),
                client=client,
                verbose=True,
                return_metrics=True,
            )
            timings["item_pipeline_wall"] = round(time.perf_counter() - t0, 4)
            save_questions_json_txt(items, str(out_dir), all_items=item_report.get("all_items"))
        timings["total_wall"] = round(time.perf_counter() - total_start, 4)
    finally:
        genai_module.Client = original_client_cls
        for module, name, original in restore:
            setattr(module, name, original)
        client.close()

    metrics = build_metrics(args, pdf_paths, captured, lo_report, item_report, los, items, timings, client.recorder)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== BASELINE METRICS ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
