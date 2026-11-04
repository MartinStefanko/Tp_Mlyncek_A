"""Baseline run of the pipeline in qgen/ (originally Hlavna_cast/, run without Streamlit).

Repeats exactly what the former app.py did on "Spustiť extrakciu textu":
    pdfs_to_text -> generate_lo_pipeline -> generate_all_items
with the same models and parameters (OCR on, as app.py always did), but passes a
RecordingClient as `llm` and `embedder` (step 6: the LLM interface) so every LLM / embedding call
is stored in JSONL. The pipeline returns its reports; this tool writes them (with the
writers in qgen/cli.py).

Models: by default every role takes its model from the settings (.env / environment,
get_settings()), like the CLI: QGEN_LLM_MODEL, _LO, _QUESTIONS, _EVAL and QGEN_EMBED_MODEL.
--app-models uses the fixed models of the former app.py instead (APP_MODELS; Gemini 2.5,
which Google no longer offers to new users). Provider and keys always come from the
settings. Thresholds, batch sizes and attempts stay at the code defaults (= app.py values),
so .env cannot change them. Before the run the tool prints the model of every role and
calls factory.check_config().

The committed baseline/ is the reference of the replay tests; --out pointing at it is
refused unless --force is given.

Usage (from the repo root):
    venv/Scripts/python.exe tools/baseline.py tests/fixtures/eko.pdf [more.pdf ...] --out baseline_v2
        [--app-models] [--force]

Outputs in --out:
    llm_calls.jsonl                    recording of all LLM and embedding calls
    segments.json                      extracted pages (pipeline input after extraction)
    learning_objects.json / _all.json  final / all LO (same JSON export as the former app.py)
    questions.json / _all.json         final / all items (same JSON export as the former app.py)
    reports/                           the pipeline reports as text files
    metrics.json                       summary metrics taken from those reports, plus the
                                       models of every role, run date and git commit ("run")
"""

import argparse
import subprocess
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import qgen.llm.factory as llm_factory  # noqa: E402
from qgen.cli import save_learning_objects_json, save_questions_json, write_reports  # noqa: E402
from qgen.config import Settings, get_settings  # noqa: E402
from qgen.llm.errors import LLMConfigError  # noqa: E402
from qgen.extraction.text_extraction import pdfs_to_text  # noqa: E402
from qgen.lo.pipeline import generate_lo_pipeline  # noqa: E402
from qgen.questions.pipeline import generate_all_items  # noqa: E402
from recording_client import RecordingClient  # noqa: E402

# Fixed models of the former Hlavna_cast/app.py (--app-models), by Settings field.
APP_MODELS = {
    "llm_model": "gemini-2.5-flash-lite",            # checks: topics, faithfulness, answerability
    "llm_model_lo": "gemini-2.5-flash-lite",         # language, LO generation, prerequisites
    "llm_model_questions": "gemini-2.5-flash-lite",  # question generation
    "llm_model_eval": "gemini-2.5-flash",            # question quality score
    "embed_model": "gemini-embedding-001",
}
MODEL_FIELDS = tuple(APP_MODELS)
# Same values as the former Hlavna_cast/app.py (= code defaults of qgen.config).
ITEM_EVALUATION_BATCH_SIZE = 20
LO_BATCH_SIZE = 10
ITEM_BATCH_SIZE = 10
REFERENCE_BASELINE_DIR = REPO_ROOT / "baseline"

class UnexpectedClientCreation(RuntimeError):
    pass


def install_client_guard() -> dict:
    """Fail loudly if any module asks the factory for its own client instead of using the
    injected recording client. Returns the original factory functions."""
    original = {"get_llm": llm_factory.get_llm, "get_embedder": llm_factory.get_embedder}

    def guarded(*args, **kwargs):
        raise UnexpectedClientCreation(
            "A pipeline module tried to create its own LLM client; "
            "the recording client would be bypassed."
        )

    llm_factory.get_llm = guarded
    llm_factory.get_embedder = guarded
    return original


def select_models(env_settings: Settings, app_models: bool) -> dict:
    """Model of every Settings model field: from the settings, or APP_MODELS."""
    if app_models:
        return dict(APP_MODELS)
    return {field: getattr(env_settings, field) for field in MODEL_FIELDS}


def models_by_role(settings: Settings) -> dict:
    """Model of every factory role, plus "embed"."""
    roles = {role: llm_factory.model_for_role(role, settings) for role in llm_factory.ROLES}
    roles[llm_factory.EMBED_ROLE] = settings.embed_model
    return roles


def is_reference_baseline(out_dir: Path) -> bool:
    return Path(out_dir).resolve() == REFERENCE_BASELINE_DIR.resolve()


def git_version(repo: Path = REPO_ROOT) -> dict:
    """Commit of the code that made the run; dirty = uncommitted changes in tracked files."""
    def git(*args):
        result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        return result.stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def run_info(run_settings: Settings, models_source: str, started_at: datetime, version: dict) -> dict:
    """Which run the metrics belong to: date, code version, providers and models per role."""
    return {
        "started_at": started_at.isoformat(timespec="seconds"),
        "git_commit": version.get("commit"),
        "git_dirty": version.get("dirty"),
        "models_source": models_source,
        "llm_provider": run_settings.llm_provider,
        "embed_provider": run_settings.embed_provider,
        "models_by_role": models_by_role(run_settings),
    }


def stats_of(reports: dict, name: str) -> dict:
    return (reports.get(name) or {}).get("stats", {}) or {}


def build_metrics(args, pdf_paths, reports, lo_report, item_report, los, items, timings, recorder,
                  pipeline_settings: Settings, run: dict):
    coverage = stats_of(reports, "topic_coverage")
    lo_faith = stats_of(reports, "lo_faithfulness")
    lo_rel = stats_of(reports, "lo_relevance_to_segment")
    item_faith = stats_of(reports, "item_faithfulness")
    item_ans = stats_of(reports, "item_answerability")
    item_rel = stats_of(reports, "item_relevance_to_lo")
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
        "run": run,
        "inputs": inputs,
        # Same keys as the reference baseline/metrics.json (read by the replay tests).
        "models": {
            "lo_generation": pipeline_settings.llm_model_lo,
            "lo_prerequisites": pipeline_settings.llm_model_lo,
            "item_generation": pipeline_settings.llm_model_questions,
            "item_evaluation": pipeline_settings.llm_model_eval,
            "item_evaluation_batch_size": ITEM_EVALUATION_BATCH_SIZE,
            "lo_batch_size": LO_BATCH_SIZE,
            "item_batch_size": ITEM_BATCH_SIZE,
        },
        "document_language": lo_details.get("document_language"),
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
        # Section counts, questions per section, dropped questions, coverage warnings.
        "section_diagnostics": reports.get("section_diagnostics"),
        "question_diagnostics": {
            key: value
            for key, value in (reports.get("question_diagnostics") or {}).items()
            if key != "sections_without_content"
        } or None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdfs", nargs="+", help="Input PDF file(s), e.g. tests/fixtures/eko.pdf")
    parser.add_argument("--out", required=True,
                        help="Output directory, e.g. baseline_v2 (baseline/ only with --force)")
    parser.add_argument("--app-models", action="store_true",
                        help="Use the fixed models of the former app.py (Gemini 2.5) instead of the settings.")
    parser.add_argument("--force", action="store_true", help="Allow --out to be the reference baseline/.")
    args = parser.parse_args(argv)

    if is_reference_baseline(Path(args.out)) and not args.force:
        parser.error(
            f"--out {args.out} is the reference baseline of the replay tests; "
            "choose another folder (e.g. baseline_v2) or add --force to overwrite it."
        )

    pdf_paths = [str(Path(p).resolve()) for p in args.pdfs]
    for p in pdf_paths:
        if not Path(p).is_file():
            parser.error(f"File not found: {p}")

    # Provider and keys from .env / environment; models from there too, or APP_MODELS.
    env_settings = get_settings()
    models = select_models(env_settings, args.app_models)
    run_settings = env_settings.model_copy(update=models)
    # Pipeline parameters: code defaults (= app.py values) plus the chosen models.
    settings = Settings.model_construct(**models)

    source = "app.py (--app-models)" if args.app_models else "settings (.env / environment)"
    print(f"Models from {source}; LLM provider {run_settings.llm_provider}, "
          f"embedding provider {run_settings.embed_provider}:")
    for role, model in models_by_role(settings).items():
        print(f"  {role:<10} {model}")
    try:
        llm_factory.check_config(run_settings)
    except LLMConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    run = run_info(run_settings, "app" if args.app_models else "settings", datetime.now(timezone.utc), git_version())

    out_dir = Path(args.out).resolve()
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    client = RecordingClient(
        llm_factory.get_llm("lo", run_settings), llm_factory.get_embedder(run_settings), out_dir / "llm_calls.jsonl"
    )

    reports: dict = {}
    lo_report, item_report = None, None
    original_factory = install_client_guard()

    timings: dict = {}
    total_start = time.perf_counter()
    try:
        # 1) Extraction – same input shape as app.py ({"path", "name"}).
        pdf_inputs = [{"path": p, "name": os.path.basename(p)} for p in pdf_paths]
        t0 = time.perf_counter()
        segments = pdfs_to_text(pdf_inputs, ocr=True)
        timings["extraction"] = round(time.perf_counter() - t0, 4)
        timings["pages"] = len(segments)
        with open(out_dir / "segments.json", "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        # 2) LO pipeline – same arguments as app.py, plus the recording client.
        t0 = time.perf_counter()
        lo_result = generate_lo_pipeline(
            segments,
            batch_size=LO_BATCH_SIZE,
            generation_model=settings.llm_model_lo,
            prerequisites_model=settings.llm_model_lo,
            llm=client,
            embedder=client,
            verbose=True,
            settings=settings,
        )
        timings["lo_pipeline_wall"] = round(time.perf_counter() - t0, 4)
        los = lo_result.learning_objects
        lo_report = lo_result.reports["lo_processing_time"]
        reports.update(lo_result.reports)
        save_learning_objects_json(los, str(out_dir), all_los=lo_result.all_learning_objects)

        # 3) Item pipeline – app.py runs it only when some LO passed the filter.
        items = []
        if los:
            t0 = time.perf_counter()
            item_result = generate_all_items(
                los,
                segments,
                batch_size=ITEM_BATCH_SIZE,
                evaluation_batch_size=ITEM_EVALUATION_BATCH_SIZE,
                generation_model=settings.llm_model_questions,
                evaluation_model=settings.llm_model_eval,
                llm=client,
                embedder=client,
                verbose=True,
                settings=settings,
            )
            timings["item_pipeline_wall"] = round(time.perf_counter() - t0, 4)
            items = item_result.questions
            item_report = item_result.reports["item_processing_time"]
            reports.update(item_result.reports)
            save_questions_json(items, str(out_dir), all_items=item_result.all_questions)
        write_reports(reports, reports_dir)
        timings["total_wall"] = round(time.perf_counter() - total_start, 4)
    finally:
        llm_factory.get_llm = original_factory["get_llm"]
        llm_factory.get_embedder = original_factory["get_embedder"]
        client.close()

    metrics = build_metrics(
        args, pdf_paths, reports, lo_report, item_report, los, items, timings, client.recorder, settings, run
    )
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== BASELINE METRICS ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
