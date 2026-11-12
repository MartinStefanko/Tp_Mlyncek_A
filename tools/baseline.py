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

Split into lessons (step 16): --split JSON (options.split) also runs the output step
(qgen.output.build_output: lessons, their titles by the LLM, units) and writes units.json and
report.json. With --base DIR the pipeline is not called at all: it is replayed from the
recording DIR (its segments.json, models and question settings from its metrics.json), and
only the calls of the split and output steps go to the API and into llm_calls.jsonl. Such a
short recording works on top of DIR (replay tests join both recordings).

Protected recordings: baseline_v6/ (reference of the replay tests: coverage rounds), the
split recordings on top of it (baseline_v6_single/, _topics/, _lessons3/, _syllabus_text/),
and the historical records baseline_v5/ with baseline_v5_* (one coverage call, commit
dff7c00), baseline_v4/ (step 13, prompts with hints), baseline_v3/ (old question format) and
baseline/ (gemini-2.5). --out pointing at one of them is refused unless
--force is given.

Usage (from the repo root):
    venv/Scripts/python.exe tools/baseline.py tests/fixtures/eko.pdf [more.pdf ...] --out baseline_v2
        [--app-models] [--force] [--split '{"mode": "topics"}']
    venv/Scripts/python.exe tools/baseline.py tests/fixtures/eko.pdf --base baseline_v6
        --split '{"mode": "topics"}' --out baseline_v6_topics --force

Outputs in --out:
    llm_calls.jsonl                    recording of all LLM and embedding calls
    segments.json                      extracted pages (pipeline input after extraction)
    learning_objects.json / _all.json  final / all LO (same JSON export as the former app.py)
    questions.json / _all.json         final / all items (same JSON export as the former app.py)
    reports/                           the pipeline reports as text files
    metrics.json                       summary metrics taken from those reports, plus the
                                       models of every role, run date and git commit ("run")
    units.json / report.json           with --split: result and report of the output step
With --base only llm_calls.jsonl (split and output calls), units.json, report.json and
metrics.json ("run", "base", "split", "llm_calls").
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
from qgen.cli import parse_split  # noqa: E402
from qgen.llm.adapters.replay import ReplayClient  # noqa: E402
from qgen.lo.pipeline import generate_lo_pipeline  # noqa: E402
from qgen.output import build_output  # noqa: E402
from qgen.pipeline import PipelineResult  # noqa: E402
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
ITEM_BATCH_SIZE = 3  # sections per question generation call (QGEN_QUESTION_GENERATION_BATCH_SIZE, step 13)
# Recordings that must not be overwritten by accident: the reference of the replay tests
# (baseline_v6/) and the historical records (baseline_v5/, baseline_v4/, baseline_v3/,
# gemini-2.5 baseline/). Short recordings of the split step on top of baseline_v6 (--base),
# references of the split replay tests (tests/output/test_split_replay.py); the ones on top
# of baseline_v5 are historical.
SPLIT_RECORDINGS = ("baseline_v6_single", "baseline_v6_topics", "baseline_v6_lessons3", "baseline_v6_syllabus_text")
HISTORICAL_SPLIT_RECORDINGS = (
    "baseline_v5_single", "baseline_v5_topics", "baseline_v5_lessons3", "baseline_v5_syllabus_text",
)
PROTECTED_BASELINE_DIRS = (
    REPO_ROOT / "baseline_v6",
    *(REPO_ROOT / name for name in SPLIT_RECORDINGS),
    REPO_ROOT / "baseline_v5",
    *(REPO_ROOT / name for name in HISTORICAL_SPLIT_RECORDINGS),
    REPO_ROOT / "baseline_v4",
    REPO_ROOT / "baseline_v3",
    REPO_ROOT / "baseline",
)

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
    out = Path(out_dir).resolve()
    return any(out == protected.resolve() for protected in PROTECTED_BASELINE_DIRS)


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
    """Which run the metrics belong to: date, code version, providers, parallel LLM calls
    (QGEN_LLM_MAX_CONCURRENCY, step 13b; the timings depend on it) and models per role."""
    return {
        "started_at": started_at.isoformat(timespec="seconds"),
        "git_commit": version.get("commit"),
        "git_dirty": version.get("dirty"),
        "models_source": models_source,
        "llm_provider": run_settings.llm_provider,
        "embed_provider": run_settings.embed_provider,
        "llm_max_concurrency": run_settings.llm_max_concurrency,
        "models_by_role": models_by_role(run_settings),
    }


def stats_of(reports: dict, name: str) -> dict:
    return (reports.get(name) or {}).get("stats", {}) or {}


def question_summary(diagnostics: dict, pipeline_settings: Settings) -> dict:
    """Summary of the new question pipeline (step 13) from question_diagnostics."""
    by_reason = (diagnostics.get("items_dropped") or {}).get("by_reason", {})
    rounds = diagnostics.get("extra_rounds") or {}
    coverage = diagnostics.get("supplementary_call") or {}
    return {
        "settings": {
            "question_generation_batch_size": pipeline_settings.question_generation_batch_size,
            "question_extra_rounds": pipeline_settings.question_extra_rounds,
            "question_coverage_rounds": pipeline_settings.question_coverage_rounds,
            "question_dedup_threshold": pipeline_settings.question_dedup_threshold,
            "llm_temperature": pipeline_settings.llm_temperature,
            "llm_max_output_tokens": pipeline_settings.llm_max_output_tokens,
        },
        "types_generated": (diagnostics.get("question_types") or {}).get("generated"),
        "types_accepted": (diagnostics.get("question_types") or {}).get("accepted"),
        "practical_accepted": (diagnostics.get("question_types") or {}).get("practical_accepted"),
        "extra_rounds_run": rounds.get("rounds_run"),
        "extra_rounds": rounds.get("rounds"),
        "generation_calls": diagnostics.get("generation_calls"),
        "coverage_rounds": coverage.get("rounds"),
        "sections_still_below": coverage.get("sections_still_below"),
        "batch_splits": diagnostics.get("batch_splits"),
        "duplicates_removed": (diagnostics.get("duplicates_removed") or {}).get("total"),
        "wrong_choice_marking": by_reason.get("wrong_choice_marking", 0),
    }


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
        # Same keys as baseline_v6/metrics.json (read by the replay tests).
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
        # Step 13: questions by type, extra rounds, batch splits, duplicates, wrong choice marking.
        "questions": question_summary(reports.get("question_diagnostics") or {}, pipeline_settings),
        # Section counts, questions per section, dropped questions, coverage warnings.
        "section_diagnostics": reports.get("section_diagnostics"),
        "question_diagnostics": {
            key: value
            for key, value in (reports.get("question_diagnostics") or {}).items()
            if key != "sections_without_content"
        } or None,
    }


# --- split into lessons (step 16) ------------------------------------------------------------


def base_settings(base_metrics: dict) -> dict:
    """Models, question settings and concurrency of a recording (its metrics.json), by Settings
    field, so the replayed pipeline runs exactly as recorded."""
    roles = base_metrics["run"]["models_by_role"]
    # Recordings before the coverage rounds (baseline_v5 and older) made one coverage call.
    recorded = {"question_coverage_rounds": 1, **base_metrics["questions"]["settings"]}
    return {
        "llm_model": roles["check"],
        "llm_model_lo": roles["lo"],
        "llm_model_questions": roles["questions"],
        "llm_model_eval": roles["eval"],
        "embed_model": roles["embed"],
        **recorded,
        "llm_max_concurrency": base_metrics["run"]["llm_max_concurrency"],
    }


def run_pipeline_steps(segments, client, settings: Settings) -> PipelineResult:
    """The LO and question pipeline as in the full run (same batch sizes)."""
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
    item_result = None
    if lo_result.learning_objects:
        item_result = generate_all_items(
            lo_result.learning_objects,
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
    return PipelineResult(
        learning_objects=lo_result.learning_objects,
        questions=item_result.questions if item_result else [],
        reports={**lo_result.reports, **(item_result.reports if item_result else {})},
        lo_result=lo_result,
        question_result=item_result,
    )


def run_output(pipeline_result, split, *, input_names, settings: Settings, client, out_dir: Path) -> dict:
    """build_output with the recording client; writes units.json and report.json, returns the
    "split" part of metrics.json."""
    started = time.perf_counter()
    output = build_output(
        pipeline_result,
        input_names=input_names,
        timings={},
        settings=settings,
        split=split,
        llm=client,
        embedder=client,
    )
    seconds = time.perf_counter() - started
    for name, model in (("units.json", output.result), ("report.json", output.report)):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            json.dump(model.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
    units = output.result.units
    meta = output.report.question_meta
    return {
        "options": split.model_dump(mode="json"),
        "lessons": len(units),
        "questions_per_lesson": [len(unit.questions) for unit in units],
        "lesson_minutes": [
            round(sum(q.time_to_handle + q.time_to_explain for q in unit.questions) / 60, 1) for unit in units
        ],
        "sections_per_lesson": [
            sorted({m.section_id for m in meta if m.unit == index}, key=lambda x: int(x[2:]))
            for index in range(len(units))
        ],
        "warnings": [w.code for w in output.report.warnings],
        "dropped": output.dropped,
        "output_seconds": round(seconds, 4),
    }


def main_on_base(args, split, parser) -> int:
    """--base: pipeline replayed from the recording, only split + output call the API."""
    base_dir = Path(args.base).resolve()
    for name in ("llm_calls.jsonl", "segments.json", "metrics.json"):
        if not (base_dir / name).is_file():
            parser.error(f"--base {args.base}: {name} is missing")
    base_metrics = json.loads((base_dir / "metrics.json").read_text(encoding="utf-8"))
    recorded = base_settings(base_metrics)
    settings = Settings.model_construct(**recorded)
    run_settings = get_settings().model_copy(update=recorded)
    print(f"Pipeline replayed from {base_dir.name}; split and output calls to {run_settings.llm_provider} "
          f"/ {run_settings.embed_provider}:")
    for role, model in models_by_role(settings).items():
        print(f"  {role:<10} {model}")
    try:
        llm_factory.check_config(run_settings)
    except LLMConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    run = run_info(run_settings, "base", datetime.now(timezone.utc), git_version())

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    segments = json.loads((base_dir / "segments.json").read_text(encoding="utf-8"))
    replay = ReplayClient(base_dir / "llm_calls.jsonl")
    client = RecordingClient(
        llm_factory.get_llm("lo", run_settings), llm_factory.get_embedder(run_settings), out_dir / "llm_calls.jsonl"
    )
    original_factory = install_client_guard()
    try:
        pipeline_result = run_pipeline_steps(segments, replay, settings)
        unused = replay.unused_records()
        if unused:
            print(f"Warning: {len(unused)} recorded calls of {base_dir.name} were not used; "
                  "the pipeline differs from the recording.", file=sys.stderr)
        split_metrics = run_output(
            pipeline_result, split, input_names=[i["name"] for i in base_metrics["inputs"]],
            settings=settings, client=client, out_dir=out_dir,
        )
    finally:
        llm_factory.get_llm = original_factory["get_llm"]
        llm_factory.get_embedder = original_factory["get_embedder"]
        client.close()

    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": run,
        "base": {
            "dir": base_dir.name,
            "git_commit": base_metrics["run"].get("git_commit"),
            "inputs": base_metrics["inputs"],
            "unused_base_calls": len(unused),
        },
        "split": split_metrics,
        "llm_calls": {"by_method": dict(client.recorder.counts), "failed": client.recorder.errors},
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("\n=== SPLIT METRICS ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {out_dir}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdfs", nargs="+", help="Input PDF file(s), e.g. tests/fixtures/eko.pdf")
    parser.add_argument("--out", required=True,
                        help="Output directory, e.g. baseline_v7 (protected recordings baseline/, baseline_v3/ to "
                             "baseline_v6/ and their split recordings only with --force)")
    parser.add_argument("--app-models", action="store_true",
                        help="Use the fixed models of the former app.py (Gemini 2.5) instead of the settings.")
    parser.add_argument("--force", action="store_true",
                        help="Allow --out to be a protected recording (baseline/, baseline_v3/ to baseline_v6/, "
                             "baseline_v5_* and baseline_v6_* split recordings).")
    parser.add_argument("--split", type=parse_split, default=None, metavar="JSON",
                        help="options.split as JSON; also runs the output step (units.json, report.json).")
    parser.add_argument("--base", default=None, metavar="DIR",
                        help="Replay the pipeline from this recording; record only the split and output calls "
                             "(needs --split).")
    args = parser.parse_args(argv)
    if args.base is not None and args.split is None:
        parser.error("--base needs --split")

    if is_reference_baseline(Path(args.out)) and not args.force:
        parser.error(
            f"--out {args.out} is a protected recording (baseline_v6/ = reference of the replay tests, "
            "baseline_v6_* = split recordings, baseline_v5/, baseline_v5_*, baseline_v4/, baseline_v3/ and "
            "baseline/ = historical records); choose another folder (e.g. baseline_v7) or add --force."
        )
    if args.base is not None:
        return main_on_base(args, args.split, parser)

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
    split_metrics = None
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
        if args.split is not None:
            pipeline_result = PipelineResult(
                learning_objects=los,
                questions=items,
                reports=reports,
                lo_result=lo_result,
                question_result=item_result if los else None,
            )
            split_metrics = run_output(
                pipeline_result, args.split, input_names=[i["name"] for i in pdf_inputs],
                settings=settings, client=client, out_dir=out_dir,
            )
        timings["total_wall"] = round(time.perf_counter() - total_start, 4)
    finally:
        llm_factory.get_llm = original_factory["get_llm"]
        llm_factory.get_embedder = original_factory["get_embedder"]
        client.close()

    metrics = build_metrics(
        args, pdf_paths, reports, lo_report, item_report, los, items, timings, client.recorder, settings, run
    )
    if split_metrics is not None:
        metrics["split"] = split_metrics
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== BASELINE METRICS ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
