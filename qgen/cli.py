"""Command line run of the pipeline; writes results and reports to a folder.

    python -m qgen.cli <pdf...> --out <folder> [--ocr] [--language auto|sk|en]

Output (today's format; the units format comes in step 14):
    segments.json                      extracted pages
    learning_objects.json / _all.json  accepted / all LO (tematické celky)
    questions.json / _all.json         accepted / all questions (only when some LO passed)
    reports/*.txt                      the pipeline reports in the former text format
    reports/*_diagnostics.json         section_diagnostics, question_diagnostics (counts,
                                       dropped questions, section_without_questions warnings)

Progress goes to the console through progress_callback. This module and tools/ are the
only places that write pipeline results to disk.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from qgen.config import Settings, get_settings
from qgen.extraction.text_extraction import OcrUnavailableError, pdfs_to_text
from qgen.llm.errors import LLMConfigError
from qgen.llm.factory import check_config, get_embedder, get_llm
from qgen.pipeline import PipelineResult, run_pipeline


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (int, float, bool)):
        return [value]
    text = str(value).strip()
    return [text] if text else []


def normalize_lo_for_export(obj):
    return {
        "id": obj.get("id"),
        "vzdelavaci_objekt": obj.get("vzdelávací_objekt", obj.get("vzdelavaci_objekt", "")),
        "bloom_level": obj.get("bloom_level", ""),
        "odporucane_aktivity": normalize_list(
            obj.get("odporúčané_aktivity", obj.get("odporucane_aktivity", []))
        ),
        "odporucane_zadania": obj.get("odporúčané_zadania", obj.get("odporucane_zadania", "")),
        "citovane_zdroje": normalize_list(
            obj.get("citovane_zdroje", obj.get("citované_zdroje", []))
        ),
        "zdroj": normalize_list(obj.get("zdroj", [])),
        "prerekvizity": normalize_list(obj.get("prerekvizity", [])),
    }


def save_learning_objects_json(los, output_dir, all_los=None):
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, "learning_objects.json")
    export_los = [normalize_lo_for_export(obj) for obj in los]
    write_json(out_path, export_los)

    if all_los is not None:
        all_out_path = os.path.join(output_dir, "learning_objects_all.json")
        export_all_los = [normalize_lo_for_export(obj) for obj in all_los]
        write_json(all_out_path, export_all_los)

    return out_path


def save_processing_time_report(report, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)

    lines = [
        f"pipeline: {report.get('pipeline', '')}",
        f"generation_seconds: {report.get('generation_seconds', 0.0)}",
        f"evaluation_seconds: {report.get('evaluation_seconds', 0.0)}",
        f"total_seconds: {report.get('total_seconds', 0.0)}",
    ]

    details = report.get("details", {})
    if details:
        lines.append("")
        lines.append("DETAILS:")
        for key, value in details.items():
            lines.append(f"- {key}: {value}")

    report_txt_path = os.path.join(output_dir, filename)
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_txt_path


def save_lo_validation_report(validation_report, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    stats = validation_report.get("stats", {})
    errors = validation_report.get("errors", [])
    warnings = validation_report.get("warnings", [])

    lines = [
        f"is_valid_json: {validation_report.get('is_valid_json')}",
        f"is_valid: {validation_report.get('is_valid')}",
        f"total: {stats.get('total', 0)}",
        f"valid: {stats.get('valid', 0)}",
        f"invalid: {stats.get('invalid', 0)}",
        f"errors_count: {len(errors)}",
        f"warnings_count: {len(warnings)}",
        "",
        "ERRORS:",
    ]

    if errors:
        for error in errors:
            lines.append(f"- {error}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("WARNINGS:")
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")

    report_txt_path = os.path.join(output_dir, "learning_objects_validation_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_txt_path


def save_item_validation_report(validation_report, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    stats = validation_report.get("stats", {})
    errors = validation_report.get("errors", [])
    warnings = validation_report.get("warnings", [])

    lines = [
        f"is_valid_json: {validation_report.get('is_valid_json')}",
        f"is_valid: {validation_report.get('is_valid')}",
        f"total: {stats.get('total', 0)}",
        f"valid: {stats.get('valid', 0)}",
        f"invalid: {stats.get('invalid', 0)}",
        f"errors_count: {len(errors)}",
        f"warnings_count: {len(warnings)}",
        "",
        "ERRORS:",
    ]

    if errors:
        for error in errors:
            lines.append(f"- {error}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("WARNINGS:")
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")

    report_txt_path = os.path.join(output_dir, "item_validation_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_txt_path


def save_item_relevance_to_lo_report(relevance_report, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    stats = relevance_report.get("stats", {})
    items = relevance_report.get("items", [])

    lines = [
        f"items_total: {stats.get('items_total', 0)}",
        f"items_compared: {stats.get('items_compared', 0)}",
        f"average_similarity: {stats.get('average_similarity', 0.0)}",
        "",
        "ITEM_RELEVANCE_TO_LO:",
    ]

    if items:
        for item in items:
            lines.append(f"- item_id: {item.get('item_id', '-')}")
            lines.append(f"  lo_id: {item.get('lo_id', '-')}")
            lines.append(f"  lo_name: {item.get('lo_name', '')}")
            lines.append(f"  has_lo: {item.get('has_lo', False)}")
            lines.append(f"  similarity: {item.get('similarity', '-')}")
            lines.append("")
    else:
        lines.append("- none")

    report_txt_path = os.path.join(output_dir, "item_relevance_to_lo_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_txt_path


def save_item_faithfulness_report(faithfulness_report, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    stats = faithfulness_report.get("stats", {})
    items = faithfulness_report.get("items", [])

    lines = [
        f"items_total: {stats.get('items_total', 0)}",
        f"items_evaluated: {stats.get('items_evaluated', 0)}",
        f"average_faithfulness_score: {stats.get('average_faithfulness_score', 0.0)}",
        f"faithful_items: {stats.get('faithful_items', 0)}",
        f"faithful_items_percent: {stats.get('faithful_items_percent', 0.0)}",
        "",
        "ITEM_FAITHFULNESS:",
    ]

    if items:
        for item in items:
            lines.append(f"- item_id: {item.get('item_id', '-')}")
            lines.append(f"  lo_id: {item.get('lo_id', '-')}")
            lines.append(f"  source_pages: {item.get('source_pages', [])}")
            lines.append(f"  faithfulness_score: {item.get('faithfulness_score', '-')}")
            lines.append(f"  faithful: {item.get('faithful', False)}")
            lines.append(f"  reason: {item.get('reason', '')}")
            lines.append("")
    else:
        lines.append("- none")

    report_txt_path = os.path.join(output_dir, "item_faithfulness_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_txt_path


def save_item_answerability_report(answerability_report, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    stats = answerability_report.get("stats", {})
    items = answerability_report.get("items", [])

    lines = [
        f"items_total: {stats.get('items_total', 0)}",
        f"items_evaluated: {stats.get('items_evaluated', 0)}",
        f"average_answerability_score: {stats.get('average_answerability_score', 0.0)}",
        f"answerable_items: {stats.get('answerable_items', 0)}",
        f"answerable_items_percent: {stats.get('answerable_items_percent', 0.0)}",
        "",
        "ITEM_ANSWERABILITY:",
    ]

    if items:
        for item in items:
            lines.append(f"- item_id: {item.get('item_id', '-')}")
            lines.append(f"  lo_id: {item.get('lo_id', '-')}")
            lines.append(f"  source_pages: {item.get('source_pages', [])}")
            lines.append(f"  answerability_score: {item.get('answerability_score', '-')}")
            lines.append(f"  answerable: {item.get('answerable', False)}")
            lines.append(f"  reason: {item.get('reason', '')}")
            lines.append("")
    else:
        lines.append("- none")

    report_txt_path = os.path.join(output_dir, "item_answerability_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_txt_path


def save_topic_coverage_report(coverage_report, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    stats = coverage_report.get("stats", {})
    topics = coverage_report.get("topics", [])

    lines = [
        f"topics_total: {stats.get('topics_total', 0)}",
        f"topics_covered: {stats.get('topics_covered', 0)}",
        f"coverage_percent: {stats.get('coverage_percent', 0.0)}",
        f"similarity_threshold: {stats.get('similarity_threshold', 0.0)}",
        "",
        "TOPICS:",
    ]

    if topics:
        for topic in topics:
            lines.append(f"- tema: {topic.get('tema', '')}")
            lines.append(f"  covered: {topic.get('covered', False)}")
            lines.append(f"  similarity: {topic.get('similarity', 0.0)}")
            lines.append(f"  best_lo_id: {topic.get('best_lo_id', '-')}")
            lines.append(f"  best_lo_name: {topic.get('best_lo_name', '')}")
            lines.append("")
    else:
        lines.append("- none")

    report_txt_path = os.path.join(output_dir, "topic_coverage_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_txt_path


def save_document_topics_txt(topics, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    if topics:
        for idx, topic in enumerate(topics, start=1):
            lines.append(f"{idx}. {topic.get('tema', '')}")
    else:
        lines.append("Ziadne temy neboli identifikovane.")

    topics_txt_path = os.path.join(output_dir, "document_topics.txt")
    with open(topics_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return topics_txt_path


def save_lo_relevance_to_segment_report(relevance_report, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    stats = relevance_report.get("stats", {})
    items = relevance_report.get("items", [])

    lines = [
        f"los_total: {stats.get('los_total', 0)}",
        f"los_compared: {stats.get('los_compared', 0)}",
        f"average_similarity: {stats.get('average_similarity', 0.0)}",
        "",
        "LO_RELEVANCE_TO_SEGMENT:",
    ]

    if items:
        for item in items:
            lines.append(f"- lo_id: {item.get('lo_id', '-')}")
            lines.append(f"  lo_name: {item.get('lo_name', '')}")
            lines.append(f"  has_source_text: {item.get('has_source_text', False)}")
            lines.append(f"  source_pages: {item.get('source_pages', [])}")
            lines.append(f"  similarity: {item.get('similarity', '-')}")
            lines.append("")
    else:
        lines.append("- none")

    report_txt_path = os.path.join(output_dir, "lo_relevance_to_segment_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_txt_path


def save_lo_faithfulness_report(faithfulness_report, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    stats = faithfulness_report.get("stats", {})
    items = faithfulness_report.get("items", [])

    lines = [
        f"los_total: {stats.get('los_total', 0)}",
        f"los_evaluated: {stats.get('los_evaluated', 0)}",
        f"average_faithfulness_score: {stats.get('average_faithfulness_score', 0.0)}",
        "",
        "LO_FAITHFULNESS:",
    ]

    if items:
        for item in items:
            lines.append(f"- lo_id: {item.get('lo_id', '-')}")
            lines.append(f"  lo_name: {item.get('lo_name', '')}")
            lines.append(f"  source_pages: {item.get('source_pages', [])}")
            lines.append(f"  faithfulness_score: {item.get('faithfulness_score', '-')}")
            lines.append(f"  reason: {item.get('reason', '')}")
            lines.append("")
    else:
        lines.append("- none")

    report_txt_path = os.path.join(output_dir, "lo_faithfulness_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_txt_path


def save_questions_json(items, output_dir, all_items=None):
    os.makedirs(output_dir, exist_ok=True)

    items_json_path = os.path.join(output_dir, "questions.json")
    write_json(items_json_path, items)

    if all_items is not None:
        all_items_json_path = os.path.join(output_dir, "questions_all.json")
        write_json(all_items_json_path, all_items)

    return items_json_path


# Report key (PipelineResult.reports) -> writer(report, reports_dir).
def save_json_report(report, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    write_json(path, report)
    return path


REPORT_WRITERS = {
    "lo_validation": save_lo_validation_report,
    "document_topics": save_document_topics_txt,
    "topic_coverage": save_topic_coverage_report,
    "lo_relevance_to_segment": save_lo_relevance_to_segment_report,
    "lo_faithfulness": save_lo_faithfulness_report,
    "lo_processing_time": lambda report, d: save_processing_time_report(report, d, "lo_processing_time_report.txt"),
    "item_validation": save_item_validation_report,
    "item_relevance_to_lo": save_item_relevance_to_lo_report,
    "item_faithfulness": save_item_faithfulness_report,
    "item_answerability": save_item_answerability_report,
    "item_processing_time": lambda report, d: save_processing_time_report(report, d, "item_processing_time_report.txt"),
    "section_diagnostics": lambda report, d: save_json_report(report, d, "section_diagnostics.json"),
    "question_diagnostics": lambda report, d: save_json_report(report, d, "question_diagnostics.json"),
}


def write_reports(reports, reports_dir):
    """Write every known report of `reports` as a text file; returns the written paths."""
    return [
        REPORT_WRITERS[key](report, str(reports_dir))
        for key, report in reports.items()
        if key in REPORT_WRITERS
    ]


def write_results(result: PipelineResult, out_dir, segments=None):
    """Write the pipeline result in today's format (see the module docstring)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if segments is not None:
        write_json(out_dir / "segments.json", segments)
    save_learning_objects_json(result.learning_objects, str(out_dir), all_los=result.all_learning_objects)
    if result.question_result is not None:
        save_questions_json(result.questions, str(out_dir), all_items=result.all_questions)
    write_reports(result.reports, out_dir / "reports")


def print_progress(stage, done, total):
    percent = f"{100 * done / total:5.1f} %" if total else "  -   "
    print(f"[{stage:<16}] {done:>4}/{total:<4} {percent}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m qgen.cli",
        description="Extract text, generate LO (tematické celky) and questions, write results to a folder.",
    )
    parser.add_argument("pdfs", nargs="+", help="Input PDF file(s).")
    parser.add_argument("--out", required=True, help="Output folder.")
    parser.add_argument(
        "--ocr",
        action="store_true",
        default=None,
        help="OCR of images in the PDF (needs easyocr). Default: QGEN_OCR_ENABLED.",
    )
    parser.add_argument(
        "--language",
        choices=("auto", "sk", "en"),
        default="auto",
        help="Document language; auto = detect (default).",
    )
    parser.add_argument(
        "--per-section",
        type=int,
        choices=range(1, 11),
        metavar="N",
        default=None,
        help="Minimum questions per section (tematický celok), 1–10; more when the content allows. Default: 3.",
    )
    parser.add_argument("--verbose", action="store_true", help="Also print the pipeline's own log lines.")
    return parser


def main(argv=None, llm=None, embedder=None, settings: Settings = None) -> int:
    """`llm`, `embedder` and `settings` are for tests (replay client, fixed settings)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = settings or get_settings()

    pdf_paths = [str(Path(p).resolve()) for p in args.pdfs]
    for p in pdf_paths:
        if not Path(p).is_file():
            parser.error(f"File not found: {p}")

    # Checked before the extraction, so a missing key or base_url fails at once and not
    # after OCR. One LLM client serves every role (QGEN_LLM_PROVIDER); the embedder has its
    # own provider (QGEN_EMBED_PROVIDER).
    try:
        check_config(settings, llm=llm is None, embedder=embedder is None)
    except LLMConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    if llm is None:
        llm = get_llm("lo", settings)
    if embedder is None:
        embedder = get_embedder(settings)

    pdf_inputs = [{"path": p, "name": os.path.basename(p)} for p in pdf_paths]
    try:
        segments = pdfs_to_text(pdf_inputs, ocr=args.ocr, progress_callback=print_progress, settings=settings)
    except OcrUnavailableError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    result = run_pipeline(
        segments,
        llm=llm,
        embedder=embedder,
        language=args.language,
        progress_callback=print_progress,
        settings=settings,
        verbose=args.verbose,
        per_section=args.per_section,
    )
    write_results(result, args.out, segments=segments)

    print(
        f"Done: {len(segments)} pages, {len(result.learning_objects)} LO, "
        f"{len(result.questions)} questions. Saved to: {Path(args.out).resolve()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
