"""Saving pipeline reports to output_dir (temporary, replaced by return values in step 5)."""

import json
import os


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
