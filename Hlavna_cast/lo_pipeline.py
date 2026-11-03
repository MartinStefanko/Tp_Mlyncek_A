import time
import re

from context_builder import build_allowed_source_refs, build_source_name_map, parse_source_refs, resolve_source_names
from document_language import detect_document_language
from lo_clustering import cluster_by_core
from lo_faithfulness import analyze_lo_faithfulness
from lo_generation import generate_learning_objects
from lo_relevance_to_segment import analyze_lo_relevance_to_segment
from outputs import (
    save_document_topics_txt,
    save_lo_faithfulness_report,
    save_lo_relevance_to_segment_report,
    save_processing_time_report,
    save_lo_validation_report,
    save_topic_coverage_report,
)
from lo_validation import validate_learning_objects
from prerequisites import infer_prerequisites
from topic_coverage import analyze_topic_coverage

LO_MIN_FAITHFULNESS_SCORE = 3


def lo_page_sort_key(lo):
    refs = parse_source_refs(lo.get("citovane_zdroje", []))
    source_id, first_page = refs[0] if refs else ("", float("inf"))
    return (source_id or "", first_page, lo.get("id", float("inf")))


def attach_source_names(los, source_name_map):
    for lo in los:
        lo["zdroj"] = resolve_source_names(lo.get("citovane_zdroje", []), source_name_map)
    return los


def generate_lo_pipeline(
    segmenty,
    batch_size=10,
    model="gemini-2.5-flash-lite",
    generation_model=None,
    prerequisites_model=None,
    output_dir=None,
    client=None,
    verbose=True,
    return_metrics=False,
):
    generation_model = generation_model or model
    prerequisites_model = prerequisites_model or model
    source_name_map = build_source_name_map(segmenty)
    language_info = detect_document_language(
        segmenty,
        client=client,
        model=generation_model,
        verbose=verbose,
    )
    document_language = language_info.get("language", "sk")

    generation_start = time.perf_counter()
    los = generate_learning_objects(
        segmenty,
        batch_size=batch_size,
        model=generation_model,
        client=client,
        verbose=verbose,
        document_language=document_language,
    )
    attach_source_names(los, source_name_map)
    generation_seconds = time.perf_counter() - generation_start
    evaluation_seconds = 0.0
    timing_report = {
        "pipeline": "learning_objects",
        "generation_seconds": round(generation_seconds, 4),
        "evaluation_seconds": round(evaluation_seconds, 4),
        "total_seconds": round(generation_seconds + evaluation_seconds, 4),
        "details": {
            "los_count": len(los),
        },
    }
    if not los:
        empty_report = validate_learning_objects([], allowed_pages=[])
        if output_dir:
            save_lo_validation_report(empty_report, output_dir)
            empty_coverage_report = analyze_topic_coverage(
                segmenty,
                [],
                client=client,
                verbose=verbose,
            )
            save_document_topics_txt(empty_coverage_report.get("topics", []), output_dir)
            save_topic_coverage_report(empty_coverage_report, output_dir)
            empty_relevance_report = analyze_lo_relevance_to_segment(
                segmenty,
                [],
                client=client,
                verbose=verbose,
            )
            save_lo_relevance_to_segment_report(empty_relevance_report, output_dir)
            empty_faithfulness_report = analyze_lo_faithfulness(
                segmenty,
                [],
                client=client,
                verbose=verbose,
            )
            save_lo_faithfulness_report(empty_faithfulness_report, output_dir)
            save_processing_time_report(timing_report, output_dir, "lo_processing_time_report.txt")
        if return_metrics:
            timing_report["all_los"] = []
            return [], timing_report
        return []

    generation_start = time.perf_counter()
    los = cluster_by_core(los, client=client)
    los.sort(key=lo_page_sort_key)
    for i, obj in enumerate(los, start=1):
        obj["id"] = i

    los = infer_prerequisites(
        los,
        model=prerequisites_model,
        client=client,
        verbose=verbose,
        document_language=document_language,
    )
    attach_source_names(los, source_name_map)
    generation_seconds += time.perf_counter() - generation_start

    allowed_pages = build_allowed_source_refs(segmenty)
    evaluation_start = time.perf_counter()
    validation_report = validate_learning_objects(los, allowed_pages=allowed_pages)
    coverage_report = analyze_topic_coverage(
        segmenty,
        los,
        client=client,
        verbose=verbose,
    )
    relevance_report = analyze_lo_relevance_to_segment(
        segmenty,
        los,
        client=client,
        verbose=verbose,
    )
    faithfulness_report = analyze_lo_faithfulness(
        segmenty,
        los,
        client=client,
        verbose=verbose,
        batch_size=batch_size,
        document_language=document_language,
    )
    accepted_los = filter_learning_objects_variant_b(los, validation_report, faithfulness_report)
    normalized_los, lo_id_map = normalize_learning_object_ids(accepted_los)
    attach_source_names(normalized_los, source_name_map)
    if output_dir:
        save_lo_validation_report(validation_report, output_dir)
        save_document_topics_txt(coverage_report.get("topics", []), output_dir)
        save_topic_coverage_report(coverage_report, output_dir)
        save_lo_relevance_to_segment_report(relevance_report, output_dir)
        save_lo_faithfulness_report(faithfulness_report, output_dir)
    evaluation_seconds = time.perf_counter() - evaluation_start
    timing_report = {
        "pipeline": "learning_objects",
        "generation_seconds": round(generation_seconds, 4),
        "evaluation_seconds": round(evaluation_seconds, 4),
        "total_seconds": round(generation_seconds + evaluation_seconds, 4),
        "details": {
            "los_count_all": len(los),
            "los_count_accepted": len(normalized_los),
            "document_language": document_language,
            "document_language_reason": language_info.get("reason", ""),
        },
    }
    if output_dir:
        save_processing_time_report(timing_report, output_dir, "lo_processing_time_report.txt")

    if verbose:
        if validation_report["is_valid"]:
            print(f"Formalna validacia LO uspesna. Overenych LO: {validation_report['stats']['valid']}")
        else:
            print(
                "Formalna validacia LO zlyhala. "
                f"Pocet chyb: {len(validation_report['errors'])}"
            )
            for error in validation_report["errors"][:10]:
                print(f"  - {error}")
        print(f"Cas generovania LO: {generation_seconds:.2f} s")
        print(f"Cas evaluacie LO: {evaluation_seconds:.2f} s")
    if return_metrics:
        timing_report["all_los"] = los
        timing_report["accepted_lo_id_map"] = lo_id_map
        timing_report["language_info"] = language_info
        return normalized_los, timing_report
    return normalized_los


def filter_learning_objects_variant_b(los, validation_report, faithfulness_report):
    invalid_lo_ids = extract_prefixed_ids(validation_report.get("errors", []), "LO")
    faithfulness_by_id = {
        row.get("lo_id"): row.get("faithfulness_score")
        for row in faithfulness_report.get("items", [])
        if row.get("lo_id") is not None
    }

    accepted = []
    for lo in los:
        lo_id = lo.get("id")
        if lo_id in invalid_lo_ids:
            continue
        if not parse_source_refs(lo.get("citovane_zdroje", [])):
            continue
        faithfulness_score = faithfulness_by_id.get(lo_id)
        if faithfulness_score is None or faithfulness_score < LO_MIN_FAITHFULNESS_SCORE:
            continue
        accepted.append(lo)
    return accepted


def normalize_learning_object_ids(los):
    normalized = []
    lo_id_map = {}

    for new_id, lo in enumerate(los, start=1):
        original_id = lo.get("id")
        cloned = dict(lo)
        cloned["id"] = new_id
        original_prereq = lo.get("prerekvizity", [])
        cloned["prerekvizity"] = list(original_prereq) if isinstance(original_prereq, list) else []
        normalized.append(cloned)
        lo_id_map[original_id] = new_id

    for lo in normalized:
        lo["prerekvizity"] = [
            lo_id_map[pr]
            for pr in lo.get("prerekvizity", [])
            if pr in lo_id_map
        ]

    return normalized, lo_id_map


def extract_prefixed_ids(errors, prefix):
    ids = set()
    pattern = re.compile(rf"{re.escape(prefix)}\s+(\d+)")
    for error in errors:
        match = pattern.search(str(error))
        if match:
            ids.add(int(match.group(1)))
    return ids
