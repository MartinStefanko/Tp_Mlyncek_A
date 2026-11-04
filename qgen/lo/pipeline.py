import time
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from qgen.config import Settings, get_settings, setting_or_default
from qgen.context_builder import build_allowed_source_refs, build_source_name_map, parse_source_refs, resolve_source_names
from qgen.extraction.document_language import resolve_document_language
from qgen.lo.clustering import cluster_by_core
from qgen.lo.faithfulness import analyze_lo_faithfulness
from qgen.lo.generation import generate_learning_objects
from qgen.lo.relevance_to_segment import analyze_lo_relevance_to_segment
from qgen.lo.validation import validate_learning_objects
from qgen.lo.prerequisites import infer_prerequisites
from qgen.lo.topic_coverage import analyze_topic_coverage
from qgen.progress import (
    STAGE_LEARNING_OBJECTS,
    STAGE_LO_EVALUATION,
    ProgressCallback,
    SubProgress,
    batch_count,
    report_progress,
)

@dataclass
class LOPipelineResult:
    """Result of generate_lo_pipeline. Nothing is written to disk.

    learning_objects      accepted LO (tematické celky) with ids 1..n
    all_learning_objects  all LO before filtering (ids before renumbering)
    lo_id_map             original id -> new id of the accepted LO
    language_info         {"language": "sk"|"en", "reason": str}
    reports               report dicts, keys: lo_validation, document_topics,
                          topic_coverage, lo_relevance_to_segment, lo_faithfulness,
                          lo_processing_time
    """

    learning_objects: list[dict]
    all_learning_objects: list[dict]
    lo_id_map: dict[Any, int]
    language_info: dict[str, str]
    reports: dict[str, Any] = field(default_factory=dict)

    @property
    def document_language(self) -> str:
        return self.language_info.get("language", "sk")


def lo_page_sort_key(lo):
    refs = parse_source_refs(lo.get("citovane_zdroje", []))
    source_id, first_page = refs[0] if refs else ("", float("inf"))
    return (source_id or "", first_page, lo.get("id", float("inf")))


def attach_source_names(los, source_name_map):
    for lo in los:
        lo["zdroj"] = resolve_source_names(lo.get("citovane_zdroje", []), source_name_map)
    return los


def _lo_evaluation_reports(segmenty, los, client, verbose, settings, batch_size, document_language, progress_callback):
    """Topic coverage, relevance and faithfulness of the LO (step lo_evaluation)."""
    total = 2 + batch_count(len(los), batch_size)
    report_progress(progress_callback, STAGE_LO_EVALUATION, 0, total)
    coverage_report = analyze_topic_coverage(
        segmenty,
        los,
        client=client,
        topic_model=settings.llm_model,
        embedding_model=settings.embed_model,
        similarity_threshold=settings.topic_match_threshold,
        verbose=verbose,
        source_max_chars=settings.topic_source_max_chars,
    )
    report_progress(progress_callback, STAGE_LO_EVALUATION, 1, total)
    relevance_report = analyze_lo_relevance_to_segment(
        segmenty,
        los,
        client=client,
        embedding_model=settings.embed_model,
        verbose=verbose,
        context_max_chars=settings.context_max_chars,
    )
    report_progress(progress_callback, STAGE_LO_EVALUATION, 2, total)
    faithfulness_progress = SubProgress(progress_callback, before=2)
    faithfulness_report = analyze_lo_faithfulness(
        segmenty,
        los,
        client=client,
        model=settings.llm_model,
        verbose=verbose,
        batch_size=batch_size,
        max_batch_attempts=settings.check_batch_attempts,
        document_language=document_language,
        context_max_chars=settings.context_max_chars,
        progress_callback=faithfulness_progress,
    )
    done = 2 + faithfulness_progress.sub_total
    report_progress(progress_callback, STAGE_LO_EVALUATION, done, done)
    return coverage_report, relevance_report, faithfulness_report


def generate_lo_pipeline(
    segmenty,
    batch_size=None,
    model=None,
    generation_model=None,
    prerequisites_model=None,
    client=None,
    verbose=True,
    document_language=None,
    progress_callback: Optional[ProgressCallback] = None,
    settings: Optional[Settings] = None,
) -> LOPipelineResult:
    """LO pipeline: language, LO generation, clustering, prerequisites, evaluation, filter.

    document_language: None or "auto" = detect with the LLM; "sk" / "en" = use it.
    Values not given come from `settings` (default: get_settings()).
    """
    settings = settings or get_settings()
    batch_size = batch_size or settings.lo_batch_size
    generation_model = generation_model or model or settings.llm_model_lo
    prerequisites_model = prerequisites_model or model or settings.llm_model_lo
    source_name_map = build_source_name_map(segmenty)
    language_info = resolve_document_language(
        segmenty,
        document_language,
        client=client,
        model=generation_model,
        settings=settings,
        verbose=verbose,
        progress_callback=progress_callback,
    )
    document_language = language_info.get("language", "sk")

    # Step learning_objects: generation batches, then clustering and prerequisites.
    generation_batches = batch_count(len(segmenty), batch_size)
    lo_steps_total = generation_batches + 2
    generation_start = time.perf_counter()
    los = generate_learning_objects(
        segmenty,
        batch_size=batch_size,
        model=generation_model,
        client=client,
        verbose=verbose,
        document_language=document_language,
        progress_callback=SubProgress(progress_callback, after=2),
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
        report_progress(progress_callback, STAGE_LEARNING_OBJECTS, lo_steps_total, lo_steps_total)
        empty_report = validate_learning_objects([], allowed_pages=[])
        empty_coverage_report, empty_relevance_report, empty_faithfulness_report = _lo_evaluation_reports(
            segmenty, [], client, verbose, settings, batch_size, document_language, progress_callback
        )
        return LOPipelineResult(
            learning_objects=[],
            all_learning_objects=[],
            lo_id_map={},
            language_info=language_info,
            reports={
                "lo_validation": empty_report,
                "document_topics": empty_coverage_report.get("topics", []),
                "topic_coverage": empty_coverage_report,
                "lo_relevance_to_segment": empty_relevance_report,
                "lo_faithfulness": empty_faithfulness_report,
                "lo_processing_time": timing_report,
            },
        )

    generation_start = time.perf_counter()
    los = cluster_by_core(
        los,
        similarity_threshold=settings.cluster_sim_threshold,
        client=client,
        embedding_model=settings.embed_model,
        require_source_overlap=settings.cluster_require_source_overlap,
    )
    report_progress(progress_callback, STAGE_LEARNING_OBJECTS, generation_batches + 1, lo_steps_total)
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
    report_progress(progress_callback, STAGE_LEARNING_OBJECTS, lo_steps_total, lo_steps_total)
    generation_seconds += time.perf_counter() - generation_start

    allowed_pages = build_allowed_source_refs(segmenty)
    evaluation_start = time.perf_counter()
    validation_report = validate_learning_objects(los, allowed_pages=allowed_pages)
    coverage_report, relevance_report, faithfulness_report = _lo_evaluation_reports(
        segmenty, los, client, verbose, settings, batch_size, document_language, progress_callback
    )
    accepted_los = filter_learning_objects_variant_b(
        los,
        validation_report,
        faithfulness_report,
        min_faithfulness_score=settings.lo_min_faithfulness_score,
    )
    normalized_los, lo_id_map = normalize_learning_object_ids(accepted_los)
    attach_source_names(normalized_los, source_name_map)
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
    return LOPipelineResult(
        learning_objects=normalized_los,
        all_learning_objects=los,
        lo_id_map=lo_id_map,
        language_info=language_info,
        reports={
            "lo_validation": validation_report,
            "document_topics": coverage_report.get("topics", []),
            "topic_coverage": coverage_report,
            "lo_relevance_to_segment": relevance_report,
            "lo_faithfulness": faithfulness_report,
            "lo_processing_time": timing_report,
        },
    )


def filter_learning_objects_variant_b(los, validation_report, faithfulness_report, min_faithfulness_score=None):
    min_faithfulness_score = setting_or_default(min_faithfulness_score, "lo_min_faithfulness_score")
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
        if faithfulness_score is None or faithfulness_score < min_faithfulness_score:
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
