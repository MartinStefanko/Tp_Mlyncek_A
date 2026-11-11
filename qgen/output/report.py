"""Report (GET /v1/jobs/{id}/report): internal quality diagnostics, not stored by ECEH.

Moved from the temporary service/jobs/legacy_output.py (step 14) without a change of the
shape (qgen.contract.report):
    sections        accepted sections (tematické celky), id "S-{lo_id}"
    question_meta   one entry per question in units (Bloom level, keywords, cited pages, topic)
    quality         pipeline scores 1–5 converted (x − 1) / 4, topic coverage / 100
    warnings        section / question diagnostics of the pipeline, then the output warnings
    timings         seconds per step
"""

from __future__ import annotations

from typing import Any, Optional

from qgen.context_builder import parse_source_refs
from qgen.contract.report import Quality, QuestionMeta, Report, ReportWarning, Section, SourceRef
from qgen.questions.types import BLOOM_FALLBACK, BLOOM_LEVELS

LEGACY_LO_TYPE = "legacy"

__all__ = [
    "LEGACY_LO_TYPE",
    "build_report",
    "diagnostic_warnings",
    "question_bloom",
    "question_meta",
    "section_id",
    "topics_by_section",
]


def section_id(lo_id: Any) -> str:
    return f"S-{lo_id}"


# --- question_meta ----------------------------------------------------------------------------


def question_meta(
    item: dict,
    section: Optional[dict],
    *,
    unit: int,
    question: int,
    position: int,
    file_names: dict[str, str],
    topics: dict[Any, str],
    warnings: list[ReportWarning],
) -> QuestionMeta:
    return QuestionMeta(
        unit=unit,
        question=question,
        section_id=section_id(item.get("lo_id")),
        bloom_level=question_bloom(item, section, warnings, position),
        keywords=[str(k) for k in item.get("keywords") or [] if str(k).strip()],
        source=_sources(item.get("source", []), file_names),
        topic=topics.get(item.get("lo_id")),
    )


def question_bloom(item: dict, section: Optional[dict], warnings: list[ReportWarning], position: int) -> str:
    """Bloom level of the question (already a code from the pipeline); else of its section."""
    if item.get("bloom_level") in BLOOM_LEVELS.values():
        return item["bloom_level"]
    value = str((section or {}).get("bloom_level", "") or "").strip()
    bloom = BLOOM_LEVELS.get(value.casefold())
    if bloom is None:
        warnings.append(
            ReportWarning(
                code="unknown_bloom_level",
                message=f"Question {position}: unknown Bloom level {value!r}, '{BLOOM_FALLBACK}' used.",
            )
        )
        return BLOOM_FALLBACK
    return bloom


def _sources(cited: Any, file_names: dict[str, str]) -> list[SourceRef]:
    pages: dict[str, list[int]] = {}
    first_file = next(iter(file_names.values()), None)
    for source_id, page in parse_source_refs(cited):
        name = file_names.get(source_id) if source_id else first_file
        if name is None or page < 1:
            continue
        file_pages = pages.setdefault(name, [])
        if page not in file_pages:
            file_pages.append(page)
    return [SourceRef(file=name, pages=sorted(found)) for name, found in pages.items()]


# --- report -----------------------------------------------------------------------------------


def build_report(
    sections: list[dict],
    meta: list[QuestionMeta],
    *,
    reports: dict[str, Any],
    topics: dict[Any, str],
    warnings: list[ReportWarning],
    timings: dict[str, float],
) -> Report:
    """`sections`: normalized sections (qgen.cli.normalize_lo_for_export); `warnings`: the
    output warnings, placed after the diagnostics of the pipeline."""
    return Report(
        sections=[_section(lo, topics.get(lo["id"])) for lo in sections],
        question_meta=meta,
        quality=_quality(reports),
        warnings=[*diagnostic_warnings(reports), *warnings],
        timings={**timings, **_pipeline_timings(reports)},
    )


def _section(lo: dict, topic: Optional[str]) -> Section:
    title = lo["vzdelavaci_objekt"] or section_id(lo["id"])
    return Section(
        id=section_id(lo["id"]),
        title=title,
        statement=str(lo.get("odporucane_zadania") or title),
        lo_type=LEGACY_LO_TYPE,
        prerequisites=[section_id(p) for p in lo["prerekvizity"] if isinstance(p, int)],
        topic=topic,
    )


def topics_by_section(reports: dict[str, Any]) -> dict[Any, str]:
    """Section id → first covered document topic whose best section it is."""
    found: dict[Any, str] = {}
    for topic in (reports.get("topic_coverage") or {}).get("topics", []):
        lo_id = topic.get("best_lo_id")
        if topic.get("covered") and lo_id is not None and topic.get("tema"):
            found.setdefault(lo_id, str(topic["tema"]))
    return found


def _score01(report: Optional[dict], average: str, count: str) -> Optional[float]:
    stats = (report or {}).get("stats", {})
    if not stats.get(count):
        return None
    return min(1.0, max(0.0, (float(stats.get(average, 0.0)) - 1) / 4))


def _quality(reports: dict[str, Any]) -> Quality:
    coverage = (reports.get("topic_coverage") or {}).get("stats", {})
    return Quality(
        section_faithfulness_avg=_score01(reports.get("lo_faithfulness"), "average_faithfulness_score", "los_evaluated"),
        question_faithfulness_avg=_score01(
            reports.get("item_faithfulness"), "average_faithfulness_score", "items_evaluated"
        ),
        answerability_avg=_score01(reports.get("item_answerability"), "average_answerability_score", "items_evaluated"),
        topic_coverage=(
            min(1.0, float(coverage.get("coverage_percent", 0.0)) / 100) if coverage.get("topics_total") else None
        ),
    )


def diagnostic_warnings(reports: dict[str, Any]) -> list[ReportWarning]:
    found = []
    for name in ("section_diagnostics", "question_diagnostics"):
        for warning in (reports.get(name) or {}).get("warnings", []):
            if warning.get("code"):
                found.append(ReportWarning(code=str(warning["code"]), message=str(warning.get("message", ""))))
    return found


def _pipeline_timings(reports: dict[str, Any]) -> dict[str, float]:
    timings: dict[str, float] = {}
    for name, generation, evaluation in (
        ("lo_processing_time", "learning_objects", "lo_evaluation"),
        ("item_processing_time", "questions", "q_evaluation"),
    ):
        report = reports.get(name) or {}
        if report:
            timings[generation] = max(0.0, float(report.get("generation_seconds", 0.0)))
            timings[evaluation] = max(0.0, float(report.get("evaluation_seconds", 0.0)))
    return timings
