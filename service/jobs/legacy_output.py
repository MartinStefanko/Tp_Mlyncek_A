"""DOČASNÉ (TEMPORARY) mapping of today's pipeline output to the API contract.

Step 14 replaces this module by the real output step (new question types from the LLM,
split into lessons, shortening by the LLM). Until then the worker maps the current
questions and sections (tematické celky, "LO" in the original code) as follows
(kap. 6 – Mapovanie zo súčasného výstupu):

    units           one lesson (split "single"); title = names of the input PDFs,
                    description = list of section titles
    name            title of the question's section, cut at a word end (no LLM name yet)
    description     otazka
    type            teoreticka_otazka and prakticka_uloha → "open" (no choices exist yet)
    choices         [{text: odpoved, is_correct: true}]
    hints           napoveda → [{name: "Nápoveda", url: "<p>…</p>"}] (HTML cleaned by nh3)
    time_*          defaults by type (kap. 7): open 300 / 0

A question that does not pass the units contract is left out with a report warning.
The report: sections from the accepted sections, question_meta per question, quality
converted (x − 1) / 4 and coverage / 100, warnings, timings.
"""

import html
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Optional

import nh3
from pydantic import ValidationError

from qgen.cli import normalize_lo_for_export
from qgen.context_builder import parse_source_refs
from qgen.contract import Report, UnitsResult
from qgen.contract.report import Quality, QuestionMeta, ReportWarning, Section, SourceRef
from qgen.contract.units import (
    QUESTION_NAME_MAX,
    UNIT_DESCRIPTION_MAX,
    UNIT_TITLE_MAX,
    Question,
    Unit,
)
from qgen.pipeline import PipelineResult
from service.jobs.models import JobRecord

# Default times per type in seconds (kap. 7): (time_to_handle, time_to_explain).
DEFAULT_TIMES: dict[str, tuple[int, int]] = {"choice": (120, 30), "open": (300, 0), "no_answer": (300, 0)}
LEGACY_TYPES: dict[str, str] = {"teoreticka_otazka": "open", "prakticka_uloha": "open"}
HINT_NAME = "Nápoveda"
HINT_TAGS = {"p", "br", "strong", "em", "ul", "ol", "li", "code", "pre", "sub", "sup"}
LEGACY_LO_TYPE = "legacy"

BLOOM_LEVELS: dict[str, str] = {
    "zapamätať si": "remember",
    "zapamätať": "remember",
    "pochopiť": "understand",
    "porozumieť": "understand",
    "aplikovať": "apply",
    "analyzovať": "analyze",
    "hodnotiť": "evaluate",
    "vytvoriť": "create",
    "remember": "remember",
    "understand": "understand",
    "apply": "apply",
    "analyze": "analyze",
    "evaluate": "evaluate",
    "create": "create",
}
BLOOM_FALLBACK = "understand"


@dataclass
class LegacyOutput:
    result: UnitsResult
    report: Report
    dropped: int


def clean_hint_html(text: str) -> str:
    """Allowed tags only (kap. 6), every attribute removed."""
    return nh3.clean(text, tags=HINT_TAGS, attributes={}, link_rel=None)


def cut_at_word(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit + 1].rsplit(" ", 1)[0] if " " in text[: limit + 1] else text[:limit]
    return cut[:limit].rstrip(" ,;:-–")


def section_id(lo_id: Any) -> str:
    return f"S-{lo_id}"


def build_output(pipeline_result: PipelineResult, record: JobRecord, *, timings: dict[str, float]) -> LegacyOutput:
    warnings: list[ReportWarning] = []
    sections_raw = [normalize_lo_for_export(lo) for lo in pipeline_result.learning_objects]
    by_id = {lo["id"]: lo for lo in sections_raw}
    topics = _topics_by_section(pipeline_result.reports)
    file_names = {f"D{i}": f.name for i, f in enumerate(record.input_files, start=1)}

    questions: list[Question] = []
    meta: list[QuestionMeta] = []
    dropped = 0
    for position, item in enumerate(pipeline_result.questions, start=1):
        section = by_id.get(item.get("lo_id"))
        question, problem = _question(item, section)
        if question is None:
            dropped += 1
            warnings.append(ReportWarning(code=problem, message=f"Question {position} was left out: {problem}."))
            continue
        bloom = _bloom(section, warnings, position)
        meta.append(
            QuestionMeta(
                unit=0,
                question=len(questions),
                section_id=section_id(item.get("lo_id")),
                bloom_level=bloom,
                keywords=[],
                source=_sources(item.get("citovane_zdroje", []), file_names),
                topic=topics.get(item.get("lo_id")),
            )
        )
        questions.append(question)

    unit = Unit(
        title=_unit_title(record),
        description=cut_at_word(
            "Tematické celky: " + ", ".join(lo["vzdelavaci_objekt"] for lo in sections_raw if lo["vzdelavaci_objekt"]),
            UNIT_DESCRIPTION_MAX,
        )
        if sections_raw
        else "",
        questions=questions,
    )
    warnings[:0] = _diagnostic_warnings(pipeline_result.reports)
    report = Report(
        sections=[_section(lo, topics.get(lo["id"])) for lo in sections_raw],
        question_meta=meta,
        quality=_quality(pipeline_result.reports),
        warnings=warnings,
        timings={**timings, **_pipeline_timings(pipeline_result.reports)},
    )
    return LegacyOutput(result=UnitsResult(units=[unit]), report=report, dropped=dropped)


# --- questions ------------------------------------------------------------------------------


def _question(item: dict, section: Optional[dict]) -> tuple[Optional[Question], str]:
    question_type = LEGACY_TYPES.get(str(item.get("typ", "")), "open")
    handle, explain = DEFAULT_TIMES[question_type]
    name_source = (section or {}).get("vzdelavaci_objekt") or str(item.get("otazka", ""))
    hint_text = str(item.get("napoveda", "") or "").strip()
    hints = []
    if hint_text:
        hints.append({"name": HINT_NAME, "url": clean_hint_html(f"<p>{html.escape(hint_text, quote=False)}</p>")})
    data = {
        "name": cut_at_word(str(name_source), QUESTION_NAME_MAX),
        "description": str(item.get("otazka", "") or "").strip(),
        "type": question_type,
        "time_to_handle": handle,
        "time_to_explain": explain,
        "hints": hints,
        "choices": [{"text": str(item.get("odpoved", "") or "").strip(), "is_correct": True}],
    }
    try:
        return Question.model_validate(data), ""
    except ValidationError as exc:
        too_long = any(error["type"] == "string_too_long" for error in exc.errors())
        return None, "field_too_long" if too_long else "invalid_question"


def _bloom(section: Optional[dict], warnings: list[ReportWarning], position: int) -> str:
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


# --- report ---------------------------------------------------------------------------------


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


def _topics_by_section(reports: dict[str, Any]) -> dict[Any, str]:
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


def _diagnostic_warnings(reports: dict[str, Any]) -> list[ReportWarning]:
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


def _unit_title(record: JobRecord) -> str:
    names = [PurePath(f.name).stem or f.name for f in record.input_files]
    return cut_at_word(", ".join(names) or "Lekcia", UNIT_TITLE_MAX) or "Lekcia"

