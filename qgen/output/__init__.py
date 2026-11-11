"""Output step: pipeline result → units (to ECEH) + report (internal diagnostics), step 14.

    build_output   the whole step (below)
    check_output   check before saving (models, JSON Schemas, report ↔ units), used by the worker

Modules: units (mapping of the questions, unit title / description), limits (ECEH length
limits, shortening by the LLM, field_too_long), sanitize (hint HTML, nh3), report (report
in the unchanged shape), check. Nothing here imports from service/.

build_output, for every question of the pipeline (position = its 1-based order there):
    1. draft of the contract fields, hint HTML cleaned (invalid_hint)
    2. too long name / description / choices: one LLM call per question (shortening_failed)
    3. name cut at a word end; another field still too long → left out (field_too_long)
    4. contract Question; any other problem → left out (invalid_question)
    5. question_meta of the accepted questions; the report
Warnings of one question stay together, in the order of the questions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from pydantic import ValidationError

from qgen.cli import normalize_lo_for_export
from qgen.config import Settings
from qgen.contract.report import QuestionMeta, Report, ReportWarning
from qgen.contract.units import Question, Unit, UnitsResult
from qgen.llm.base import LLMClient
from qgen.output.check import OutputCheckError, check_output
from qgen.output.limits import cut_at_word, enforce_limits, shorten_questions
from qgen.output.report import build_report, question_meta, topics_by_section
from qgen.output.sanitize import clean_hint_html
from qgen.output.units import question_draft, unit_description, unit_title

__all__ = [
    "OutputCheckError",
    "OutputResult",
    "build_output",
    "check_output",
    "clean_hint_html",
    "cut_at_word",
]


@dataclass
class OutputResult:
    result: UnitsResult
    report: Report
    dropped: int


def build_output(
    pipeline_result: Any,
    *,
    input_names: list[str],
    timings: dict[str, float],
    settings: Settings,
    llm: Optional[LLMClient] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> OutputResult:
    """`pipeline_result`: qgen.pipeline.PipelineResult; `input_names`: original names of the
    input PDFs in upload order (D1, D2, …); `llm`: client for the shortening (None = factory)."""
    sections = [normalize_lo_for_export(lo) for lo in pipeline_result.learning_objects]
    by_id = {lo["id"]: lo for lo in sections}
    topics = topics_by_section(pipeline_result.reports)
    file_names = {f"D{i}": name for i, name in enumerate(input_names, start=1)}
    language = _language(pipeline_result)

    warnings: dict[int, list[ReportWarning]] = {}
    items = list(enumerate(pipeline_result.questions, start=1))
    drafts = []
    for position, item in items:
        draft, hint_warnings = question_draft(item, by_id.get(item.get("lo_id")), position)
        drafts.append(draft)
        warnings[position] = hint_warnings

    failed = shorten_questions(
        drafts, llm=llm, settings=settings, language=language, progress_callback=progress_callback
    )
    for index in failed:
        position = items[index][0]
        warnings[position].append(
            ReportWarning(code="shortening_failed", message=f"Question {position}: shortening by the LLM failed (llm_error).")
        )

    questions: list[Question] = []
    meta: list[QuestionMeta] = []
    dropped = 0
    for (position, item), draft in zip(items, drafts):
        problem = _accept(draft, position)
        if isinstance(problem, ReportWarning):
            dropped += 1
            warnings[position].append(problem)
            continue
        section = by_id.get(item.get("lo_id"))
        meta.append(
            question_meta(
                item, section, unit=0, question=len(questions), position=position,
                file_names=file_names, topics=topics, warnings=warnings[position],
            )
        )
        questions.append(problem)

    unit = Unit(title=unit_title(input_names), description=unit_description(sections), questions=questions)
    report = build_report(
        sections,
        meta,
        reports=pipeline_result.reports,
        topics=topics,
        warnings=[warning for position, _ in items for warning in warnings[position]],
        timings=timings,
    )
    return OutputResult(result=UnitsResult(units=[unit]), report=report, dropped=dropped)


def _accept(draft: dict, position: int) -> Question | ReportWarning:
    too_long = enforce_limits(draft)
    if too_long is not None:
        return ReportWarning(
            code="field_too_long",
            message=(
                f"Question {position} was left out: field {too_long.field} has {too_long.length} "
                f"characters (limit {too_long.limit})."
            ),
        )
    try:
        return Question.model_validate(draft)
    except ValidationError:
        return ReportWarning(code="invalid_question", message=f"Question {position} was left out: invalid_question.")


def _language(pipeline_result: Any) -> str:
    try:
        language = (pipeline_result.language_info or {}).get("language")
    except AttributeError:  # no language detection ran
        language = None
    return "en" if language == "en" else "sk"
