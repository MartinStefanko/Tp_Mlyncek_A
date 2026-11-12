"""Output step: pipeline result → units (to ECEH) + report (internal diagnostics), steps 14 and 16.

    build_output   the whole step (below)
    check_output   check before saving (models, JSON Schemas, report ↔ units), used by the worker

Modules: split (split into lessons: topics, syllabus plan, lesson titles by the LLM), units
(mapping of the questions), limits (ECEH length limits, shortening by the LLM,
field_too_long), sanitize (hint HTML, nh3), report (report in the unchanged shape), check.
Nothing here imports from service/.

build_output:
    split (stage "split"): lessons, order of the questions, title and description of every
       lesson (qgen.output.split; also mode single gets its title from the LLM)
    then for every question of the pipeline (position = its 1-based order there):
    1. draft of the contract fields, hint HTML cleaned (invalid_hint)
    2. too long name / description / choices: one LLM call per question (shortening_failed)
    3. name cut at a word end; another field still too long → left out (field_too_long)
    4. contract Question; any other problem → left out (invalid_question)
    5. units by lessons, questions in the order of the split; questions of sections the
       syllabus left out are in no unit and have no question_meta; a lesson without an
       accepted question is left out (empty_lesson); no lesson at all → one empty unit
       (no_questions, the last warning), as the contract needs at least one
Warnings in the report: pipeline diagnostics, split warnings, then the warnings of the
questions (those of one question together, in the order of the questions).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pydantic import ValidationError

from qgen.cli import normalize_lo_for_export
from qgen.config import Settings
from qgen.contract.options import SingleSplit, SplitOptions
from qgen.contract.report import QuestionMeta, Report, ReportWarning
from qgen.contract.units import Question, Unit, UnitsResult
from qgen.llm.base import EmbeddingClient, LLMClient
from qgen.output.check import OutputCheckError, check_output
from qgen.output.limits import cut_at_word, enforce_limits, shorten_questions
from qgen.output.report import build_report, question_meta, topics_by_section
from qgen.output.sanitize import clean_hint_html
from qgen.output.split import run_split
from qgen.output.units import question_draft, unit_title

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
    split: Optional[SplitOptions] = None,
    llm: Optional[LLMClient] = None,
    embedder: Optional[EmbeddingClient] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> OutputResult:
    """`pipeline_result`: qgen.pipeline.PipelineResult; `input_names`: original names of the
    input PDFs in upload order (D1, D2, …); `split`: options.split (None = single);
    `llm` / `embedder`: clients for the split and the shortening (None = factory)."""
    sections = [normalize_lo_for_export(lo) for lo in pipeline_result.learning_objects]
    by_id = {lo["id"]: lo for lo in sections}
    file_names = {f"D{i}": name for i, name in enumerate(input_names, start=1)}
    language = _language(pipeline_result)

    split_start = time.perf_counter()
    outcome = run_split(
        pipeline_result,
        sections=sections,
        split=split or SingleSplit(),
        input_names=input_names,
        language=language,
        settings=settings,
        llm=llm,
        embedder=embedder,
        progress_callback=progress_callback,
    )
    split_seconds = time.perf_counter() - split_start
    topics = outcome.section_topics or topics_by_section(pipeline_result.reports)
    if progress_callback is not None:
        progress_callback("output", 0, 1)

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

    accepted: dict[int, Question] = {}
    dropped = 0
    for (position, _item), draft in zip(items, drafts):
        problem = _accept(draft, position)
        if isinstance(problem, ReportWarning):
            dropped += 1
            warnings[position].append(problem)
        else:
            accepted[position - 1] = problem

    units: list[Unit] = []
    meta: list[QuestionMeta] = []
    split_warnings = list(outcome.warnings)
    for number, (order, text) in enumerate(zip(outcome.question_order, outcome.texts), start=1):
        questions: list[Question] = []
        for index in order:
            if index not in accepted:
                continue
            position, item = items[index]
            meta.append(
                question_meta(
                    item, by_id.get(item.get("lo_id")), unit=len(units), question=len(questions),
                    position=position, file_names=file_names, topics=topics, warnings=warnings[position],
                )
            )
            questions.append(accepted[index])
        if not questions:
            split_warnings.append(
                ReportWarning(code="empty_lesson", message=f"Lesson {number} has no accepted question; it is left out.")
            )
            continue
        units.append(Unit(title=text.title, description=text.description, questions=questions))
    closing: list[ReportWarning] = []
    if not units:
        # One empty unit (the contract needs one); the empty lessons need no warning of their own.
        split_warnings = [w for w in split_warnings if w.code != "empty_lesson"]
        closing.append(
            ReportWarning(code="no_questions", message="No question was accepted; the result has one empty unit.")
        )
        units.append(Unit(title=unit_title(input_names), description="", questions=[]))

    report = build_report(
        sections,
        meta,
        reports=pipeline_result.reports,
        topics=topics,
        warnings=[
            *split_warnings,
            *(warning for position, _ in items for warning in warnings[position]),
            *closing,
        ],
        timings={**timings, "split": round(split_seconds, 4)},
    )
    return OutputResult(result=UnitsResult(units=units), report=report, dropped=dropped)


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
