"""Split step (stage "split", kap. 8): the pure algorithm of qgen.split plus the calls it needs.

    topics        the document topics and the similarity section × topic, from the vectors the
                  topic coverage already computed (LOPipelineResult.embeddings); no new call
    syllabus      list from options → plan; syllabus_text → plan by the LLM (one repeat on an
                  invalid plan, then no plan → fallback to topics, syllabus_plan_invalid);
                  similarity section × syllabus lesson: one embedding call for the lessons
    split         qgen.split.split_lessons + order_questions
    titles        title and description of every lesson by the LLM, all lessons in calls of
                  QGEN_SPLIT_TITLE_BATCH_SIZE (also mode single: one lesson). Syllabus lessons
                  keep the teacher's title (and description, if given). A failed call →
                  titles from topics / sections, lesson_titles_failed. The job never fails here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
from pydantic import ValidationError

from qgen.config import Settings
from qgen.contract.options import SplitOptions
from qgen.contract.report import ReportWarning
from qgen.contract.units import UNIT_DESCRIPTION_MAX, UNIT_TITLE_MAX
from qgen.llm.base import EmbeddingClient, LLMClient
from qgen.llm.errors import LLMError
from qgen.llm.parallel import run_batches
from qgen.llm.retry import embed_texts, generate_json
from qgen.lo.topic_coverage import lo_to_text
from qgen.output.limits import cut_at_word
from qgen.output.prompts import (
    LessonPromptItem,
    PlanSection,
    build_plan_prompt,
    build_plan_retry_prompt,
    build_titles_prompt,
)
from qgen.output.sanitize import plain_text
from qgen.output.units import unit_description, unit_title
from qgen.split import (
    SplitQuestion,
    SplitResult,
    SyllabusPlan,
    build_split_sections,
    order_questions,
    plan_from_syllabus,
    split_lessons,
)

SPLIT_ROLE = "split"
STAGE = "split"
PLAN_ERRORS_SHOWN = 10

__all__ = ["LessonText", "SplitOutcome", "make_syllabus_plan", "run_split"]


@dataclass(frozen=True)
class LessonText:
    title: str
    description: str


@dataclass
class SplitOutcome:
    result: SplitResult
    question_order: list[list[int]]  # indexes into the pipeline questions, per lesson
    texts: list[LessonText]  # per lesson
    warnings: list[ReportWarning] = field(default_factory=list)  # split + plan + titles
    section_topics: dict[Any, str] = field(default_factory=dict)  # lo id → topic name (spec c)


class _Progress:
    def __init__(self, callback: Optional[Callable[[str, int, int], None]]) -> None:
        self.callback = callback
        self.done = 0
        self.total = 0

    def expect(self, calls: int) -> None:
        self.total += calls
        self.report()

    def step(self) -> None:
        self.done += 1
        self.report()

    def report(self) -> None:
        if self.callback is not None:
            self.callback(STAGE, self.done, max(self.total, self.done))


def run_split(
    pipeline_result: Any,
    *,
    sections: list[dict],
    split: SplitOptions,
    input_names: list[str],
    language: str,
    settings: Settings,
    llm: Optional[LLMClient] = None,
    embedder: Optional[EmbeddingClient] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> SplitOutcome:
    """`sections`: normalized sections (qgen.cli.normalize_lo_for_export) in pipeline order."""
    progress = _Progress(progress_callback)
    progress.report()
    questions = list(pipeline_result.questions)
    by_id = {lo["id"]: lo for lo in sections}
    warnings: list[ReportWarning] = []

    split_sections = build_split_sections(sections, questions)
    topics, similarity = _topic_similarity(getattr(pipeline_result.lo_result, "embeddings", None))
    section_topics = {
        sid: topics[max(range(len(topics)), key=lambda k: (row[k], -k))] for sid, row in similarity.items()
    } if topics else {}

    plan: Optional[SyllabusPlan] = None
    syllabus_similarity = None
    if split.mode == "syllabus":
        if split.syllabus is not None:
            plan = plan_from_syllabus(split.syllabus)
        else:
            progress.expect(1)
            plan_sections = [
                PlanSection(
                    id=s.id,
                    title=by_id[s.id]["vzdelavaci_objekt"],
                    pages=", ".join(f"{doc}:{page}" if doc else str(page) for doc, page in s.pages),
                    topic=section_topics.get(s.id),
                )
                for s in split_sections
            ]
            plan, problem = make_syllabus_plan(
                split.syllabus_text or "", plan_sections, language=language, llm=llm, settings=settings,
                on_call=progress.step,
            )
            if plan is None:
                warnings.append(
                    ReportWarning(
                        code="syllabus_plan_invalid",
                        message=f"The syllabus text could not be turned into a plan ({problem}).",
                    )
                )
        if plan is not None and plan.lessons and split_sections:
            progress.expect(1)
            syllabus_similarity = _syllabus_similarity(
                split_sections, plan, pipeline_result, embedder=embedder, settings=settings
            )
            progress.step()

    result = split_lessons(
        split_sections,
        split,
        topics=topics,
        topic_similarity=similarity or None,
        syllabus_similarity=syllabus_similarity,
        plan=plan,
    )
    known = {s.id for s in split_sections}
    order = order_questions(
        result,
        [
            SplitQuestion(key=index, section_id=item.get("lo_id"), type=item.get("type"))
            for index, item in enumerate(questions)
            if item.get("lo_id") in known
        ],
    )
    texts, title_warnings = _lesson_texts(
        result, by_id, input_names=input_names, language=language, settings=settings, llm=llm, progress=progress
    )
    progress.report()
    return SplitOutcome(
        result=result,
        question_order=order,
        texts=texts,
        warnings=[*warnings, *result.warnings, *title_warnings],
        section_topics=section_topics,
    )


# --- topics and similarity -----------------------------------------------------------------------


def _unit_rows(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _topic_similarity(embeddings: Any) -> tuple[list[str], dict[int, list[float]]]:
    if embeddings is None or not embeddings.topics or not embeddings.section_vectors:
        return [], {}
    ids = list(embeddings.section_vectors)
    cosine = _unit_rows([embeddings.section_vectors[i] for i in ids]) @ _unit_rows(embeddings.topic_vectors).T
    return list(embeddings.topics), {sid: [float(x) for x in row] for sid, row in zip(ids, cosine)}


def _syllabus_similarity(split_sections, plan: SyllabusPlan, pipeline_result, *, embedder, settings) -> list[list[float]]:
    """Cosine section × syllabus lesson (title + description). Section vectors come from the
    pipeline; sections without one are embedded here (same text as the topic coverage)."""
    embeddings = getattr(pipeline_result.lo_result, "embeddings", None)
    known = dict(embeddings.section_vectors) if embeddings is not None else {}
    missing = [s.id for s in split_sections if s.id not in known]
    texts = [" ".join(filter(None, [lesson.title, lesson.description or ""])) for lesson in plan.lessons]
    if missing:
        los = {lo.get("id"): lo for lo in pipeline_result.learning_objects}
        vectors = embed_texts(embedder, [lo_to_text(los.get(sid, {})) for sid in missing] + texts, settings=settings)
        known.update(zip(missing, vectors[: len(missing)]))
        lesson_vectors = vectors[len(missing) :]
    else:
        lesson_vectors = embed_texts(embedder, texts, settings=settings)
    cosine = _unit_rows([known[s.id] for s in split_sections]) @ _unit_rows(lesson_vectors).T
    return [[float(x) for x in row] for row in cosine]


# --- syllabus plan ---------------------------------------------------------------------------------


def make_syllabus_plan(
    syllabus_text: str,
    sections: Sequence[PlanSection],
    *,
    language: str,
    llm: Optional[LLMClient],
    settings: Settings,
    on_call: Optional[Callable[[], None]] = None,
) -> tuple[Optional[SyllabusPlan], Optional[str]]:
    """(plan, None), or (None, reason). An invalid plan gets one repeated call with the list
    of its errors (paths and kinds only); LLMError → no plan."""
    prompt = build_plan_prompt(syllabus_text, sections, language)
    current = prompt
    errors = ""
    for _attempt in range(2):
        try:
            rows = generate_json(llm, current, role=SPLIT_ROLE, settings=settings, items_key="items")
        except LLMError:
            return None, "llm_error"
        finally:
            if on_call is not None:
                on_call()
        try:
            if len(rows) != 1:
                raise ValueError("items must contain exactly one object")
            return SyllabusPlan.model_validate(rows[0]), None
        except ValidationError as error:
            errors = "; ".join(
                f"{'.'.join(str(part) for part in e['loc']) or 'plan'}: {e['type']}"
                for e in error.errors(include_url=False)[:PLAN_ERRORS_SHOWN]
            )
        except ValueError as error:
            errors = str(error)
        current = build_plan_retry_prompt(prompt, errors, language)
    return None, f"invalid plan twice: {errors}"


# --- lesson titles ---------------------------------------------------------------------------------


def _lesson_texts(result: SplitResult, by_id, *, input_names, language, settings, llm, progress) -> tuple[list[LessonText], list[ReportWarning]]:
    teacher: list[tuple[Optional[str], Optional[str]]] = []
    for lesson in result.lessons:
        planned = None
        if result.plan is not None and lesson.syllabus_index is not None:
            planned = result.plan.lessons[lesson.syllabus_index]
        teacher.append((planned.title, planned.description) if planned else (None, None))

    asked = [
        LessonPromptItem(
            number=number,
            topics=lesson.topics,
            teacher_title=teacher[number - 1][0],
            section_titles=tuple(by_id[sid]["vzdelavaci_objekt"] for sid in lesson.section_ids),
        )
        for number, lesson in enumerate(result.lessons, start=1)
        if not teacher[number - 1][1]  # the teacher's description is kept, no call for it
    ]
    batch = max(1, int(settings.split_title_batch_size))
    batches = [asked[i : i + batch] for i in range(0, len(asked), batch)]
    progress.expect(len(batches))

    def task(lessons: list[LessonPromptItem]) -> Callable[[], Optional[list]]:
        def call() -> Optional[list]:
            try:
                return generate_json(
                    llm, build_titles_prompt(lessons, language), role=SPLIT_ROLE, settings=settings, items_key="items"
                )
            except LLMError:
                return None

        return call

    answers = run_batches([task(b) for b in batches], settings=settings, on_done=lambda _i, _a: progress.step())

    found: dict[int, tuple[str, str]] = {}
    warnings: list[ReportWarning] = []
    for lessons, answer in zip(batches, answers):
        numbers = {lesson.number for lesson in lessons}
        missing = set(numbers)
        for row in answer or []:
            if not isinstance(row, dict) or row.get("lesson") not in numbers:
                continue
            title = cut_at_word(plain_text(str(row.get("title") or "")), UNIT_TITLE_MAX)
            description = cut_at_word(plain_text(str(row.get("description") or "")), UNIT_DESCRIPTION_MAX)
            if title:
                found[row["lesson"]] = (title, description)
                missing.discard(row["lesson"])
        if missing:
            reason = "the call failed" if answer is None else "no title in the answer"
            warnings.append(
                ReportWarning(
                    code="lesson_titles_failed",
                    message=(
                        f"Lessons {', '.join(str(n) for n in sorted(missing))}: {reason}; "
                        "titles made from the topics or sections."
                    ),
                )
            )

    texts = []
    for number, lesson in enumerate(result.lessons, start=1):
        teacher_title, teacher_description = teacher[number - 1]
        generated_title, generated_description = found.get(number, (None, None))
        sections = [by_id[sid] for sid in lesson.section_ids]
        title = teacher_title or generated_title or _fallback_title(lesson.topics, sections, input_names, language)
        description = teacher_description or generated_description or unit_description(sections)
        texts.append(
            LessonText(
                title=cut_at_word(" ".join(title.split()), UNIT_TITLE_MAX) or unit_title(input_names),
                description=cut_at_word(description, UNIT_DESCRIPTION_MAX),
            )
        )
    return texts, warnings


def _fallback_title(topics: Sequence[str], sections: list[dict], input_names: list[str], language: str) -> str:
    if topics:
        return ", ".join(topics)
    names = [lo["vzdelavaci_objekt"] for lo in sections if lo.get("vzdelavaci_objekt")]
    if names:
        more = " and more" if language == "en" else " a ďalšie"
        return names[0] + (more if len(names) > 1 else "")
    return unit_title(input_names)
