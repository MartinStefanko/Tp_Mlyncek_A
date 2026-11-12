"""Syllabus plan: the structured form of the teacher's syllabus (split mode "syllabus").

The list from options (`split.syllabus`) becomes a plan with `plan_from_syllabus`; free text
(`split.syllabus_text`) is turned into a plan by the LLM in step 16 – the text itself is never
executed. A plan either lists lessons (each with pages OR section ids OR neither) and the
sections to leave out, or gives only lesson_count ("rozdeľ na 5 hodín", the lessons
algorithm) or only lesson_minutes ("hodiny po 90 minút": the length of a lesson is not used,
the split goes by topics with a warning).

`assign_sections` applies a plan with lessons. A bad plan never raises: unknown ids and
duplicates are warnings; a plan that leaves no lesson with a section is reported as
`SyllabusAssignment.fallback_reason` (the caller then splits by topics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from qgen.contract.options import (
    PAGE_RANGES_PATTERN,
    SYLLABUS_MAX_LESSONS,
    LessonCount,
    LessonMinutes,
    SyllabusLesson,
    page_range_errors,
    parse_page_ranges,
)
from qgen.contract.report import ReportWarning
from qgen.contract.units import UNIT_DESCRIPTION_MAX, UNIT_TITLE_MAX
from qgen.split.sections import SplitSection, document_number, section_label

__all__ = [
    "PlannedLesson",
    "SyllabusAssignment",
    "SyllabusPlan",
    "assign_sections",
    "check_plan_ids",
    "plan_from_syllabus",
]


class _PlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlannedLesson(_PlanModel):
    title: str = Field(min_length=1, max_length=UNIT_TITLE_MAX)
    description: Optional[str] = Field(default=None, max_length=UNIT_DESCRIPTION_MAX)
    pages: Optional[str] = Field(default=None, pattern=PAGE_RANGES_PATTERN)
    section_ids: list[int] = Field(default_factory=list, description="Section (lo) ids of the lesson.")

    @model_validator(mode="after")
    def _pages_or_sections(self) -> "PlannedLesson":
        if self.pages is not None and self.section_ids:
            raise ValueError("a planned lesson has pages or section_ids, not both")
        errors = page_range_errors(self.pages) if self.pages is not None else []
        if errors:
            raise ValueError("pages: " + "; ".join(errors))
        return self


class SyllabusPlan(_PlanModel):
    lessons: list[PlannedLesson] = Field(default_factory=list, max_length=SYLLABUS_MAX_LESSONS)
    skipped_section_ids: list[int] = Field(default_factory=list, description="Sections left out of all lessons.")
    lesson_count: Optional[LessonCount] = None
    lesson_minutes: Optional[LessonMinutes] = Field(
        default=None, description="Lesson length asked for in the text; not used (warning)."
    )

    @model_validator(mode="after")
    def _lessons_or_count(self) -> "SyllabusPlan":
        if self.lessons and self.lesson_count is not None:
            raise ValueError("a syllabus plan has lessons or lesson_count, not both")
        if not self.lessons and self.lesson_count is None and self.lesson_minutes is None:
            raise ValueError("a syllabus plan needs lessons, lesson_count or lesson_minutes")
        return self


def plan_from_syllabus(syllabus: Sequence[SyllabusLesson]) -> SyllabusPlan:
    """The teacher's list of lessons (split.syllabus) as a plan."""
    return SyllabusPlan(
        lessons=[PlannedLesson(title=x.title, description=x.description, pages=x.pages) for x in syllabus]
    )


@dataclass
class SyllabusAssignment:
    groups: list[tuple[int, list[SplitSection]]]  # (index in plan.lessons, sections in order)
    skipped: list[int] = field(default_factory=list)
    fallback_reason: Optional[str] = None


def check_plan_ids(
    plan: SyllabusPlan, known: set[int], warnings: list[ReportWarning]
) -> tuple[dict[int, int], set[int]]:
    """Section ids of the plan: (section → lesson index, skipped sections).

    Unknown id → ignored (`syllabus_unknown_section`); a section in more lessons stays in
    the first one, a section in a lesson and in the skipped list stays in the lesson
    (`syllabus_duplicate_section`).
    """
    claimed: dict[int, int] = {}
    unknown: list[int] = []
    for index, lesson in enumerate(plan.lessons):
        for sid in lesson.section_ids:
            if sid not in known:
                unknown.append(sid)
            elif sid not in claimed:
                claimed[sid] = index
            elif claimed[sid] != index:
                warnings.append(
                    ReportWarning(
                        code="syllabus_duplicate_section",
                        message=(
                            f"Syllabus: {section_label(sid)} is in lessons {claimed[sid] + 1} and "
                            f"{index + 1}; it stays in lesson {claimed[sid] + 1}."
                        ),
                    )
                )
    skipped: set[int] = set()
    for sid in plan.skipped_section_ids:
        if sid not in known:
            unknown.append(sid)
        elif sid in claimed:
            warnings.append(
                ReportWarning(
                    code="syllabus_duplicate_section",
                    message=(
                        f"Syllabus: {section_label(sid)} is in lesson {claimed[sid] + 1} and among the "
                        "skipped sections; it stays in the lesson."
                    ),
                )
            )
        else:
            skipped.add(sid)
    for sid in dict.fromkeys(unknown):
        warnings.append(
            ReportWarning(
                code="syllabus_unknown_section",
                message=f"Syllabus: section id {sid} does not exist; ignored.",
            )
        )
    return claimed, skipped


def assign_sections(
    ordered: list[SplitSection],
    plan: SyllabusPlan,
    *,
    claimed: dict[int, int],
    skipped: set[int],
    similarity: Optional[dict[int, Sequence[float]]],
    warnings: list[ReportWarning],
) -> SyllabusAssignment:
    """Sections (in the order of step a) → lessons of a plan with lessons.

    Listed ids go to their lesson; the other sections (not skipped) to the lesson with most
    matching pages (lessons given by section ids do not take part), ties by similarity, then
    the earlier lesson; without a matching page to the most similar lesson.
    `similarity[section id][lesson index]`.
    """
    ranges = {
        index: parse_page_ranges(lesson.pages)
        for index, lesson in enumerate(plan.lessons)
        if lesson.pages is not None and not lesson.section_ids
    }
    lesson_of: dict[int, int] = {}
    for section in ordered:
        if section.id in skipped:
            continue
        if section.id in claimed:
            lesson_of[section.id] = claimed[section.id]
            continue
        row = similarity.get(section.id) if similarity is not None else None
        scores = {index: _page_matches(section, found) for index, found in ranges.items()}
        best = max(scores.values(), default=0)
        if best > 0:
            candidates = [index for index, score in scores.items() if score == best]
        else:
            if row is None:
                raise ValueError(
                    f"similarity is needed for section {section.id}: no page of it is in the syllabus"
                )
            candidates = list(range(len(plan.lessons)))
        lesson_of[section.id] = min(candidates, key=lambda i: (-(row[i] if row is not None else 0.0), i))

    groups = []
    for index in range(len(plan.lessons)):
        members = [s for s in ordered if lesson_of.get(s.id) == index]
        if members:
            groups.append((index, members))
        else:
            warnings.append(
                ReportWarning(
                    code="empty_lesson",
                    message=f"Syllabus lesson {index + 1} got no section; it is left out.",
                )
            )
    result = SyllabusAssignment(groups=groups, skipped=[s.id for s in ordered if s.id in skipped])
    if not groups:
        result.fallback_reason = "no lesson of the syllabus got a section"
    return result


def _page_matches(section: SplitSection, ranges: list[tuple[int, int, int]]) -> int:
    count = 0
    for source_id, page in section.pages:
        doc = document_number(source_id)
        if doc is not None and any(d == doc and first <= page <= last for d, first, last in ranges):
            count += 1
    return count
