"""Split of the sections (tematické celky) into lessons (kap. 8, Algoritmus a–e, g).

Pure functions: no LLM, no embeddings. The topics of the sections and the similarity of
sections to syllabus lessons come ready from the caller (step 16), as do the title and
description of a lesson (step f).

    a. order        topological by prerequisites, ties by pages (qgen.split.order)
    b. time         section = Σ time_to_handle + time_to_explain; capacity C = lesson_minutes × 60
    d. split        single: one lesson
                    lessons: N contiguous runs by dynamic programming, smallest Σ (T − C)²,
                             a cut between two topics preferred; N = lesson_count, else
                             max(1, round(Σ / C))
                    topics: lesson = run of one topic; long topic split as in lessons, short
                            neighbours merged
                    syllabus: by the plan (qgen.split.syllabus); a plan that cannot be used →
                              topics (fallback)
    e. questions    in the order of the sections; in a section choice → open → no_answer
    g. checks       prerequisites, empty lessons, time against the capacity

A section longer than the capacity is a lesson of its own and is not divided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from qgen.contract.options import OptionDefaults, SplitOptions
from qgen.contract.report import ReportWarning
from qgen.split.order import order_sections
from qgen.split.sections import SplitQuestion, SplitSection, section_label
from qgen.split.syllabus import SyllabusPlan, assign_sections, check_plan_ids, plan_from_syllabus

# Allowed deviation of a lesson from the capacity (lessons, topics): ±20 % (kap. 8 g).
DEVIATION_PERCENT = 20
# Extra cost of a cut inside one topic, as a share of C²: a cut between two topics wins when
# it makes the two neighbouring lessons worse by about ±10 % at most.
TOPIC_CUT_PENALTY = 0.02
# topics: a lesson under 80 % of C is short; two neighbours are merged up to 120 % of C.
SHORT_PERCENT = 80
MERGE_LIMIT_PERCENT = 120

QUESTION_TYPE_ORDER = {"choice": 0, "open": 1, "no_answer": 2}

__all__ = [
    "Lesson",
    "SplitResult",
    "order_questions",
    "split_lessons",
]


@dataclass(frozen=True)
class Lesson:
    section_ids: tuple[int, ...]
    seconds: int
    topics: tuple[str, ...] = ()
    syllabus_index: Optional[int] = None  # index in the syllabus plan (mode syllabus)


@dataclass
class SplitResult:
    lessons: list[Lesson]
    warnings: list[ReportWarning] = field(default_factory=list)
    skipped_section_ids: list[int] = field(default_factory=list)  # left out by the syllabus
    fallback: Optional[str] = None  # "topics" when the syllabus could not be used


def split_lessons(
    sections: Sequence[SplitSection],
    split: SplitOptions,
    similarity: Optional[Sequence[Sequence[float]]] = None,
    plan: Optional[SyllabusPlan] = None,
) -> SplitResult:
    """Split the sections into lessons by options.split.

    `similarity[i][j]`: similarity of `sections[i]` to lesson j of the syllabus plan (mode
    syllabus only; needed for sections that no page or id assigns). `plan`: the syllabus as
    a plan (step 16 makes it from syllabus_text); without it the list split.syllabus is used.
    No sections → no lessons.
    """
    sections = list(sections)
    if not sections:
        return SplitResult(lessons=[])
    order = order_sections(sections)
    ordered = order.sections
    warnings = list(order.warnings)
    minutes = split.lesson_minutes if split.lesson_minutes is not None else OptionDefaults().lesson_minutes
    capacity = minutes * 60

    if split.mode == "single":
        result = SplitResult(lessons=[_lesson(ordered)], warnings=warnings)
        _check_over_capacity(result, capacity)
    elif split.mode == "lessons":
        result = SplitResult(lessons=_lessons_mode(ordered, capacity, split.lesson_count, warnings), warnings=warnings)
        _check_deviation(result, capacity)
    elif split.mode == "topics":
        result = SplitResult(lessons=_topics_mode(ordered, capacity, warnings), warnings=warnings)
        _check_deviation(result, capacity)
    else:
        result = _syllabus_mode(sections, ordered, split, similarity, plan, capacity, warnings)

    _check_not_empty(result)
    _check_prerequisites(result, sections, order.broken)
    return result


def order_questions(result: SplitResult, questions: Sequence[SplitQuestion]) -> list[list[Any]]:
    """Keys of the questions per lesson: sections in the order of the lesson, in a section
    choice → open → no_answer, otherwise the given order. Questions of sections the syllabus
    left out are not in any lesson; a question of an unknown section is a ValueError."""
    place: dict[int, tuple[int, int]] = {}
    for lesson_index, lesson in enumerate(result.lessons):
        for position, sid in enumerate(lesson.section_ids):
            place[sid] = (lesson_index, position)
    skipped = set(result.skipped_section_ids)
    ordered: list[list[tuple[tuple, Any]]] = [[] for _ in result.lessons]
    for index, question in enumerate(questions):
        if question.section_id in skipped:
            continue
        if question.section_id not in place:
            raise ValueError(f"question {question.key!r}: section {question.section_id} is in no lesson")
        lesson_index, position = place[question.section_id]
        ordered[lesson_index].append(((position, QUESTION_TYPE_ORDER[question.type], index), question.key))
    return [[key for _, key in sorted(items, key=lambda x: x[0])] for items in ordered]


# --- modes ------------------------------------------------------------------------------------


def _lesson(sections: Sequence[SplitSection], syllabus_index: Optional[int] = None) -> Lesson:
    topics: list[str] = []
    for section in sections:
        if section.topic is not None and section.topic not in topics:
            topics.append(section.topic)
    return Lesson(
        section_ids=tuple(s.id for s in sections),
        seconds=sum(s.seconds for s in sections),
        topics=tuple(topics),
        syllabus_index=syllabus_index,
    )


def _lessons_mode(
    ordered: list[SplitSection], capacity: int, lesson_count: Optional[int], warnings: list[ReportWarning]
) -> list[Lesson]:
    _warn_long_sections(ordered, capacity, warnings)
    total = sum(s.seconds for s in ordered)
    wanted = lesson_count if lesson_count is not None else max(1, round(total / capacity))
    count = min(max(wanted, _forced_runs(ordered, capacity)), len(ordered))
    if lesson_count is not None and count < lesson_count:
        warnings.append(
            ReportWarning(
                code="lesson_count_reduced",
                message=f"lesson_count {lesson_count} is more than the {len(ordered)} sections; {count} lessons made.",
            )
        )
    elif lesson_count is not None and count > lesson_count:
        warnings.append(
            ReportWarning(
                code="lesson_count_increased",
                message=(
                    f"lesson_count {lesson_count} is too small: sections longer than a lesson need "
                    f"lessons of their own; {count} lessons made."
                ),
            )
        )
    return [_lesson(run) for run in _split_runs(ordered, count, capacity)]


def _topics_mode(ordered: list[SplitSection], capacity: int, warnings: list[ReportWarning]) -> list[Lesson]:
    _warn_long_sections(ordered, capacity, warnings)
    lessons: list[list[SplitSection]] = []
    for group in _topic_groups(ordered):
        total = sum(s.seconds for s in group)
        count = min(max(max(1, round(total / capacity)), _forced_runs(group, capacity)), len(group))
        lessons.extend(_split_runs(group, count, capacity))
    return [_lesson(run) for run in _merge_short(lessons, capacity)]


def _syllabus_mode(sections, ordered, split, similarity, plan, capacity, warnings) -> SplitResult:
    if plan is None:
        if split.syllabus is None:
            return _fallback(ordered, capacity, warnings, "the syllabus text was not turned into a plan")
        plan = plan_from_syllabus(split.syllabus)

    claimed, skipped = check_plan_ids(plan, {s.id for s in ordered}, warnings)
    remaining = [s for s in ordered if s.id not in skipped]
    if not remaining:
        return _fallback(ordered, capacity, warnings, "the syllabus leaves out every section")

    if plan.lesson_minutes is not None:
        capacity = plan.lesson_minutes * 60
    if not plan.lessons:
        # Only lesson_count / lesson_minutes ("rozdeľ na 5 hodín"): the lessons algorithm.
        result = SplitResult(
            lessons=_lessons_mode(remaining, capacity, plan.lesson_count, warnings),
            warnings=warnings,
            skipped_section_ids=[s.id for s in ordered if s.id in skipped],
        )
        _warn_skipped(result)
        _check_deviation(result, capacity)
        return result

    rows = None
    if similarity is not None:
        if len(similarity) != len(sections) or any(len(row) != len(plan.lessons) for row in similarity):
            raise ValueError(
                f"similarity must be {len(sections)} × {len(plan.lessons)} (sections × syllabus lessons)"
            )
        rows = {section.id: row for section, row in zip(sections, similarity)}
    assignment = assign_sections(ordered, plan, claimed=claimed, skipped=skipped, similarity=rows, warnings=warnings)
    if assignment.fallback_reason is not None:
        return _fallback(ordered, capacity, warnings, assignment.fallback_reason)
    result = SplitResult(
        lessons=[_lesson(members, syllabus_index=index) for index, members in assignment.groups],
        warnings=warnings,
        skipped_section_ids=assignment.skipped,
    )
    _warn_skipped(result)
    _check_over_capacity(result, capacity)
    return result


def _fallback(ordered, capacity, warnings, reason: str) -> SplitResult:
    warnings.append(
        ReportWarning(code="syllabus_fallback", message=f"Syllabus not used ({reason}); lessons split by topics.")
    )
    result = SplitResult(lessons=_topics_mode(ordered, capacity, warnings), warnings=warnings, fallback="topics")
    _check_deviation(result, capacity)
    return result


def _warn_skipped(result: SplitResult) -> None:
    if result.skipped_section_ids:
        result.warnings.append(
            ReportWarning(
                code="syllabus_skipped_sections",
                message="Sections left out by the syllabus: "
                + ", ".join(section_label(sid) for sid in result.skipped_section_ids)
                + ".",
            )
        )


# --- dynamic programming ------------------------------------------------------------------


def _is_long(section: SplitSection, capacity: int) -> bool:
    return section.seconds > capacity


def _forced_runs(sections: Sequence[SplitSection], capacity: int) -> int:
    """Least number of lessons: every long section alone, plus one per non-empty gap."""
    runs = 0
    in_gap = False
    for section in sections:
        if _is_long(section, capacity):
            runs += 1
            in_gap = False
        elif not in_gap:
            runs += 1
            in_gap = True
    return runs


def _cut_penalty(before: SplitSection, after: SplitSection, capacity: int) -> int:
    if before.topic is not None and after.topic is not None and before.topic != after.topic:
        return 0
    return round(TOPIC_CUT_PENALTY * capacity * capacity)


def _split_runs(sections: list[SplitSection], count: int, capacity: int) -> list[list[SplitSection]]:
    """`count` contiguous non-empty runs with the smallest Σ (T − C)² + cut penalties.

    Integer costs and the earliest cut on a tie, so the result is deterministic. A run with a
    long section and anything else is not allowed; `count` must lie between _forced_runs and
    len(sections).
    """
    n = len(sections)
    prefix = [0]
    long_prefix = [0]
    for section in sections:
        prefix.append(prefix[-1] + section.seconds)
        long_prefix.append(long_prefix[-1] + int(_is_long(section, capacity)))

    def allowed(i: int, j: int) -> bool:
        return j - i == 1 or long_prefix[j] == long_prefix[i]

    penalty = [0] + [_cut_penalty(sections[i - 1], sections[i], capacity) for i in range(1, n)]
    inf = None
    best: list[list[Optional[int]]] = [[inf] * (n + 1) for _ in range(count + 1)]
    back = [[0] * (n + 1) for _ in range(count + 1)]
    best[0][0] = 0
    for k in range(1, count + 1):
        for j in range(k, n + 1):
            for i in range(k - 1, j):
                previous = best[k - 1][i]
                if previous is None or not allowed(i, j):
                    continue
                cost = previous + (prefix[j] - prefix[i] - capacity) ** 2 + (penalty[i] if i > 0 else 0)
                if best[k][j] is None or cost < best[k][j]:
                    best[k][j] = cost
                    back[k][j] = i
    if best[count][n] is None:
        raise AssertionError(f"no split of {n} sections into {count} lessons")
    runs = []
    j = n
    for k in range(count, 0, -1):
        i = back[k][j]
        runs.append(sections[i:j])
        j = i
    return runs[::-1]


# --- topics -----------------------------------------------------------------------------------


def _topic_groups(ordered: list[SplitSection]) -> list[list[SplitSection]]:
    """Maximal runs of one topic in the order of step a. A section without a topic continues
    the run before it (the first ones join the run after them)."""
    groups: list[list[SplitSection]] = []
    topics: list[Optional[str]] = []
    for section in ordered:
        if groups and (section.topic is None or section.topic == topics[-1] or topics[-1] is None):
            groups[-1].append(section)
            if topics[-1] is None:
                topics[-1] = section.topic
        else:
            groups.append([section])
            topics.append(section.topic)
    return groups


def _merge_short(lessons: list[list[SplitSection]], capacity: int) -> list[list[SplitSection]]:
    """Merge neighbours while one of them is short (< 80 % C) and together they fit in
    120 % C; the pair with the smallest sum first, the left one on a tie. A lesson with a
    section longer than C is never merged."""
    lessons = [list(run) for run in lessons]

    def seconds(run):
        return sum(s.seconds for s in run)

    def mergeable(run):
        return not any(_is_long(s, capacity) for s in run)

    while True:
        candidates = []
        for i in range(len(lessons) - 1):
            a, b = seconds(lessons[i]), seconds(lessons[i + 1])
            short = 100 * a < SHORT_PERCENT * capacity or 100 * b < SHORT_PERCENT * capacity
            fits = 100 * (a + b) <= MERGE_LIMIT_PERCENT * capacity
            if short and fits and mergeable(lessons[i]) and mergeable(lessons[i + 1]):
                candidates.append((a + b, i))
        if not candidates:
            return lessons
        _, i = min(candidates)
        lessons[i : i + 2] = [lessons[i] + lessons[i + 1]]


# --- checks (g) ------------------------------------------------------------------------------


def _warn_long_sections(ordered: list[SplitSection], capacity: int, warnings: list[ReportWarning]) -> None:
    for section in ordered:
        if _is_long(section, capacity):
            warnings.append(
                ReportWarning(
                    code="section_over_capacity",
                    message=(
                        f"{section_label(section.id)} takes {_minutes(section.seconds)}, more than one "
                        f"lesson ({_minutes(capacity)}); it is a lesson of its own."
                    ),
                )
            )


def _check_deviation(result: SplitResult, capacity: int) -> None:
    """lessons, topics: deviation over ±20 % of the capacity that could not be removed. A lesson
    of one section longer than the capacity already has section_over_capacity."""
    for number, lesson in enumerate(result.lessons, start=1):
        if len(lesson.section_ids) == 1 and lesson.seconds > capacity:
            continue
        if 100 * abs(lesson.seconds - capacity) > DEVIATION_PERCENT * capacity:
            result.warnings.append(
                ReportWarning(
                    code="lesson_time_deviation",
                    message=(
                        f"Lesson {number} takes {_minutes(lesson.seconds)}, more than "
                        f"{DEVIATION_PERCENT} % away from {_minutes(capacity)}."
                    ),
                )
            )


def _check_over_capacity(result: SplitResult, capacity: int) -> None:
    """single, syllabus: time over the capacity is only reported."""
    for number, lesson in enumerate(result.lessons, start=1):
        if lesson.seconds > capacity:
            result.warnings.append(
                ReportWarning(
                    code="lesson_over_capacity",
                    message=f"Lesson {number} takes {_minutes(lesson.seconds)}, more than {_minutes(capacity)}.",
                )
            )


def _check_not_empty(result: SplitResult) -> None:
    for number, lesson in enumerate(result.lessons, start=1):
        if not lesson.section_ids:
            raise AssertionError(f"lesson {number} is empty")


def _check_prerequisites(result: SplitResult, sections: list[SplitSection], broken: set[tuple[int, int]]) -> None:
    """A section must not come before its prerequisite (earlier lesson or earlier place)."""
    place: dict[int, tuple[int, int]] = {}
    for lesson_index, lesson in enumerate(result.lessons):
        for position, sid in enumerate(lesson.section_ids):
            place[sid] = (lesson_index, position)
    for section in sections:
        if section.id not in place:
            continue
        for prerequisite in section.prerequisites:
            if prerequisite == section.id or (section.id, prerequisite) in broken or prerequisite not in place:
                continue
            if place[prerequisite] > place[section.id]:
                result.warnings.append(
                    ReportWarning(
                        code="prerequisite_order",
                        message=(
                            f"{section_label(section.id)} (lesson {place[section.id][0] + 1}) comes before its "
                            f"prerequisite {section_label(prerequisite)} (lesson {place[prerequisite][0] + 1})."
                        ),
                    )
                )


def _minutes(seconds: int) -> str:
    return f"{round(seconds / 60, 1):g} min"
