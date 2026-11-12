"""Split of the sections (tematické celky) into lessons (kap. 8, Algoritmus a–e, g).

Pure functions: no LLM, no embeddings. The document topics and the similarities come ready
from the caller (step 16), as do the title and description of a lesson (step f).

The number of lessons never comes from the time of the questions: the questions are a pool
the teacher picks from in ECEH (eko.pdf: 560 min of questions for one presentation). Time
only balances lessons against each other.

    a. order        topological by prerequisites, ties by pages (qgen.split.order)
    d. split        single: one lesson
                    lessons: lesson_count contiguous runs by dynamic programming, smallest
                             Σ T² (= most even lessons), a cut between two topic runs preferred
                    topics: lesson = run of one topic (qgen.split.topics); a run under 50 % of
                            the median run is merged with its smaller neighbour, a run over
                            3 × the median is split into round(T / median) even parts
                    syllabus: by the plan (qgen.split.syllabus); a plan that cannot be used →
                              topics (fallback)
    e. questions    in the order of the sections; in a section choice → open → no_answer
    g. checks       prerequisites, empty lessons, lessons_unbalanced (lessons, topics)
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from qgen.contract.options import SplitOptions
from qgen.contract.report import ReportWarning
from qgen.split.order import order_sections
from qgen.split.sections import SplitQuestion, SplitSection, section_label
from qgen.split.syllabus import SyllabusPlan, assign_sections, check_plan_ids, plan_from_syllabus
from qgen.split.topics import TopicSimilarity, topic_segments

# Extra cost of a cut inside one topic run, as a share of (mean lesson time)²: a cut between
# two runs wins when it makes the two neighbouring lessons uneven by about ±10 % at most.
TOPIC_CUT_PENALTY = 0.02
# topics: run shorter than this share of the median run is merged, longer than LONG is split.
SHORT_RUN_RATIO = 0.5
LONG_RUN_RATIO = 3.0
# lessons_unbalanced: a lesson more than this share away from the mean lesson time.
UNBALANCED_RATIO = 0.5

QUESTION_TYPE_ORDER = {"choice": 0, "open": 1, "no_answer": 2}

__all__ = ["Lesson", "SplitResult", "order_questions", "split_lessons"]


@dataclass(frozen=True)
class Lesson:
    section_ids: tuple[int, ...]
    seconds: int
    topics: tuple[str, ...] = ()  # names of the topic runs in the lesson (base of the title)
    syllabus_index: Optional[int] = None  # index in SplitResult.plan.lessons (mode syllabus)


@dataclass
class SplitResult:
    lessons: list[Lesson]
    warnings: list[ReportWarning] = field(default_factory=list)
    skipped_section_ids: list[int] = field(default_factory=list)  # left out by the syllabus
    fallback: Optional[str] = None  # "topics" when the syllabus could not be used
    plan: Optional[SyllabusPlan] = None  # the syllabus plan that was applied


@dataclass(frozen=True)
class _Topics:
    names: tuple[str, ...]
    similarity: Optional[TopicSimilarity]


def split_lessons(
    sections: Sequence[SplitSection],
    split: SplitOptions,
    *,
    topics: Sequence[str] = (),
    topic_similarity: Optional[TopicSimilarity] = None,
    syllabus_similarity: Optional[Sequence[Sequence[float]]] = None,
    plan: Optional[SyllabusPlan] = None,
) -> SplitResult:
    """Split the sections into lessons by options.split.

    `topics`: names of the document topics; `topic_similarity[section id][k]`: similarity of
    the section to topic k. `syllabus_similarity[i][j]`: similarity of `sections[i]` to lesson
    j of the syllabus plan (mode syllabus only; needed for sections that no page or id
    assigns). `plan`: the syllabus as a plan (step 16 makes it from syllabus_text); without it
    the list split.syllabus is used. No sections → no lessons.
    """
    sections = list(sections)
    if not sections:
        return SplitResult(lessons=[])
    order = order_sections(sections)
    ordered = order.sections
    warnings = list(order.warnings)
    known = _Topics(tuple(topics), topic_similarity if topics else None)

    if split.mode == "single":
        result = SplitResult(lessons=[_lesson(ordered, known, _run_topics(ordered, known))], warnings=warnings)
    elif split.mode == "lessons":
        result = SplitResult(lessons=_lessons_mode(ordered, split.lesson_count, known, warnings), warnings=warnings)
        _check_balance(result)
    elif split.mode == "topics":
        result = SplitResult(lessons=_topics_mode(ordered, known, warnings), warnings=warnings)
        _check_balance(result)
    else:
        result = _syllabus_mode(sections, ordered, split, syllabus_similarity, plan, known, warnings)

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


def _run_topics(ordered: Sequence[SplitSection], known: _Topics) -> dict[int, int]:
    """Section id → topic index of its run (empty without topics)."""
    found: dict[int, int] = {}
    for topic, members in topic_segments(ordered, len(known.names), known.similarity):
        for section in members:
            found[section.id] = topic
    return found


def _lesson(
    sections: Sequence[SplitSection],
    known: _Topics,
    run_topic: dict[int, int],
    syllabus_index: Optional[int] = None,
) -> Lesson:
    names: list[str] = []
    for section in sections:
        topic = run_topic.get(section.id)
        if topic is not None and known.names[topic] not in names:
            names.append(known.names[topic])
    return Lesson(
        section_ids=tuple(s.id for s in sections),
        seconds=sum(s.seconds for s in sections),
        topics=tuple(names),
        syllabus_index=syllabus_index,
    )


def _lessons_mode(
    ordered: list[SplitSection], lesson_count: int, known: _Topics, warnings: list[ReportWarning]
) -> list[Lesson]:
    count = min(lesson_count, len(ordered))
    if count < lesson_count:
        warnings.append(
            ReportWarning(
                code="lesson_count_reduced",
                message=f"lesson_count {lesson_count} is more than the {len(ordered)} sections; {count} lessons made.",
            )
        )
    run_topic = _run_topics(ordered, known)
    return [_lesson(run, known, run_topic) for run in _split_runs(ordered, count, run_topic)]


def _topics_mode(ordered: list[SplitSection], known: _Topics, warnings: list[ReportWarning]) -> list[Lesson]:
    segments = topic_segments(ordered, len(known.names), known.similarity)
    if not segments:
        warnings.append(
            ReportWarning(code="no_document_topics", message="The document has no topics; one lesson made.")
        )
        return [_lesson(ordered, known, {})]
    run_topic = {s.id: topic for topic, members in segments for s in members}
    runs = _merge_short([members for _, members in segments])
    median = statistics.median(_seconds(run) for run in runs)
    lessons: list[list[SplitSection]] = []
    for run in runs:
        total = _seconds(run)
        if median > 0 and total > LONG_RUN_RATIO * median and len(run) > 1:
            lessons.extend(_split_runs(run, min(len(run), round(total / median)), {}))
        else:
            lessons.append(run)
    return [_lesson(run, known, run_topic) for run in lessons]


def _syllabus_mode(sections, ordered, split, similarity, plan, known, warnings) -> SplitResult:
    if plan is None:
        if split.syllabus is None:
            return _fallback(ordered, known, warnings, "the syllabus text was not turned into a plan")
        plan = plan_from_syllabus(split.syllabus)

    claimed, skipped = check_plan_ids(plan, {s.id for s in ordered}, warnings)
    remaining = [s for s in ordered if s.id not in skipped]
    if not remaining:
        return _fallback(ordered, known, warnings, "the syllabus leaves out every section")
    skipped_ids = [s.id for s in ordered if s.id in skipped]

    if plan.lesson_minutes is not None:
        warnings.append(
            ReportWarning(
                code="syllabus_lesson_minutes_ignored",
                message=(
                    f"Syllabus asks for {plan.lesson_minutes}-minute lessons; the length of a lesson is not used"
                    + (" (lessons split by topics)." if not plan.lessons and plan.lesson_count is None else ".")
                ),
            )
        )
    if not plan.lessons:
        if plan.lesson_count is not None:  # "rozdeľ na 5 hodín": the lessons algorithm
            lessons = _lessons_mode(remaining, plan.lesson_count, known, warnings)
        else:  # only the lesson length: by topics
            lessons = _topics_mode(remaining, known, warnings)
        result = SplitResult(lessons=lessons, warnings=warnings, skipped_section_ids=skipped_ids, plan=plan)
        _warn_skipped(result)
        _check_balance(result)
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
        return _fallback(ordered, known, warnings, assignment.fallback_reason)
    run_topic = _run_topics(ordered, known)
    result = SplitResult(
        lessons=[_lesson(members, known, run_topic, syllabus_index=index) for index, members in assignment.groups],
        warnings=warnings,
        skipped_section_ids=assignment.skipped,
        plan=plan,
    )
    _warn_skipped(result)
    return result


def _fallback(ordered, known, warnings, reason: str) -> SplitResult:
    warnings.append(
        ReportWarning(code="syllabus_fallback", message=f"Syllabus not used ({reason}); lessons split by topics.")
    )
    result = SplitResult(lessons=_topics_mode(ordered, known, warnings), warnings=warnings, fallback="topics")
    _check_balance(result)
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


def _seconds(run: Sequence[SplitSection]) -> int:
    return sum(s.seconds for s in run)


def _split_runs(sections: list[SplitSection], count: int, run_topic: dict[int, int]) -> list[list[SplitSection]]:
    """`count` contiguous non-empty runs with the smallest Σ T² (+ a penalty for a cut inside
    one topic run). For a fixed count Σ T² is smallest when the runs are as even as possible.

    Integer costs and the earliest cut on a tie, so the result is deterministic.
    """
    n = len(sections)
    if not 1 <= count <= n:
        raise ValueError(f"cannot split {n} sections into {count} lessons")
    prefix = [0]
    for section in sections:
        prefix.append(prefix[-1] + section.seconds)
    mean = prefix[-1] / count
    inside = round(TOPIC_CUT_PENALTY * mean * mean)

    def penalty(i: int) -> int:
        before, after = run_topic.get(sections[i - 1].id), run_topic.get(sections[i].id)
        return 0 if before is not None and after is not None and before != after else inside

    best: list[list[Optional[int]]] = [[None] * (n + 1) for _ in range(count + 1)]
    back = [[0] * (n + 1) for _ in range(count + 1)]
    best[0][0] = 0
    for k in range(1, count + 1):
        for j in range(k, n - (count - k) + 1):
            for i in range(k - 1, j):
                previous = best[k - 1][i]
                if previous is None:
                    continue
                cost = previous + (prefix[j] - prefix[i]) ** 2 + (penalty(i) if i > 0 else 0)
                if best[k][j] is None or cost < best[k][j]:
                    best[k][j] = cost
                    back[k][j] = i
    runs = []
    j = n
    for k in range(count, 0, -1):
        i = back[k][j]
        runs.append(sections[i:j])
        j = i
    return runs[::-1]


def _merge_short(runs: list[list[SplitSection]]) -> list[list[SplitSection]]:
    """Merge the shortest run under SHORT_RUN_RATIO × median with its smaller neighbour (the
    left one on a tie), recompute the median, repeat."""
    runs = [list(run) for run in runs]
    while len(runs) > 1:
        times = [_seconds(run) for run in runs]
        median = statistics.median(times)
        short = [i for i, t in enumerate(times) if t < SHORT_RUN_RATIO * median]
        if not short:
            break
        i = min(short, key=lambda x: (times[x], x))
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(runs)]
        j = min(neighbours, key=lambda x: (times[x], x))
        left, right = sorted((i, j))
        runs[left : right + 1] = [runs[left] + runs[right]]
    return runs


# --- checks (g) ------------------------------------------------------------------------------


def _check_balance(result: SplitResult) -> None:
    """lessons, topics: a lesson more than 50 % away from the mean lesson time."""
    if len(result.lessons) < 2:
        return
    mean = sum(lesson.seconds for lesson in result.lessons) / len(result.lessons)
    for number, lesson in enumerate(result.lessons, start=1):
        if abs(lesson.seconds - mean) > UNBALANCED_RATIO * mean:
            result.warnings.append(
                ReportWarning(
                    code="lessons_unbalanced",
                    message=(
                        f"Lesson {number} has {_minutes(lesson.seconds)} of questions, more than "
                        f"{round(UNBALANCED_RATIO * 100)} % away from the mean {_minutes(round(mean))}."
                    ),
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
