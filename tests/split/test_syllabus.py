"""Mode syllabus: list from options, structured plan (from the free text in step 16), checks
of a bad plan and the fallback to topics. A bad syllabus never raises."""

import pytest
from pydantic import ValidationError

from qgen.contract.options import SyllabusLesson
from qgen.split import PlannedLesson, SplitQuestion, SyllabusPlan, order_questions, plan_from_syllabus, split_lessons

from .helpers import chain, codes, ids, section, syllabus, syllabus_text, topical


def lesson(title, pages=None, description=None):
    return SyllabusLesson(title=title, pages=pages, description=description)


def planned(title, pages=None, section_ids=()):
    return PlannedLesson(title=title, pages=pages, section_ids=list(section_ids))


SIX = chain([10] * 6)  # pages 1–6, prerequisite chain


# --- list from options -----------------------------------------------------------------------------


def test_assignment_by_pages():
    result = split_lessons(SIX, syllabus(lesson("A", "D1:1-3"), lesson("B", "D1:4-6")))
    assert ids(result) == [[1, 2, 3], [4, 5, 6]]
    assert [x.syllabus_index for x in result.lessons] == [0, 1]
    assert result.warnings == []
    assert result.fallback is None


def test_without_pages_only_similarity():
    sim = [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.1, 0.9], [0.5, 0.5]]
    result = split_lessons(SIX, syllabus(lesson("A"), lesson("B")), syllabus_similarity=sim)
    assert ids(result) == [[1, 3, 6], [2, 4, 5]]  # 6: tie → earlier lesson
    # Chain 1 → 2 → … : 3 in lesson 1 needs 2 from lesson 2.
    assert set(codes(result)) == {"prerequisite_order"}


def test_page_tie_decided_by_similarity():
    sections = [section(1, 10, page=5)]
    split = syllabus(lesson("A", "D1:1-5"), lesson("B", "D1:5-9"))
    assert ids(split_lessons(sections, split, syllabus_similarity=[[0.2, 0.7]])) == [[1]]
    assert split_lessons(sections, split, syllabus_similarity=[[0.2, 0.7]]).lessons[0].syllabus_index == 1
    assert split_lessons(sections, split).lessons[0].syllabus_index == 0  # no matrix → earlier


def test_section_outside_pages_uses_similarity():
    sections = [section(1, 10, page=2), section(2, 10, page=40)]
    split = syllabus(lesson("A", "D1:1-5"), lesson("B", "D1:6-9"))
    result = split_lessons(sections, split, syllabus_similarity=[[0.5, 0.1], [0.1, 0.6]])
    assert ids(result) == [[1], [2]]


def test_other_document_does_not_match():
    sections = [section(1, 10, pages=(("D2", 3),))]
    split = syllabus(lesson("A", "D1:1-5"), lesson("B", "D2:1-5"))
    assert split_lessons(sections, split).lessons[0].syllabus_index == 1


def test_missing_or_wrong_similarity_is_a_caller_error():
    split = syllabus(lesson("A"), lesson("B"))
    with pytest.raises(ValueError):
        split_lessons(SIX, split)
    with pytest.raises(ValueError):
        split_lessons(SIX, split, syllabus_similarity=[[0.1, 0.2]] * 5)
    with pytest.raises(ValueError):
        split_lessons(SIX, split, syllabus_similarity=[[0.1]] * 6)


def test_empty_syllabus_lesson_is_left_out():
    result = split_lessons(SIX, syllabus(lesson("A", "D1:1-6"), lesson("B", "D1:50-60"), lesson("C", "D1:6")))
    assert ids(result) == [[1, 2, 3, 4, 5, 6]]  # 6: tie (D1:1-6, D1:6) → earlier lesson
    assert [x.syllabus_index for x in result.lessons] == [0]
    assert codes(result) == ["empty_lesson", "empty_lesson"]


def test_long_lesson_is_not_reported():
    result = split_lessons(chain([300, 300]), syllabus(lesson("A", "D1:1-2")))
    assert result.warnings == []


def test_prerequisite_conflict_keeps_teachers_order():
    sections = [section(1, 10, page=1, prerequisites=(2,)), section(2, 10, page=8)]
    result = split_lessons(sections, syllabus(lesson("A", "D1:1-3"), lesson("B", "D1:5-9")))
    assert ids(result) == [[1], [2]]
    assert codes(result) == ["prerequisite_order"]


# --- plan with section ids -------------------------------------------------------------------------


def test_plan_with_section_ids():
    plan = SyllabusPlan(lessons=[planned("A", section_ids=[4, 5, 6]), planned("B", section_ids=[1, 2, 3])])
    result = split_lessons(SIX, syllabus_text(), plan=plan)
    assert ids(result) == [[4, 5, 6], [1, 2, 3]]
    assert codes(result) == ["prerequisite_order"]  # 4 needs 3 from lesson 2


def test_plan_with_skipped_sections():
    plan = SyllabusPlan(
        lessons=[planned("A", section_ids=[1, 2]), planned("B", section_ids=[3, 4])], skipped_section_ids=[5, 6]
    )
    result = split_lessons(SIX, syllabus_text(), plan=plan)
    assert ids(result) == [[1, 2], [3, 4]]
    assert result.skipped_section_ids == [5, 6]
    assert codes(result) == ["syllabus_skipped_sections"]
    assert "S-5, S-6" in result.warnings[0].message


def test_skipped_questions_are_not_in_lessons():
    plan = SyllabusPlan(lessons=[planned("A", section_ids=[1])], skipped_section_ids=[2])
    result = split_lessons(chain([10, 10]), syllabus_text(), plan=plan)
    questions = [SplitQuestion("a", 2, "choice"), SplitQuestion("b", 1, "open")]
    assert order_questions(result, questions) == [["b"]]


def test_section_in_two_lessons_stays_in_first():
    plan = SyllabusPlan(lessons=[planned("A", section_ids=[1, 2, 3]), planned("B", section_ids=[3, 4, 5, 6])])
    result = split_lessons(SIX, syllabus_text(), plan=plan)
    assert ids(result) == [[1, 2, 3], [4, 5, 6]]
    assert codes(result) == ["syllabus_duplicate_section"]


def test_section_in_lesson_and_skipped_stays_in_lesson():
    plan = SyllabusPlan(lessons=[planned("A", section_ids=[1, 2, 3, 4, 5, 6])], skipped_section_ids=[6])
    result = split_lessons(SIX, syllabus_text(), plan=plan)
    assert ids(result) == [[1, 2, 3, 4, 5, 6]]
    assert result.skipped_section_ids == []
    assert "syllabus_duplicate_section" in codes(result)


def test_unknown_ids_are_ignored():
    plan = SyllabusPlan(
        lessons=[planned("A", section_ids=[1, 2, 3, 99]), planned("B", section_ids=[4, 5, 6])],
        skipped_section_ids=[77],
    )
    result = split_lessons(SIX, syllabus_text(), plan=plan)
    assert ids(result) == [[1, 2, 3], [4, 5, 6]]
    assert codes(result) == ["syllabus_unknown_section", "syllabus_unknown_section"]


def test_unmentioned_sections_by_pages_or_similarity():
    # Lesson A by ids, lesson B by pages; 3 is not mentioned and its page is in no range.
    plan = SyllabusPlan(lessons=[planned("A", section_ids=[1, 2]), planned("B", pages="D1:4-6")])
    sim = [[0.0, 0.0], [0.0, 0.0], [0.9, 0.1], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    result = split_lessons(SIX, syllabus_text(), syllabus_similarity=sim, plan=plan)
    assert ids(result) == [[1, 2, 3], [4, 5, 6]]
    assert result.warnings == []


def test_lesson_by_ids_does_not_take_sections_by_pages():
    plan = SyllabusPlan(lessons=[planned("A", section_ids=[1]), planned("B", pages="D1:1-6")])
    result = split_lessons(SIX, syllabus_text(), plan=plan)
    assert ids(result) == [[1], [2, 3, 4, 5, 6]]


def test_everything_skipped_falls_back_to_topics():
    plan = SyllabusPlan(lessons=[planned("A", section_ids=[])], skipped_section_ids=[1, 2, 3, 4, 5, 6])
    result = split_lessons(SIX, syllabus_text(), plan=plan)
    assert result.fallback == "topics"
    assert result.skipped_section_ids == []
    assert ids(result) == [[1, 2, 3, 4, 5, 6]]  # no topics → one lesson
    assert codes(result) == ["syllabus_fallback", "no_document_topics"]


def test_text_without_plan_falls_back_to_topics():
    result = split_lessons(chain([20, 20, 20, 20]), syllabus_text(), **topical(list("AABB")))
    assert result.fallback == "topics"
    assert result.plan is None
    assert ids(result) == [[1, 2], [3, 4]]
    assert codes(result) == ["syllabus_fallback"]


# --- plan with only lesson_count / lesson_minutes ----------------------------------------------------


def test_plan_with_lesson_count_runs_lessons():
    plan = SyllabusPlan(lesson_count=2, skipped_section_ids=[6])
    result = split_lessons(chain([15] * 6), syllabus_text("2 hodiny, posledný celok vynechaj"), plan=plan)
    assert ids(result) == [[1, 2], [3, 4, 5]]  # tie 30 + 45 / 45 + 30 → the earlier cut
    assert result.skipped_section_ids == [6]
    assert all(x.syllabus_index is None for x in result.lessons)


def test_plan_with_only_lesson_minutes_splits_by_topics():
    plan = SyllabusPlan(lesson_minutes=90, skipped_section_ids=[4])
    result = split_lessons(chain([20] * 4), syllabus_text("hodiny po 90 minút"), plan=plan, **topical(list("AABB")))
    assert ids(result) == [[1, 2], [3]]
    assert result.skipped_section_ids == [4]
    assert codes(result) == ["syllabus_lesson_minutes_ignored", "syllabus_skipped_sections"]
    assert "(lessons split by topics)" in result.warnings[0].message


def test_plan_lessons_with_minutes_ignores_minutes():
    plan = SyllabusPlan(lessons=[planned("A", section_ids=[1, 2, 3, 4, 5, 6])], lesson_minutes=90)
    result = split_lessons(SIX, syllabus_text(), plan=plan)
    assert ids(result) == [[1, 2, 3, 4, 5, 6]]
    assert codes(result) == ["syllabus_lesson_minutes_ignored"]
    assert result.plan is plan


# --- plan models ---------------------------------------------------------------------------------------


def test_plan_with_lessons_and_lesson_count_is_invalid():
    with pytest.raises(ValidationError):
        SyllabusPlan(lessons=[planned("A")], lesson_count=3)


def test_plan_without_lessons_or_count_is_invalid():
    with pytest.raises(ValidationError):
        SyllabusPlan(skipped_section_ids=[1])


def test_plan_lessons_with_minutes_is_valid():
    assert SyllabusPlan(lessons=[planned("A")], lesson_minutes=90).lesson_minutes == 90


@pytest.mark.parametrize(
    "fields",
    [
        {"title": "A", "pages": "D1:1-3", "section_ids": [1]},
        {"title": ""},
        {"title": "A", "pages": "D1:5-2"},
        {"title": "A", "unknown": 1},
    ],
)
def test_invalid_planned_lesson(fields):
    with pytest.raises(ValidationError):
        PlannedLesson(**fields)


def test_plan_from_syllabus_keeps_list():
    plan = plan_from_syllabus([lesson("A", "D1:1-3", "úvod"), lesson("B")])
    assert [(x.title, x.description, x.pages, x.section_ids) for x in plan.lessons] == [
        ("A", "úvod", "D1:1-3", []),
        ("B", None, None, []),
    ]
    assert plan.skipped_section_ids == []
