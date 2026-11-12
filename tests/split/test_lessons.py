"""Modes single / lessons / topics, checks of step g and the order of the questions (kap. 8)."""

import random

import pytest

from qgen.split import SplitQuestion, order_questions, split_lessons

from .helpers import chain, codes, ids, lessons, section, single, syllabus_text, topics

ALL_MODES = [single(), lessons(minutes=45), lessons(count=3), topics()]


# --- edge cases common to all modes -------------------------------------------------------------


@pytest.mark.parametrize("split", [*ALL_MODES, syllabus_text()])
def test_empty_input_gives_no_lessons(split):
    result = split_lessons([], split)
    assert (result.lessons, result.warnings) == ([], [])


@pytest.mark.parametrize("split", ALL_MODES)
def test_one_section_is_one_lesson(split):
    result = split_lessons([section(1, 30)], split)
    assert ids(result) == [[1]]
    assert result.lessons[0].seconds == 1800


@pytest.mark.parametrize("split", ALL_MODES)
def test_no_lesson_is_empty_and_every_section_is_used_once(split):
    rng = random.Random(15)
    for _ in range(30):
        n = rng.randint(1, 25)
        sections = [
            section(
                i,
                rng.choice([0, 3, 8, 15, 25, 40, 60]),
                prerequisites=tuple(p for p in range(1, n + 1) if p != i and rng.random() < 0.08),
                page=rng.randint(1, 40),
                topic=rng.choice(["A", "B", "C", None]),
            )
            for i in range(1, n + 1)
        ]
        result = split_lessons(sections, split)
        assert all(lesson.section_ids for lesson in result.lessons)
        assert sorted(sid for lesson in result.lessons for sid in lesson.section_ids) == list(range(1, n + 1))
        assert "prerequisite_order" not in codes(result)


@pytest.mark.parametrize("split", [*ALL_MODES, lessons(count=2)])
def test_result_is_deterministic(split):
    sections = chain([12, 7, 20, 5, 30, 18, 9, 22, 14], topics=list("AABBBCCDD"))
    first = split_lessons(sections, split)
    again = split_lessons(list(reversed(sections)), split)
    assert ids(first) == ids(again)
    assert codes(first) == codes(again)


# --- single ---------------------------------------------------------------------------------------


def test_single_puts_everything_in_one_lesson_in_order():
    sections = [section(1, 10, page=9, prerequisites=(2,)), section(2, 10, page=20), section(3, 10, page=1)]
    result = split_lessons(sections, single())
    assert ids(result) == [[3, 2, 1]]
    assert result.warnings == []


def test_single_over_capacity_is_only_reported():
    result = split_lessons(chain([30, 30]), single(45))
    assert ids(result) == [[1, 2]]
    assert codes(result) == ["lesson_over_capacity"]


# --- lessons -------------------------------------------------------------------------------------


def test_lesson_count_from_round_not_ceil():
    assert ids(split_lessons(chain([25, 25]), lessons(minutes=45))) == [[1, 2]]  # 50 min → 1
    assert len(split_lessons(chain([35, 35]), lessons(minutes=45)).lessons) == 2  # 70 min → 2


def test_equal_times_split_evenly():
    result = split_lessons(chain([15] * 6), lessons(minutes=45))
    assert ids(result) == [[1, 2, 3], [4, 5, 6]]
    assert [lesson.seconds for lesson in result.lessons] == [2700, 2700]
    assert result.warnings == []


def test_equal_times_tie_is_resolved_the_same_way():
    assert ids(split_lessons(chain([15] * 4), lessons(count=3))) == [[1], [2], [3, 4]]


def test_given_lesson_count_balances_lengths():
    result = split_lessons(chain([10] * 8), lessons(count=4, minutes=45))
    assert ids(result) == [[1, 2], [3, 4], [5, 6], [7, 8]]
    assert codes(result) == ["lesson_time_deviation"] * 4  # 20 min against 45 min


def test_lesson_count_larger_than_sections():
    result = split_lessons(chain([10, 10]), lessons(count=5))
    assert ids(result) == [[1], [2]]
    assert codes(result)[0] == "lesson_count_reduced"


def test_section_longer_than_lesson_is_own_lesson():
    result = split_lessons(chain([10, 60, 10, 10]), lessons(minutes=45))
    assert ids(result) == [[1], [2], [3, 4]]
    assert codes(result) == ["section_over_capacity", "lesson_time_deviation", "lesson_time_deviation"]


def test_lesson_count_raised_for_long_section():
    result = split_lessons(chain([10, 60, 10]), lessons(count=2))
    assert ids(result) == [[1], [2], [3]]
    assert "lesson_count_increased" in codes(result)


def test_cut_prefers_topic_boundary_when_close():
    # 30 × 3 min, 45-min lessons: the ideal cut is after 15 sections; topic A ends after 16
    # (48 / 42 min, ±7 %), so the cut moves to the boundary.
    sections = chain([3] * 30, topics=["A"] * 16 + ["B"] * 14)
    result = split_lessons(sections, lessons(minutes=45))
    assert [len(lesson.section_ids) for lesson in result.lessons] == [16, 14]
    assert [lesson.topics for lesson in result.lessons] == [("A",), ("B",)]


def test_cut_ignores_topic_boundary_when_far():
    # Topic A ends after 20 sections (60 / 30 min): balancing wins.
    sections = chain([3] * 30, topics=["A"] * 20 + ["B"] * 10)
    result = split_lessons(sections, lessons(minutes=45))
    assert [len(lesson.section_ids) for lesson in result.lessons] == [15, 15]


def test_lessons_follow_prerequisites():
    sections = [section(1, 30, page=1, prerequisites=(3,)), section(2, 30, page=2), section(3, 30, page=50)]
    result = split_lessons(sections, lessons(count=3))
    assert ids(result) == [[2], [3], [1]]


def test_cycle_is_reported_and_not_a_prerequisite_violation():
    sections = [section(1, 40, page=1, prerequisites=(2,)), section(2, 40, page=2, prerequisites=(1,))]
    result = split_lessons(sections, lessons(count=2))
    assert ids(result) == [[1], [2]]
    assert codes(result) == ["prerequisite_cycle"]


# --- topics ---------------------------------------------------------------------------------------


def test_topics_one_lesson_per_topic():
    result = split_lessons(chain([20, 20, 20, 20], topics=list("AABB")), topics())
    assert ids(result) == [[1, 2], [3, 4]]
    assert [lesson.topics for lesson in result.lessons] == [("A",), ("B",)]
    assert result.warnings == []


def test_topics_long_topic_is_split():
    result = split_lessons(chain([15] * 6 + [40], topics=list("AAAAAAB")), topics())
    assert ids(result) == [[1, 2, 3], [4, 5, 6], [7]]


def test_topics_short_neighbours_are_merged():
    result = split_lessons(chain([10, 15, 40], topics=list("ABC")), topics())
    assert ids(result) == [[1, 2], [3]]
    assert result.lessons[0].topics == ("A", "B")
    assert codes(result) == ["lesson_time_deviation"]  # 25 min, nothing more to merge


def test_topics_same_topic_in_two_runs():
    # Prerequisites put topic B between two parts of topic A.
    result = split_lessons(chain([40, 40, 40], topics=list("ABA")), topics())
    assert ids(result) == [[1], [2], [3]]


def test_topics_section_without_topic_joins_neighbour():
    result = split_lessons(chain([20, 20, 20, 20], topics=[None, "A", None, "B"]), topics())
    assert ids(result) == [[1, 2, 3], [4]]


def test_topics_long_section_not_merged():
    result = split_lessons(chain([50, 5], topics=list("AB")), topics())
    assert ids(result) == [[1], [2]]
    assert codes(result) == ["section_over_capacity", "lesson_time_deviation"]


# --- order of the questions (e) -------------------------------------------------------------------


def test_questions_follow_sections_then_type():
    result = split_lessons(chain([20, 20, 20]), lessons(count=2))
    questions = [
        SplitQuestion("q1", 2, "no_answer"),
        SplitQuestion("q2", 1, "open"),
        SplitQuestion("q3", 2, "choice"),
        SplitQuestion("q4", 1, "choice"),
        SplitQuestion("q5", 1, "open"),
        SplitQuestion("q6", 3, "open"),
        SplitQuestion("q7", 3, "choice"),
    ]
    assert ids(result) == [[1], [2, 3]]  # tie 20 + 40 / 40 + 20 → the earlier cut
    assert order_questions(result, questions) == [["q4", "q2", "q5"], ["q3", "q1", "q7", "q6"]]


def test_question_of_unknown_section_is_an_error():
    result = split_lessons(chain([20]), single())
    with pytest.raises(ValueError):
        order_questions(result, [SplitQuestion(0, 9, "open")])
