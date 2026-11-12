"""Modes single / lessons / topics, checks of step g and the order of the questions (kap. 8).

Since step 16 the number of lessons never comes from the time of the questions: lessons
takes lesson_count, topics the topic runs of the document.
"""

import random

import pytest

from qgen.split import SplitQuestion, order_questions, section_topics, split_lessons, topic_segments

from .helpers import chain, codes, ids, lessons, section, single, syllabus_text, topical, topics

ALL_MODES = [single(), lessons(3), topics()]


# --- edge cases common to all modes -------------------------------------------------------------


@pytest.mark.parametrize("split", [*ALL_MODES, syllabus_text()])
def test_empty_input_gives_no_lessons(split):
    result = split_lessons([], split)
    assert (result.lessons, result.warnings) == ([], [])


@pytest.mark.parametrize("split", ALL_MODES)
def test_one_section_is_one_lesson(split):
    result = split_lessons([section(1, 30)], split, **topical(["A"]))
    assert ids(result) == [[1]]
    assert result.lessons[0].seconds == 1800
    assert result.lessons[0].topics == ("A",)


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
            )
            for i in range(1, n + 1)
        ]
        labels = [rng.choice(["A", "B", "C", None]) for _ in range(n)]
        result = split_lessons(sections, split, **topical(labels))
        assert all(lesson.section_ids for lesson in result.lessons)
        assert sorted(sid for lesson in result.lessons for sid in lesson.section_ids) == list(range(1, n + 1))
        assert "prerequisite_order" not in codes(result)


@pytest.mark.parametrize("split", [*ALL_MODES, lessons(2)])
def test_result_is_deterministic(split):
    sections = chain([12, 7, 20, 5, 30, 18, 9, 22, 14])
    labels = list("AABBBCCDD")
    first = split_lessons(sections, split, **topical(labels))
    again = split_lessons(list(reversed(sections)), split, **topical(labels))
    assert ids(first) == ids(again)
    assert codes(first) == codes(again)


# --- single ---------------------------------------------------------------------------------------


def test_single_puts_everything_in_one_lesson_in_order():
    sections = [section(1, 10, page=9, prerequisites=(2,)), section(2, 10, page=20), section(3, 10, page=1)]
    result = split_lessons(sections, single())
    assert ids(result) == [[3, 2, 1]]
    assert result.warnings == []


def test_single_has_no_time_limit():
    result = split_lessons(chain([300, 300]), single())
    assert ids(result) == [[1, 2]]
    assert result.warnings == []


# --- lessons -------------------------------------------------------------------------------------


def test_lesson_count_sets_the_number_not_the_time():
    # 560 minutes of questions in 3 lessons: still 3 lessons.
    result = split_lessons(chain([20] * 28), lessons(3))
    assert len(result.lessons) == 3
    assert [len(x.section_ids) for x in result.lessons] == [9, 9, 10]


def test_equal_times_split_evenly():
    result = split_lessons(chain([15] * 6), lessons(2))
    assert ids(result) == [[1, 2, 3], [4, 5, 6]]
    assert result.warnings == []


def test_equal_times_tie_is_resolved_the_same_way():
    assert ids(split_lessons(chain([15] * 4), lessons(3))) == [[1], [2], [3, 4]]


def test_lesson_count_larger_than_sections():
    result = split_lessons(chain([10, 10]), lessons(5))
    assert ids(result) == [[1], [2]]
    assert codes(result) == ["lesson_count_reduced"]


def test_big_section_makes_lessons_unbalanced():
    result = split_lessons(chain([10, 60, 10, 10]), lessons(2))
    assert ids(result) == [[1, 2], [3, 4]]  # 70 / 20 min, the most even split
    assert codes(result) == ["lessons_unbalanced", "lessons_unbalanced"]


def test_cut_prefers_topic_boundary_when_close():
    # 30 × 3 min in 2 lessons: the even cut is after 15 sections; topic A ends after 16
    # (48 / 42 min, ±7 %), so the cut moves to the boundary.
    result = split_lessons(chain([3] * 30), lessons(2), **topical(["A"] * 16 + ["B"] * 14))
    assert [len(lesson.section_ids) for lesson in result.lessons] == [16, 14]
    assert [lesson.topics for lesson in result.lessons] == [("A",), ("B",)]


def test_cut_ignores_topic_boundary_when_far():
    result = split_lessons(chain([3] * 30), lessons(2), **topical(["A"] * 20 + ["B"] * 10))
    assert [len(lesson.section_ids) for lesson in result.lessons] == [15, 15]
    assert [lesson.topics for lesson in result.lessons] == [("A",), ("A", "B")]


def test_lessons_follow_prerequisites():
    sections = [section(1, 30, page=1, prerequisites=(3,)), section(2, 30, page=2), section(3, 30, page=50)]
    result = split_lessons(sections, lessons(3))
    assert ids(result) == [[2], [3], [1]]


def test_cycle_is_reported_and_not_a_prerequisite_violation():
    sections = [section(1, 40, page=1, prerequisites=(2,)), section(2, 40, page=2, prerequisites=(1,))]
    result = split_lessons(sections, lessons(2))
    assert ids(result) == [[1], [2]]
    assert codes(result) == ["prerequisite_cycle"]


# --- topics ---------------------------------------------------------------------------------------


def test_topics_without_topics_is_one_lesson():
    result = split_lessons(chain([20, 20, 20]), topics())
    assert ids(result) == [[1, 2, 3]]
    assert codes(result) == ["no_document_topics"]


def test_topics_one_lesson_per_topic():
    result = split_lessons(chain([20, 20, 20, 20]), topics(), **topical(list("AABB")))
    assert ids(result) == [[1, 2], [3, 4]]
    assert [lesson.topics for lesson in result.lessons] == [("A",), ("B",)]
    assert result.warnings == []


def test_topics_short_run_merged_with_smaller_neighbour():
    # Runs 10 / 40 / 40 min, median 40: A (< 20) joins B.
    result = split_lessons(chain([10, 40, 40]), topics(), **topical(list("ABC")))
    assert ids(result) == [[1, 2], [3]]
    assert result.lessons[0].topics == ("A", "B")


def test_topics_short_run_picks_the_smaller_side():
    # Runs 40 / 5 / 25 / 30 min, median 27.5: B (< 13.75) joins C (25 < 40).
    result = split_lessons(chain([40, 5, 25, 30]), topics(), **topical(list("ABCD")))
    assert ids(result) == [[1], [2, 3], [4]]


def test_topics_long_run_is_split_evenly():
    # Runs 80 / 20 / 20 min, median 20: A is 4 × the median → 4 parts of 20 min.
    result = split_lessons(chain([10] * 8 + [20, 20]), topics(), **topical(list("AAAAAAAABC")))
    assert ids(result) == [[1, 2], [3, 4], [5, 6], [7, 8], [9], [10]]
    assert [x.topics for x in result.lessons][:4] == [("A",)] * 4


def test_topics_run_under_three_medians_stays():
    result = split_lessons(chain([25, 25, 20, 20]), topics(), **topical(list("AABC")))
    assert ids(result) == [[1, 2], [3], [4]]  # 50 min < 3 × 20
    assert codes(result) == ["lessons_unbalanced"]  # 50 against the mean 30


def test_topics_same_topic_in_two_runs():
    result = split_lessons(chain([40, 40, 40]), topics(), **topical(list("ABA")))
    assert ids(result) == [[1], [2], [3]]


def test_topics_section_without_similarity_stays_in_its_run():
    result = split_lessons(chain([20, 20, 20, 20]), topics(), **topical(["A", None, "A", "B"]))
    assert ids(result) == [[1, 2, 3], [4]]


def test_topic_runs_ignore_a_single_weak_change():
    sections = chain([10, 10, 10, 10])
    similarity = {1: [0.9, 0.1], 2: [0.9, 0.1], 3: [0.50, 0.55], 4: [0.9, 0.1]}
    runs = topic_segments(sections, 2, similarity)
    assert [(topic, [s.id for s in members]) for topic, members in runs] == [(0, [1, 2, 3, 4])]
    assert section_topics(sections, 2, similarity) == {1: 0, 2: 0, 3: 1, 4: 0}


def test_topic_runs_follow_a_clear_change():
    sections = chain([10, 10, 10, 10])
    similarity = {1: [0.9, 0.1], 2: [0.9, 0.1], 3: [0.2, 0.9], 4: [0.1, 0.9]}
    runs = topic_segments(sections, 2, similarity)
    assert [(topic, [s.id for s in members]) for topic, members in runs] == [(0, [1, 2]), (1, [3, 4])]


def test_wrong_similarity_row_is_a_caller_error():
    with pytest.raises(ValueError):
        split_lessons(chain([10]), topics(), topics=["A", "B"], topic_similarity={1: [0.5]})


# --- order of the questions (e) -------------------------------------------------------------------


def test_questions_follow_sections_then_type():
    result = split_lessons(chain([20, 20, 20]), lessons(2))
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
