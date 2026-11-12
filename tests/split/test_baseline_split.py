"""Split on real data: eko.pdf from the recording baseline_v6 (pipeline replayed, no API call).

The topics and the section vectors come from the replayed topic coverage
(LOPipelineResult.embeddings). The expected values come from the recordings, never from
numbers in the code: the number of topics from baseline_v6/metrics.json, the lessons from
the split recordings baseline_v6_topics / _lessons3 (metrics.json "split":
sections_per_lesson). Without a split recording only the properties are checked (no empty
lesson, every section exactly once in the order of step a, no prerequisite_order, lessons
= lesson_count). Run with -s to see the lessons:
    venv/Scripts/python.exe -m pytest tests/split/test_baseline_split.py -s
"""

import pytest

from qgen.cli import normalize_lo_for_export
from qgen.contract.options import LessonsSplit, TopicsSplit
from qgen.output import build_output, check_output
from qgen.output.split import _topic_similarity
from qgen.pipeline import run_pipeline
from qgen.split import build_split_sections, order_sections, split_lessons
from tests.baseline_data import (
    baseline_settings,
    load_json,
    recorded_sections_per_lesson,
    recorded_split,
    replay_client,
    require_baseline,
)
from tests.fakes import TitlesLLM


@pytest.fixture(scope="module")
def eko():
    require_baseline()
    replay = replay_client()
    result = run_pipeline(load_json("segments.json"), llm=replay, embedder=replay, settings=baseline_settings())
    sections = build_split_sections(
        [normalize_lo_for_export(lo) for lo in result.learning_objects], result.questions
    )
    topics, similarity = _topic_similarity(result.lo_result.embeddings)
    return result, sections, topics, similarity


def _show(title, result, topics_used=True):
    print(f"\n{title}: {len(result.lessons)} lessons")
    for number, lesson in enumerate(result.lessons, start=1):
        first, last = lesson.section_ids[0], lesson.section_ids[-1]
        topic = f" – {', '.join(lesson.topics)}" if topics_used and lesson.topics else ""
        print(f"  {number:>2}. sections {first}–{last} ({len(lesson.section_ids)}), "
              f"{lesson.seconds / 60:.1f} min{topic}")
    print(f"  warnings: {[w.code for w in result.warnings]}")


def _check(result, sections, topics=None):
    """Properties of every split: no empty lesson, every section exactly once, in the
    topological order of step a (prerequisites first, lessons are contiguous runs)."""
    assert all(lesson.section_ids for lesson in result.lessons)
    placed = [sid for lesson in result.lessons for sid in lesson.section_ids]
    assert sorted(placed) == sorted(s.id for s in sections) and len(placed) == len(set(placed))
    assert placed == [s.id for s in order_sections(sections).sections]  # contiguous runs of step a
    assert "prerequisite_order" not in [w.code for w in result.warnings]
    if topics is not None:
        assert all(lesson.topics and set(lesson.topics) <= set(topics) for lesson in result.lessons)


def _check_against_recording(result, name):
    """Lessons of `result` = sections per lesson of the split recording `name`; the split
    warnings of `result` were also recorded. Nothing to compare without the recording."""
    recorded = recorded_split(name)
    if recorded is None:
        return
    assert [list(lesson.section_ids) for lesson in result.lessons] == recorded_sections_per_lesson(recorded)
    assert len(result.lessons) == recorded["lessons"]
    assert {w.code for w in result.warnings} <= set(recorded["warnings"])


def test_baseline_vectors_cover_every_section(eko):
    result, sections, topics, similarity = eko
    assert len(topics) == load_json("metrics.json")["topic_coverage"]["topics_total"]
    assert set(similarity) == {lo["id"] for lo in result.learning_objects}


def test_baseline_topics(eko):
    _result, sections, topics, similarity = eko
    split = split_lessons(sections, TopicsSplit(mode="topics"), topics=topics, topic_similarity=similarity)
    _show("baseline_v6, topics", split)
    _check(split, sections, topics)
    assert 1 <= len(split.lessons) <= len(sections)
    _check_against_recording(split, "baseline_v6_topics")


def test_baseline_lessons_3(eko):
    _result, sections, topics, similarity = eko
    split = split_lessons(
        sections, LessonsSplit(mode="lessons", lesson_count=3), topics=topics, topic_similarity=similarity
    )
    _show("baseline_v6, lessons, lesson_count 3", split)
    _check(split, sections)
    assert len(split.lessons) == 3  # = lesson_count (eko.pdf has far more sections)
    assert "lesson_count_reduced" not in [w.code for w in split.warnings]
    _check_against_recording(split, "baseline_v6_lessons3")


def test_baseline_output_by_topics(eko):
    """The whole output step on eko.pdf (lesson titles from a fake): one unit per lesson of
    the topics split (number from baseline_v6_topics when recorded), every question of the
    pipeline in exactly one unit, no empty unit, check_output passes."""
    result, sections, topics, similarity = eko
    output = build_output(
        result,
        input_names=["eko.pdf"],
        timings={},
        settings=baseline_settings(),
        split=TopicsSplit(mode="topics"),
        llm=TitlesLLM(None),
        embedder=None,
    )
    check_output(output.result, output.report)
    units = output.result.units
    expected = split_lessons(sections, TopicsSplit(mode="topics"), topics=topics, topic_similarity=similarity)
    assert len(units) == len(expected.lessons)
    recorded = recorded_split("baseline_v6_topics")
    if recorded is not None:
        assert len(units) == recorded["lessons"]
        assert [len(u.questions) for u in units] == recorded["questions_per_lesson"]
    assert all(u.questions for u in units)
    assert sum(len(u.questions) for u in units) == len(result.questions) == len(output.report.question_meta)
    assert {m.unit for m in output.report.question_meta} == set(range(len(units)))
