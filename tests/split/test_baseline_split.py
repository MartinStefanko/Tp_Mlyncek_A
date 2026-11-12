"""Split on real data: sections and questions of the recording baseline_v5 (no API call).

Run with -s to see the lessons:
    venv/Scripts/python.exe -m pytest tests/split/test_baseline_split.py -s
"""

from qgen.split import build_split_sections, order_sections, split_lessons
from tests.baseline_data import load_json, require_baseline

from .helpers import lessons, topics


def _baseline_sections():
    require_baseline()
    return build_split_sections(load_json("learning_objects.json"), load_json("questions.json"))


def test_baseline_lessons_45_min():
    sections = _baseline_sections()
    total = sum(s.seconds for s in sections)
    result = split_lessons(sections, lessons(minutes=45))

    minutes = [round(lesson.seconds / 60, 1) for lesson in result.lessons]
    print(
        f"\nbaseline_v5, lessons 45 min: {len(sections)} sections, {total / 60:.1f} min, "
        f"{len(result.lessons)} lessons: {minutes}; warnings: {[w.code for w in result.warnings]}"
    )

    assert len(result.lessons) == max(1, round(total / 2700))
    assert all(lesson.section_ids for lesson in result.lessons)
    placed = [sid for lesson in result.lessons for sid in lesson.section_ids]
    assert sorted(placed) == sorted(s.id for s in sections)
    assert placed == [s.id for s in order_sections(sections).sections]  # contiguous runs of step a
    assert "prerequisite_order" not in [w.code for w in result.warnings]


def test_baseline_without_topics_topics_mode_equals_lessons():
    # No topics in the recording input → one run, split like lessons.
    sections = _baseline_sections()
    by_topics = split_lessons(sections, topics(45))
    by_lessons = split_lessons(sections, lessons(minutes=45))
    assert [x.section_ids for x in by_topics.lessons] == [x.section_ids for x in by_lessons.lessons]
