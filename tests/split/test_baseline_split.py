"""Split on real data: eko.pdf from the recording baseline_v5 (pipeline replayed, no API call).

The topics and the section vectors come from the replayed topic coverage
(LOPipelineResult.embeddings). Run with -s to see the lessons:
    venv/Scripts/python.exe -m pytest tests/split/test_baseline_split.py -s
"""

import pytest

from qgen.cli import normalize_lo_for_export
from qgen.contract.options import LessonsSplit, TopicsSplit
from qgen.output import build_output, check_output
from qgen.output.split import _topic_similarity
from qgen.pipeline import run_pipeline
from qgen.split import build_split_sections, order_sections, split_lessons
from tests.baseline_data import baseline_settings, load_json, replay_client, require_baseline
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


def _check(result, sections):
    assert all(lesson.section_ids for lesson in result.lessons)
    placed = [sid for lesson in result.lessons for sid in lesson.section_ids]
    assert placed == [s.id for s in order_sections(sections).sections]  # contiguous runs of step a
    assert "prerequisite_order" not in [w.code for w in result.warnings]


def test_baseline_vectors_cover_every_section(eko):
    result, sections, topics, similarity = eko
    assert len(topics) == load_json("metrics.json")["topic_coverage"]["topics_total"] == 8
    assert set(similarity) == {lo["id"] for lo in result.learning_objects}


def test_baseline_topics(eko):
    _result, sections, topics, similarity = eko
    split = split_lessons(sections, TopicsSplit(mode="topics"), topics=topics, topic_similarity=similarity)
    _show("baseline_v5, topics", split)
    _check(split, sections)
    assert len(split.lessons) == 8
    assert split.lessons[0].topics == ("Časová hodnota peňazí",)
    assert [x.topics for x in split.lessons[-3:]] == [("Riziko a diverzifikácia portfólia",)] * 3


def test_baseline_lessons_3(eko):
    _result, sections, topics, similarity = eko
    split = split_lessons(
        sections, LessonsSplit(mode="lessons", lesson_count=3), topics=topics, topic_similarity=similarity
    )
    _show("baseline_v5, lessons, lesson_count 3", split)
    _check(split, sections)
    assert [(x.section_ids[0], x.section_ids[-1]) for x in split.lessons] == [(1, 15), (16, 32), (33, 42)]
    assert split.warnings == []


def test_baseline_output_by_topics(eko):
    """The whole output step on eko.pdf (lesson titles from a fake): 8 units, every question
    of the pipeline in exactly one unit, check_output passes."""
    result, *_ = eko
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
    assert len(output.result.units) == 8
    assert sum(len(u.questions) for u in output.result.units) == len(result.questions)
    assert {m.unit for m in output.report.question_meta} == set(range(8))
