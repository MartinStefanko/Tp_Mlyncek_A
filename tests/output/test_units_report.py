"""Mapping of the pipeline result to units + report (qgen.output.build_output, step 14)."""

import json
from pathlib import Path

import jsonschema

from qgen.contract import Report, UnitsResult
from qgen.contract.schemas import build_schema
from qgen.output import check_output
from qgen.output.check import check_units
from tests.output.data import build, question, section

FIXTURES = Path(__file__).resolve().parent.parent / "contract" / "fixtures"


def _assert_valid(out):
    """Model and JSON Schema, for units and report; report matches the units."""
    check_output(out.result, out.report)
    units_data, report_data = out.result.model_dump(mode="json"), out.report.model_dump(mode="json")
    UnitsResult.model_validate(units_data)
    Report.model_validate(report_data)
    jsonschema.Draft202012Validator(build_schema("lesson")).validate(units_data)
    jsonschema.Draft202012Validator(build_schema("report")).validate(report_data)


def test_question_mapping_and_meta():
    sections = [section(1, "Fotosyntéza"), section(2, "Dýchanie", "Vytvoriť", [1])]
    questions = [
        question(1, zdroje=("D1:2", "D2:5", "D1:2"), hints=('<p onclick="x">bio.pdf, strana 2</p>',)),
        question(2, name="Výpočet", answer="Riešenie krok za krokom.", hints=(), bloom="create"),
        question(1, question_type="choice", keywords=("a", "b"), choices=[
            {"text": "Áno", "is_correct": True}, {"text": "Nie", "is_correct": False}]),
    ]
    out = build(sections, questions, names=("bio.pdf", "chem.pdf"), timings={"extract": 1.0})
    _assert_valid(out)

    unit = out.result.units[0]
    # mode single: title and description of the one lesson from the LLM (step 16)
    assert unit.title == "Hodina 1 – názov"
    assert unit.description == "Popis hodiny 1."
    # order of the split: sections in order, in a section choice → open → no_answer
    third, first, second = unit.questions
    # empty name -> section title; the HTML of the hints is cleaned
    assert (first.name, first.description, first.type) == ("Fotosyntéza", "Čo je to?", "open")
    assert (first.time_to_handle, first.time_to_explain) == (300, 60)
    assert [c.model_dump() for c in first.choices] == [{"text": "Toto.", "is_correct": True}]
    assert first.hints[0].model_dump() == {"name": "Zdroj", "url": "<p>bio.pdf, strana 2</p>"}
    assert second.name == "Výpočet" and second.hints == [] and second.choices[0].text.startswith("Riešenie")
    assert third.type == "choice" and [c.is_correct for c in third.choices] == [True, False]

    meta = out.report.question_meta
    assert [(m.unit, m.question, m.section_id, m.bloom_level) for m in meta] == [
        (0, 0, "S-1", "understand"), (0, 1, "S-1", "understand"), (0, 2, "S-2", "create")]
    assert meta[0].keywords == ["a", "b"]
    assert [(s.file, s.pages) for s in meta[1].source] == [("bio.pdf", [2]), ("chem.pdf", [5])]
    assert out.report.sections[1].prerequisites == ["S-1"]
    assert out.report.sections[0].lo_type == "legacy"
    assert out.report.timings.pop("split") >= 0
    assert out.report.timings == {"extract": 1.0}
    assert out.report.warnings == [] and out.dropped == 0


def test_invalid_question_is_left_out_with_a_warning():
    questions = [question(1), question(1, answer=""), question(1)]
    out = build([section(1, "A")], questions)
    _assert_valid(out)
    assert len(out.result.units[0].questions) == 2 and out.dropped == 1
    assert [(w.code, w.message) for w in out.report.warnings] == [
        ("invalid_question", "Question 2 was left out: invalid_question.")]
    assert [m.question for m in out.report.question_meta] == [0, 1]


def test_unknown_bloom_level_warns():
    out = build([section(1, "A", bloom="Zázrak")], [question(1, bloom="")])
    assert out.report.question_meta[0].bloom_level == "understand"
    assert out.report.warnings[0].code == "unknown_bloom_level"


def test_quality_topics_warnings_and_timings_from_reports():
    reports = {
        "lo_faithfulness": {"stats": {"los_evaluated": 3, "average_faithfulness_score": 4.0}},
        "item_faithfulness": {"stats": {"items_evaluated": 2, "average_faithfulness_score": 5.0}},
        "item_answerability": {"stats": {"items_evaluated": 0, "average_answerability_score": 0.0}},
        "topic_coverage": {
            "stats": {"topics_total": 4, "coverage_percent": 75.0},
            "topics": [{"tema": "Rastliny", "best_lo_id": 1, "covered": True},
                       {"tema": "Iné", "best_lo_id": 1, "covered": True}],
        },
        "question_diagnostics": {"warnings": [{"code": "section_without_questions", "message": "Section 2 …"}]},
        "lo_processing_time": {"generation_seconds": 10.0, "evaluation_seconds": 2.5},
        "item_processing_time": {"generation_seconds": 7.0, "evaluation_seconds": 3.0},
    }
    out = build([section(1, "A")], [question(1), question(1, answer="")], reports=reports, timings={"extract": 1.0})
    _assert_valid(out)
    quality = out.report.quality
    assert (quality.section_faithfulness_avg, quality.question_faithfulness_avg) == (0.75, 1.0)
    assert quality.answerability_avg is None
    assert quality.topic_coverage == 0.75
    assert out.report.sections[0].topic == "Rastliny" == out.report.question_meta[0].topic
    # diagnostics of the pipeline first, then the output warnings
    assert [w.code for w in out.report.warnings] == ["section_without_questions", "invalid_question"]
    assert out.report.timings.pop("split") >= 0
    assert out.report.timings == {"extract": 1.0, "learning_objects": 10.0, "lo_evaluation": 2.5,
                                  "questions": 7.0, "q_evaluation": 3.0}


def test_warnings_stay_in_the_order_of_the_questions():
    questions = [question(1, answer=""), question(1, bloom="", hints=("<p></p>",))]
    out = build([section(1, "A", bloom="Zázrak")], questions)
    assert [w.code for w in out.report.warnings] == ["invalid_question", "invalid_hint", "unknown_bloom_level"]


def test_no_questions_gives_one_empty_unit():
    out = build([], [])
    _assert_valid(out)
    assert len(out.result.units) == 1 and out.result.units[0].questions == []
    assert out.result.units[0].description == ""
    assert out.report.quality.section_faithfulness_avg is None


def test_official_units_example_still_passes():
    for name in ("units_official_example.json", "units_example.json"):
        data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        assert check_units(data) == [], name
