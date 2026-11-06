"""Temporary mapping of the pipeline output to units + report (service.jobs.legacy_output)."""

from datetime import datetime, timezone

from qgen.contract import JobOptions, OptionDefaults, resolve_options
from qgen.lo.pipeline import LOPipelineResult
from qgen.pipeline import PipelineResult
from service.jobs.legacy_output import build_output, clean_hint_html, cut_at_word
from service.jobs.models import InputFile, JobRecord
from service import worker as worker_main

NOW = datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc)


def _record(*names: str) -> JobRecord:
    return JobRecord(
        job_id="j1",
        owner="k",
        status="running",
        options=resolve_options(JobOptions(), OptionDefaults()),
        input_files=[
            InputFile(name=name, stored_as=f"D{i}.pdf", size=1, sha256="0" * 64, pages=3)
            for i, name in enumerate(names, start=1)
        ],
        created_at=NOW,
        updated_at=NOW,
    )


def _section(lo_id, name, bloom="Pochopiť", prerequisites=()):
    return {"id": lo_id, "vzdelávací_objekt": name, "bloom_level": bloom, "odporúčané_zadania": "Vysvetli.",
            "citovane_zdroje": [f"D1:{lo_id}"], "prerekvizity": list(prerequisites)}


def _question(lo_id, description="Čo je to?", answer="Toto.", hints=("<p>Pozri.</p>",), zdroje=("D1:1",),
              question_type="open", name="", bloom="understand", keywords=("pojem",), choices=None):
    """A question in the internal format of the pipeline (qgen.questions.types)."""
    if choices is None:
        choices = [{"text": answer, "is_correct": True}] if question_type == "open" else []
    names = ("Ľahká nápoveda", "Ťažká nápoveda")
    return {"id": lo_id, "lo_id": lo_id, "name": name, "description": description, "type": question_type,
            "time_to_handle": 300, "time_to_explain": 60,
            "hints": [{"name": n, "url": h} for n, h in zip(names, hints)], "choices": choices,
            "bloom_level": bloom, "keywords": list(keywords), "source": list(zdroje), "practical": False}


def _result(sections, questions, reports=None) -> PipelineResult:
    lo_result = LOPipelineResult.__new__(LOPipelineResult)
    return PipelineResult(learning_objects=sections, questions=questions, reports=reports or {}, lo_result=lo_result)


def test_question_mapping_and_meta():
    sections = [_section(1, "Fotosyntéza"), _section(2, "Dýchanie", "Vytvoriť", [1])]
    questions = [
        _question(1, zdroje=("D1:2", "D2:5", "D1:2"), hints=('<p onclick="x">Pozri.</p>',)),
        _question(2, name="Výpočet", answer="Riešenie krok za krokom.", hints=(), bloom="create"),
        _question(1, question_type="choice", keywords=("a", "b"), choices=[
            {"text": "Áno", "is_correct": True}, {"text": "Nie", "is_correct": False}]),
    ]
    out = build_output(_result(sections, questions), _record("bio.pdf", "chem.pdf"), timings={"extract": 1.0})

    unit = out.result.units[0]
    assert unit.title == "bio, chem"
    assert unit.description == "Tematické celky: Fotosyntéza, Dýchanie"
    first, second, third = unit.questions
    # empty name -> section title; the HTML of the hints is cleaned
    assert (first.name, first.description, first.type) == ("Fotosyntéza", "Čo je to?", "open")
    assert (first.time_to_handle, first.time_to_explain) == (300, 60)
    assert [c.model_dump() for c in first.choices] == [{"text": "Toto.", "is_correct": True}]
    assert first.hints[0].model_dump() == {"name": "Ľahká nápoveda", "url": "<p>Pozri.</p>"}
    assert second.name == "Výpočet" and second.hints == [] and second.choices[0].text.startswith("Riešenie")
    assert third.type == "choice" and [c.is_correct for c in third.choices] == [True, False]

    meta = out.report.question_meta
    assert [(m.question, m.section_id, m.bloom_level) for m in meta] == [
        (0, "S-1", "understand"), (1, "S-2", "create"), (2, "S-1", "understand")]
    assert meta[2].keywords == ["a", "b"]
    assert [(s.file, s.pages) for s in meta[0].source] == [("bio.pdf", [2]), ("chem.pdf", [5])]
    assert out.report.sections[1].prerequisites == ["S-1"]
    assert out.report.sections[0].lo_type == "legacy"
    assert out.report.timings == {"extract": 1.0}


def test_invalid_question_is_left_out_with_a_warning():
    questions = [_question(1), _question(1, description="x" * 5000), _question(1, answer="")]
    out = build_output(_result([_section(1, "A")], questions), _record("a.pdf"), timings={})
    assert len(out.result.units[0].questions) == 1 and out.dropped == 2
    codes = [w.code for w in out.report.warnings]
    assert codes == ["field_too_long", "invalid_question"]
    assert all("x" * 20 not in w.message for w in out.report.warnings)  # no question text
    assert [m.question for m in out.report.question_meta] == [0]


def test_unknown_bloom_level_warns():
    out = build_output(_result([_section(1, "A", bloom="Zázrak")], [_question(1, bloom="")]), _record("a.pdf"),
                       timings={})
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
    out = build_output(_result([_section(1, "A")], [_question(1)], reports), _record("a.pdf"), timings={"extract": 1.0})
    quality = out.report.quality
    assert (quality.section_faithfulness_avg, quality.question_faithfulness_avg) == (0.75, 1.0)
    assert quality.answerability_avg is None
    assert quality.topic_coverage == 0.75
    assert out.report.sections[0].topic == "Rastliny" == out.report.question_meta[0].topic
    assert out.report.warnings[0].code == "section_without_questions"
    assert out.report.timings == {"extract": 1.0, "learning_objects": 10.0, "lo_evaluation": 2.5,
                                  "questions": 7.0, "q_evaluation": 3.0}


def test_no_questions_gives_one_empty_unit():
    out = build_output(_result([], []), _record("a.pdf"), timings={})
    assert len(out.result.units) == 1 and out.result.units[0].questions == []
    assert out.report.quality.section_faithfulness_avg is None


def test_hint_html_is_cleaned():
    assert clean_hint_html('<p style="x" onclick="y">a<script>b</script><em>c</em><img src=x></p>') == "<p>a<em>c</em></p>"


def test_cut_at_word():
    assert cut_at_word("jeden dva tri", 9) == "jeden dva"
    assert cut_at_word("dlhéslovobezmedzery", 5) == "dlhés"
    assert cut_at_word("  krátke  ", 50) == "krátke"


def test_worker_does_not_start_with_bad_config(monkeypatch):
    from service.settings import ServiceSettings

    bad = ServiceSettings(_env_file=None, llm_provider="gemini", gemini_api_key="", embed_provider="gemini")
    monkeypatch.setattr(worker_main, "ServiceSettings", lambda: bad)
    assert worker_main.main() == 1
