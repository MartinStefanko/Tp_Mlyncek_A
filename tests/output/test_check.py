"""Check of the output before saving (qgen.output.check_output)."""

import pytest

from qgen.contract import Choice, Question, Report, Unit, UnitsResult
from qgen.contract.report import QuestionMeta
from qgen.output import OutputCheckError, check_output
from tests.output.data import build, question, section


def _output():
    out = build([section(1, "A"), section(2, "B")], [question(1), question(2), question(1)])
    check_output(out.result, out.report)
    return out.result, out.report


def _with_meta(report: Report, meta: list[QuestionMeta]) -> Report:
    return report.model_copy(update={"question_meta": meta})


def _problems(result, report) -> list[str]:
    with pytest.raises(OutputCheckError) as caught:
        check_output(result, report)
    return caught.value.problems


def test_missing_meta():
    result, report = _output()
    assert _problems(result, _with_meta(report, report.question_meta[:2])) == [
        "question_meta: no entry for unit 0 question 2"]


def test_extra_and_duplicate_meta():
    result, report = _output()
    meta = report.question_meta
    extra = meta[0].model_copy(update={"question": 7})
    assert _problems(result, _with_meta(report, [*meta, extra])) == ["question_meta: unit 0 question 7 not in units"]
    problems = _problems(result, _with_meta(report, [*meta, meta[1]]))
    assert problems == ["question_meta: unit 0 question 1 listed 2 times"]
    other_unit = meta[0].model_copy(update={"unit": 1})
    assert "question_meta: unit 1 question 0 not in units" in _problems(result, _with_meta(report, [*meta, other_unit]))


def test_unknown_section_id():
    result, report = _output()
    meta = [report.question_meta[0].model_copy(update={"section_id": "S-9"}), *report.question_meta[1:]]
    assert _problems(result, _with_meta(report, meta)) == [
        "question_meta: unit 0 question 0 has unknown section_id S-9"]


def test_invalid_data_fails_model_and_schema_without_the_value():
    result, report = _output()
    bad = Question.model_construct(
        name="x" * 300, description="tajný text dokumentu", type="open", time_to_handle=1, time_to_explain=0,
        hints=[], choices=[Choice(text="a", is_correct=True)],
    )
    broken = UnitsResult.model_construct(units=[Unit.model_construct(title="t", description="", questions=[
        *result.units[0].questions[:2], bad])])
    problems = _problems(broken, report)
    assert "units model: units.0.questions.2.name string_too_long" in problems
    assert "lesson schema: $.units[0].questions[2].name maxLength" in problems
    assert all("tajný" not in p and "xxx" not in p for p in problems)


def test_message_lists_at_most_five_problems():
    error = OutputCheckError([f"p{i}" for i in range(7)])
    assert str(error) == "output check failed: p0; p1; p2; p3; p4 (+2 more)"
