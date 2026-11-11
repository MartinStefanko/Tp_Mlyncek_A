"""ECEH length limits and shortening by the (fake) LLM (qgen.output.limits, step 14)."""

import json

import pytest

from qgen.contract.units import CHOICE_TEXT_MAX, QUESTION_DESCRIPTION_MAX, QUESTION_NAME_MAX
from qgen.llm import retry
from qgen.output import check_output, cut_at_word
from qgen.output.limits import SHORTEN_ROLE, build_shorten_prompt, too_long_fields
from tests.fakes import INVALID_JSON, FakeLLM, client_error, items
from tests.output.data import build, question, section, settings

LONG = "slovo " * 900  # 5400 characters, over every limit
SECRET = "tajnytext"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(retry, "_sleep", lambda seconds: None)


def _fields_in(prompt: str) -> list[dict]:
    return json.loads(prompt.rsplit("(JSON):", 1)[1])


def _choice_question(texts, correct=(True, False, False)):
    return question(1, question_type="choice",
                    choices=[{"text": t, "is_correct": c} for t, c in zip(texts, correct)])


def test_nothing_too_long_makes_no_call():
    llm = FakeLLM()
    out = build([section(1, "A")], [question(1), _choice_question(["a", "b", "c"])], llm=llm)
    assert llm.calls == [] and out.dropped == 0


def test_shortening_succeeds_and_keeps_order_and_is_correct():
    texts = ["Krátka", SECRET + " " + LONG, "Iná"]
    item = _choice_question(texts, correct=(False, True, False))
    item["description"] = LONG
    llm = FakeLLM([items([{"field": "choices[1]", "text": "Skrátená možnosť"},
                          {"field": "description", "text": "Skrátené zadanie?"},
                          {"field": "choices[0]", "text": "nežiadané pole sa nezmení"}])])
    out = build([section(1, "A")], [item], llm=llm)

    assert len(llm.calls) == 1
    model, prompt = llm.calls[0]
    assert model == settings().llm_model_questions  # role "questions"
    assert SHORTEN_ROLE == "questions"
    # only the too long fields with their limits
    assert [(f["field"], f["limit"]) for f in _fields_in(prompt)] == [
        ("description", QUESTION_DESCRIPTION_MAX), ("choices[1]", CHOICE_TEXT_MAX)]
    assert "is_correct" not in prompt and "Skráť" in prompt

    shortened = out.result.units[0].questions[0]
    assert shortened.description == "Skrátené zadanie?"
    assert [(c.text, c.is_correct) for c in shortened.choices] == [
        ("Krátka", False), ("Skrátená možnosť", True), ("Iná", False)]
    assert out.report.warnings == [] and out.dropped == 0
    check_output(out.result, out.report)


def test_still_too_long_is_left_out_with_field_too_long():
    item = _choice_question(["a", SECRET + LONG, "c"])
    llm = FakeLLM([items([{"field": "choices[1]", "text": SECRET + LONG}])])
    out = build([section(1, "A")], [question(1), item], llm=llm)

    assert len(out.result.units[0].questions) == 1 and out.dropped == 1
    (warning,) = out.report.warnings
    assert warning.code == "field_too_long"
    length = len((SECRET + LONG).strip())  # the answer of the LLM is stripped
    assert warning.message == (
        f"Question 2 was left out: field choices[1] has {length} characters (limit {CHOICE_TEXT_MAX}).")
    assert SECRET not in warning.message
    assert [m.question for m in out.report.question_meta] == [0]
    check_output(out.result, out.report)


def test_name_is_cut_at_a_word_end_when_still_too_long():
    item = question(1, name="Dlhý názov " * 40)  # 440 characters
    llm = FakeLLM([items([])])  # the LLM shortens nothing
    out = build([section(1, "A")], [item], llm=llm)
    assert [f["field"] for f in _fields_in(llm.calls[0][1])] == ["name"]
    name = out.result.units[0].questions[0].name
    assert len(name) <= QUESTION_NAME_MAX and name.endswith("názov")
    assert out.dropped == 0 and out.report.warnings == []


def test_name_shortened_by_the_llm():
    llm = FakeLLM([items([{"field": "name", "text": "Krátky názov"}])])
    out = build([section(1, "A")], [question(1, name="x " * 200)], llm=llm)
    assert out.result.units[0].questions[0].name == "Krátky názov"


@pytest.mark.parametrize("failure", [client_error(400), INVALID_JSON], ids=["api_error", "invalid_json"])
def test_failed_shortening_does_not_stop_the_job(failure):
    item = question(1, description=LONG)
    llm = FakeLLM(default=failure)
    out = build([section(1, "A")], [item, question(1, name="n " * 200)], llm=llm)

    assert out.dropped == 1
    assert [w.code for w in out.report.warnings] == [
        "shortening_failed", "field_too_long", "shortening_failed"]
    assert out.report.warnings[0].message == "Question 1: shortening by the LLM failed (llm_error)."
    kept = out.result.units[0].questions
    assert len(kept) == 1 and len(kept[0].name) <= QUESTION_NAME_MAX  # name still cut at a word end
    check_output(out.result, out.report)


def test_wrong_answer_rows_are_ignored():
    llm = FakeLLM([items([{"field": "choices[0]", "text": "x"}, "nonsense", {"field": "description", "text": ""}])])
    out = build([section(1, "A")], [question(1, description=LONG)], llm=llm)
    assert out.dropped == 1 and out.report.warnings[0].code == "field_too_long"


def test_too_long_hint_is_not_sent_to_the_llm_and_drops_the_question():
    item = question(1, hints=("<p>" + "a" * 2100 + "</p>",))
    llm = FakeLLM()
    out = build([section(1, "A")], [item], llm=llm)
    assert llm.calls == []
    assert out.dropped == 1
    assert out.report.warnings[0].message.startswith("Question 1 was left out: field hints[0].url has 2107")


def test_english_document_gets_the_english_prompt():
    llm = FakeLLM([items([{"field": "description", "text": "Short task?"}])])
    out = build([section(1, "A")], [question(1, description=LONG)], llm=llm, language="en")
    assert "Shorten every text" in llm.calls[0][1]
    assert out.result.units[0].questions[0].description == "Short task?"


def test_progress_callback_per_shortening_call():
    seen = []
    llm = FakeLLM(default=items([]))
    build([section(1, "A")], [question(1, name="x " * 200), question(1), question(1, name="y " * 200)],
          llm=llm, progress_callback=lambda *args: seen.append(args))
    assert seen == [("output", 1, 2), ("output", 2, 2)]


def test_parallel_shortening_keeps_the_order():
    def answer(prompt, model):
        text = _fields_in(prompt)[0]["text"]
        return items([{"field": "description", "text": text[:5] + "?"}])

    questions = [question(1, description=f"Q{i} " + LONG) for i in range(5)]
    out = build([section(1, "A")], questions, llm=FakeLLM(default=answer),
                settings=settings(llm_max_concurrency=4))
    assert [q.description for q in out.result.units[0].questions] == [f"Q{i} sl?" for i in range(5)]


def test_too_long_fields_and_prompt():
    data = {"name": "n", "description": "d" * 4001, "choices": [{"text": "c" * 4001}],
            "hints": [{"name": "h" * 51, "url": "<p>u</p>"}]}
    assert [(f.field, f.length, f.limit) for f in too_long_fields(data)] == [
        ("description", 4001, 4000), ("choices[0]", 4001, 4000), ("hints[0].name", 51, 50)]
    prompt = build_shorten_prompt([{"field": "description", "limit": 4000, "text": "ľščť"}], "sk")
    assert "ľščť" in prompt and '"limit": 4000' in prompt


def test_cut_at_word():
    assert cut_at_word("jeden dva tri", 9) == "jeden dva"
    assert cut_at_word("dlhéslovobezmedzery", 5) == "dlhés"
    assert cut_at_word("  krátke  ", 50) == "krátke"
