"""Number of questions and coverage of sections (tematické celky), with FakeLLM, no network.

- per_section (options.questions.per_section) is a lower bound in the generation prompt,
  never a target; the main instruction is "as many different questions as possible".
- Coverage (step 13): after the filter, sections with fewer accepted questions than
  per_section get one more call; a section still without a question gets a
  section_without_questions warning.
- Every prompt asks for {"items": [...]}; another shape is invalid JSON: one repeated call,
  then LLMError(invalid_json=True).
- Diagnostics: items_dropped by reason, questions per section.
"""

import json
import re

import pytest

from qgen.config import Settings
from qgen.llm import retry
from qgen.llm.errors import LLMError
from qgen.lo.generation import build_lo_generation_prompt, generate_learning_objects
from qgen.pipeline import run_pipeline
from qgen.questions.pipeline import (
    DEFAULT_PER_SECTION,
    DROP_INVALID_ITEM_SHAPE,
    DROP_MISSING_LO_ID,
    WARNING_SECTION_WITHOUT_QUESTIONS,
    build_item_generation_prompt,
    generate_all_items,
)
from qgen.questions.types import QuestionSettings
from tests.fakes import (
    EMPTY_ITEMS,
    FakeEmbedder,
    FakeLLM,
    OrthogonalVectors,
    check_answer,
    is_coverage_call,
    is_extra_round,
    is_question_generation,
    items,
    question,
    section_ids,
)

SEGMENTS = [
    {"page": 1, "source_id": "D1", "source_name": "a.pdf", "text": "Fotosyntéza je proces, pri ktorom rastliny tvoria cukor."},
    {"page": 2, "source_id": "D1", "source_name": "a.pdf", "text": "Dýchanie rastlín uvoľňuje energiu z cukru."},
    {"page": 3, "source_id": "D1", "source_name": "a.pdf", "text": "Transpirácia je odparovanie vody listami."},
]
SECTIONS = [
    {"id": 1, "vzdelávací_objekt": "Fotosyntéza", "bloom_level": "Pochopiť", "citovane_zdroje": ["D1:1"]},
    {"id": 2, "vzdelávací_objekt": "Dýchanie", "bloom_level": "Aplikovať", "citovane_zdroje": ["D1:2"]},
    {"id": 3, "vzdelávací_objekt": "Transpirácia", "bloom_level": "Analyzovať", "citovane_zdroje": ["D1:3"]},
]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(retry, "_sleep", lambda seconds: None)


@pytest.fixture
def settings():
    # all sections in one generation call, no extra round: one call per phase
    return Settings.model_construct(question_generation_batch_size=10, question_extra_rounds=0)


class ScriptedModel:
    """Answers by the type of prompt. `generate(section_ids, call_no)` scripts the first
    generation calls and the coverage calls, `extra(...)` the extra rounds; the quality
    score and the checks give `score`."""

    def __init__(self, generate, extra=None, score=5):
        self.generate = generate
        self.extra = extra or (lambda ids, n: EMPTY_ITEMS)
        self.score = score
        self.generation_prompts = []

    def __call__(self, prompt, model):
        if is_question_generation(prompt):
            self.generation_prompts.append(prompt)
            if is_extra_round(prompt):
                return self.extra(section_ids(prompt), len(self.generation_prompts))
            return self.generate(section_ids(prompt), len(self.generation_prompts))
        return check_answer(prompt, self.score) or EMPTY_ITEMS


def _run(model, settings, per_section=None, **kwargs):
    llm = FakeLLM(default=model)
    result = generate_all_items(
        SECTIONS, SEGMENTS, llm=llm, embedder=FakeEmbedder(default=OrthogonalVectors()), verbose=False,
        document_language="sk", settings=settings, per_section=per_section, **kwargs,
    )
    return result, result.reports["question_diagnostics"]


def _one_each(ids, n):
    return items(question(i, n) for i in ids)


# --- prompts --------------------------------------------------------------------------

@pytest.mark.parametrize("language", ["sk", "en"])
def test_main_instruction_is_as_many_as_possible_and_per_section_is_only_a_lower_bound(language):
    prompt = build_item_generation_prompt("blocks", language, QuestionSettings(per_section=5))
    if language == "sk":
        assert "vytvor čo najviac rôznych otázok, ktoré obsah tematického celku dovoľuje" in prompt
        assert "pokry každý pojem, vzorec, postup, príklad a súvislosť" in prompt
        assert "nesmie mať menej otázok ako 5. Je to iba poistka, nie cieľ" in prompt
        assert 'jediný objekt {"lo_id": <id>, "reason": "<krátky dôvod>"}' in prompt
    else:
        assert "create as many different questions as its content allows" in prompt
        assert "fewer questions than 5. This is only a safeguard, not a target" in prompt
        assert 'a single object {"lo_id": <id>, "reason": "<short reason>"}' in prompt
    # no fixed number as a target, the output is an object with "items"
    assert "presne 5" not in prompt and "exactly 5 questions" not in prompt
    assert '{"items": [...]}' in prompt


def test_per_section_default_is_three():
    assert DEFAULT_PER_SECTION == 3
    assert "nesmie mať menej otázok ako 3." in build_item_generation_prompt("x", "sk")


@pytest.mark.parametrize("language, phrase", [
    ("sk", "Z materiálu vytiahni VŠETKY samostatne vyučiteľné časti učiva"),
    ("en", "Extract ALL separately teachable parts of the material"),
])
def test_section_prompt_asks_for_all_teachable_parts(language, phrase):
    assert phrase in build_lo_generation_prompt("text", language)


def test_per_section_reaches_the_generation_prompt(settings):
    model = ScriptedModel(_one_each)
    _, diagnostics = _run(model, settings, per_section=7)
    assert "nesmie mať menej otázok ako 7." in model.generation_prompts[0]
    assert diagnostics["per_section_minimum"] == 7


def test_run_pipeline_passes_per_section(settings):
    lo_answer = items(
        {"vzdelávací_objekt": name, "bloom_level": "Pochopiť", "odporúčané_aktivity": ["čítanie"],
         "odporúčané_zadania": "Vysvetli.", "citovane_zdroje": [f"D1:{page}"]}
        for page, name in ((1, "Fotosyntéza"), (2, "Dýchanie"), (3, "Transpirácia"))
    )
    model = ScriptedModel(_one_each)

    def answer(prompt, model_name):
        if "extrahuj merateľné vzdelávacie ciele" in prompt:
            return lo_answer
        return model(prompt, model_name)

    llm = FakeLLM(default=answer)
    result = run_pipeline(SEGMENTS, llm=llm, embedder=FakeEmbedder(default=OrthogonalVectors()), language="sk",
                          settings=settings, per_section=4)
    sections = result.reports["section_diagnostics"]
    assert sections["sections_generated"] == 3
    assert sections["sections_accepted"] == len(result.learning_objects) >= 1
    assert model.generation_prompts
    assert all("nesmie mať menej otázok ako 4." in p for p in model.generation_prompts)
    assert result.reports["question_diagnostics"]["per_section_minimum"] == 4


# --- coverage call --------------------------------------------------------------------

def test_sections_below_the_minimum_get_one_coverage_call(settings):
    def generate(ids, call_no):
        if call_no == 1:  # section 1 gets 2 questions, section 2 one, section 3 none
            return items([question(1, 1), question(1, 2), question(2, 1)])
        return items(question(i, 10 + n) for i in ids for n in range(2))

    model = ScriptedModel(generate)
    result, diagnostics = _run(model, settings, per_section=2)
    assert len(model.generation_prompts) == 2
    assert section_ids(model.generation_prompts[0]) == [1, 2, 3]
    coverage = model.generation_prompts[1]
    assert is_coverage_call(coverage)
    assert section_ids(coverage) == [2, 3]  # only the sections below the minimum
    assert "- lo_id 2 (chýba: 1):" in coverage and "- lo_id 3 (chýba: 2):" in coverage
    assert "[choice] Otázka 1 k celku 2?" in coverage  # existing questions are listed
    assert diagnostics["supplementary_call"] == {"sections_missing": [2, 3], "calls": 1, "sections_filled": [2, 3]}
    assert {q["lo_id"] for q in result.questions} == {1, 2, 3}
    assert {q["origin"] for q in result.questions if q["lo_id"] == 3} == {"coverage"}
    assert diagnostics["warnings"] == []


def test_section_whose_questions_were_all_dropped_gets_the_coverage_call(settings):
    def generate(ids, call_no):
        if call_no == 1:
            return items(question(i, 1) for i in ids)
        return items(question(i, 2) for i in ids)

    def score(prompt, model_name):
        if "ITEM ID:" in prompt and "Otázka 1 k celku 2?" in prompt:
            ids = [int(x) for x in re.findall(r"ITEM ID: (\d+)", prompt)]
            low = {int(x) for x in re.findall(r"ITEM ID: (\d+)\n[^\n]*\n[^\n]*\n[^\n]*Otázka 1 k celku 2\?", prompt)}
            return items({"item_id": i, "skore": 1 if i in low else 5, "zdovodnenie": "x"} for i in ids)
        return model(prompt, model_name)

    model = ScriptedModel(generate)
    llm = FakeLLM(default=score)
    result = generate_all_items(SECTIONS, SEGMENTS, llm=llm, embedder=FakeEmbedder(default=OrthogonalVectors()),
                                verbose=False, document_language="sk", settings=settings, per_section=1)
    diagnostics = result.reports["question_diagnostics"]
    assert diagnostics["supplementary_call"]["sections_missing"] == [2]
    assert diagnostics["supplementary_call"]["sections_filled"] == [2]
    assert [q["description"] for q in result.questions if q["lo_id"] == 2] == ["Otázka 2 k celku 2?"]


def test_no_coverage_call_when_every_section_has_enough(settings):
    model = ScriptedModel(_one_each)
    _, diagnostics = _run(model, settings, per_section=1)
    assert len(model.generation_prompts) == 1
    assert diagnostics["supplementary_call"] == {"sections_missing": [], "calls": 0, "sections_filled": []}


def test_section_still_without_question_gets_a_warning_with_the_model_reason(settings):
    def generate(ids, call_no):
        if call_no == 1:
            return items([question(1), question(2),
                          {"lo_id": 3, "reason": "Text obsahuje iba obsah kapitoly."}])
        return EMPTY_ITEMS

    model = ScriptedModel(generate)
    result, diagnostics = _run(model, settings, per_section=1)
    assert len(model.generation_prompts) == 2  # first call + coverage for section 3
    assert diagnostics["warnings"] == [{
        "code": WARNING_SECTION_WITHOUT_QUESTIONS,
        "message": "Section 3 (Bloom: Analyzovať) has no question (generation): Text obsahuje iba obsah kapitoly.",
        "section_id": 3,
        "bloom_level": "Analyzovať",
        "stage": "generation",
        "reason": "Text obsahuje iba obsah kapitoly.",
    }]
    assert diagnostics["sections_without_content"] == {"3": "Text obsahuje iba obsah kapitoly."}
    assert diagnostics["supplementary_call"]["sections_filled"] == []
    # The marker is not a question and not a dropped item.
    assert all(q["description"] for q in result.all_questions)
    assert diagnostics["items_dropped"]["total"] == 0


def test_warning_without_model_reason(settings):
    model = ScriptedModel(lambda ids, n: items([question(1)]) if n == 1 else EMPTY_ITEMS)
    _, diagnostics = _run(model, settings, per_section=1)
    warned = {(w["section_id"], w["stage"]) for w in diagnostics["warnings"]}
    assert warned == {(2, "generation"), (3, "generation")}
    assert all("also not in the coverage call" in w["reason"] for w in diagnostics["warnings"])


# --- dropped items --------------------------------------------------------------------

def test_items_dropped_by_reason(settings):
    def generate(ids, call_no):
        no_lo_id = {k: v for k, v in question(1, 5).items() if k != "lo_id"}
        return items([question(1), question(2), question(3), no_lo_id, "not an object", 42])

    _, diagnostics = _run(ScriptedModel(generate), settings, per_section=1)
    by_reason = diagnostics["items_dropped"]["by_reason"]
    assert by_reason[DROP_MISSING_LO_ID] == 1
    assert by_reason[DROP_INVALID_ITEM_SHAPE] == 2
    assert by_reason["validation"] == 0 and by_reason["low_score"] == 0
    assert diagnostics["items_dropped"]["total"] == 3


def test_items_dropped_by_the_filter_are_counted(settings):
    model = ScriptedModel(lambda ids, n: items(question(i, n) for i in ids), score=1)
    result, diagnostics = _run(model, settings, per_section=1)
    assert result.questions == []
    assert len(model.generation_prompts) == 2  # first call + coverage (all sections had 0 accepted)
    assert diagnostics["items_dropped"]["by_reason"]["low_score"] == 6
    assert {w["stage"] for w in diagnostics["warnings"]} == {"filter"}
    assert diagnostics["questions_per_section"]["accepted"] == {"min": 0, "avg": 0.0, "max": 0}


# --- wrong JSON shape = invalid JSON ----------------------------------------------------

ARRAY = json.dumps([question(1)], ensure_ascii=False)


@pytest.mark.parametrize("answer", [ARRAY, '{"questions": []}', '{"items": {"lo_id": 1}}'])
def test_question_generation_wrong_shape_raises(settings, answer):
    model = ScriptedModel(lambda ids, n: answer)
    with pytest.raises(LLMError) as info:
        _run(model, settings)
    assert info.value.invalid_json and info.value.role == "questions"
    assert "wrong shape" in str(info.value)
    assert len(model.generation_prompts) == 2  # first answer + one repeated call


def test_question_generation_array_then_object_succeeds(settings):
    model = ScriptedModel(lambda ids, n: ARRAY if n == 1 else _one_each(ids, n))
    result, _ = _run(model, settings, per_section=1)
    assert len(model.generation_prompts) == 2 and len(result.questions) == 3


@pytest.mark.parametrize("kind", ["Item id:", "ITEM ID:"])
def test_evaluation_and_checks_wrong_shape_raise(settings, kind):
    model = ScriptedModel(_one_each)

    def answer(prompt, model_name):
        if kind in prompt:
            return "[]"
        return model(prompt, model_name)

    with pytest.raises(LLMError) as info:
        generate_all_items(SECTIONS, SEGMENTS, llm=FakeLLM(default=answer),
                           embedder=FakeEmbedder(default=OrthogonalVectors()), verbose=False,
                           document_language="sk", settings=settings, per_section=1)
    assert info.value.invalid_json and info.value.role in ("eval", "check")


def test_section_generation_wrong_shape_raises(settings):
    llm = FakeLLM(default=json.dumps([{"vzdelávací_objekt": "X"}]))
    with pytest.raises(LLMError) as info:
        generate_learning_objects(SEGMENTS, llm=llm, verbose=False, document_language="sk", settings=settings)
    assert info.value.invalid_json and info.value.role == "lo"
    assert len(llm.calls) == 2


def test_missing_sources_wrong_shape_raises(settings):
    without_sources = items([{"vzdelávací_objekt": "X", "bloom_level": "Pochopiť", "citovane_zdroje": []}])
    llm = FakeLLM([without_sources], default=json.dumps([{"id": 1, "citovane_zdroje": ["D1:1"]}]))
    with pytest.raises(LLMError) as info:
        generate_learning_objects(SEGMENTS, llm=llm, verbose=False, document_language="sk", settings=settings)
    assert info.value.invalid_json and len(llm.calls) == 3  # generation + missing sources twice


def test_generate_json_items_key(settings):
    assert retry.generate_json(FakeLLM(['{"a": 1}']), "p", role="lo", model="m", settings=settings) == {"a": 1}
    llm = FakeLLM(["[1]", '{"items": [1]}'])
    assert retry.generate_json(llm, "p", role="lo", model="m", settings=settings, items_key="items") == [1]
    with pytest.raises(LLMError) as info:
        retry.generate_json(FakeLLM(default='{"x": []}'), "p", role="lo", model="m", settings=settings,
                            items_key="items")
    assert info.value.invalid_json and "object without 'items'" in str(info.value)
    with pytest.raises(LLMError) as info:
        retry.generate_json(FakeLLM(default='{"items": "x"}'), "p", role="lo", model="m", settings=settings,
                            items_key="items")
    assert "'items' is str" in str(info.value)
    # Truly invalid JSON keeps its own message.
    with pytest.raises(LLMError) as info:
        retry.generate_json(FakeLLM(default="no json"), "p", role="lo", model="m", settings=settings,
                            items_key="items")
    assert "invalid JSON" in str(info.value)
