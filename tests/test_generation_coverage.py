"""Number of questions and coverage of sections (tematické celky), with FakeLLM, no network.

- per_section (options.questions.per_section) is the MINIMUM of questions per section in
  the question generation prompt, not a limit; the prompt no longer allows "nothing".
- Sections without a generated question get one supplementary call with only them; a
  section still without a question gets a section_without_questions warning.
- Valid JSON of the wrong shape (an object instead of an array) in section or question
  generation is invalid JSON: one repeated call, then LLMError(invalid_json=True).
- Diagnostics: items_dropped by reason, section counts, questions per section.
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
from tests.fakes import FakeEmbedder, FakeLLM

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
    return Settings.model_construct()


def _question(lo_id, n=1):
    return {
        "lo_id": lo_id,
        "typ": "teoreticka_otazka",
        "otazka": f"Otázka {n} k celku {lo_id}?",
        "odpoved": "Odpoveď.",
        "napoveda": "Pozri definíciu.",
        "citovane_zdroje": [f"D1:{lo_id}"],
    }


def _section_ids(prompt):
    return [int(x) for x in re.findall(r"- lo_id: (\d+)", prompt)]


def _is_question_generation(prompt):
    return "LO 1:" in prompt and "- lo_id:" in prompt


class ScriptedModel:
    """Answers by the type of prompt. `generate(section_ids, call_no)` scripts the question
    generation calls; quality score and the checks always give 5."""

    def __init__(self, generate):
        self.generate = generate
        self.generation_prompts = []

    def __call__(self, prompt, model):
        if _is_question_generation(prompt):
            self.generation_prompts.append(prompt)
            return self.generate(_section_ids(prompt), len(self.generation_prompts))
        if "Item id:" in prompt:  # question quality score
            ids = [int(x) for x in re.findall(r"Item id: (\d+)", prompt)]
            return json.dumps([{"id": i, "skore": 5, "zdovodnenie": "ok"} for i in ids])
        if "ITEM ID:" in prompt:  # faithfulness / answerability
            ids = [int(x) for x in re.findall(r"ITEM ID: (\d+)", prompt)]
            return json.dumps([{"item_id": i, "skore": 5, "zdovodnenie": "ok"} for i in ids])
        return "[]"


def _run(model, settings, per_section=None):
    llm = FakeLLM(default=model)
    result = generate_all_items(
        SECTIONS, SEGMENTS, llm=llm, embedder=FakeEmbedder(), verbose=False,
        document_language="sk", settings=settings, per_section=per_section,
    )
    return result, result.reports["question_diagnostics"]


# --- prompts --------------------------------------------------------------------------

@pytest.mark.parametrize("language", ["sk", "en"])
def test_per_section_is_a_minimum_in_the_prompt(language):
    prompt = build_item_generation_prompt("LO 1:\n- lo_id: 1", language, per_section=5)
    if language == "sk":
        assert "Pre každé LO vytvor najmenej tento počet položiek: 5." in prompt
        assert "minimum, nie strop" in prompt
        assert "pre dané LO nevytváraj nič" not in prompt
        assert 'jediný objekt {"lo_id": <id>, "reason": "<krátky dôvod>"}' in prompt
        assert 'Aplikovať, Analyzovať, Hodnotiť alebo Vytvoriť vytvor aspoň jednu položku typu "prakticka_uloha"' in prompt
    else:
        assert "For each LO, create at least 5 items; if the content allows, create more." in prompt
        assert "create nothing for that LO" not in prompt
        assert 'a single object {"lo_id": <id>, "reason": "<short reason>"}' in prompt
        assert 'create at least one item of type "prakticka_uloha" (without code)' in prompt
    # The output format stays an array of items.
    assert "pole objektov" in prompt or "array of objects" in prompt


def test_per_section_default_is_three():
    assert DEFAULT_PER_SECTION == 3
    assert "najmenej tento počet položiek: 3." in build_item_generation_prompt("x", "sk")


@pytest.mark.parametrize("language, phrase", [
    ("sk", "Z materiálu vytiahni VŠETKY samostatne vyučiteľné časti učiva"),
    ("en", "Extract ALL separately teachable parts of the material"),
])
def test_section_prompt_asks_for_all_teachable_parts(language, phrase):
    assert phrase in build_lo_generation_prompt("text", language)


def test_per_section_reaches_the_generation_prompt(settings):
    model = ScriptedModel(lambda ids, n: json.dumps([_question(i) for i in ids]))
    _, diagnostics = _run(model, settings, per_section=7)
    assert "najmenej tento počet položiek: 7." in model.generation_prompts[0]
    assert diagnostics["per_section_minimum"] == 7


def test_run_pipeline_passes_per_section(settings):
    lo_answer = json.dumps([
        {"vzdelávací_objekt": name, "bloom_level": "Pochopiť", "odporúčané_aktivity": ["čítanie"],
         "odporúčané_zadania": "Vysvetli.", "citovane_zdroje": [f"D1:{page}"]}
        for page, name in ((1, "Fotosyntéza"), (2, "Dýchanie"), (3, "Transpirácia"))
    ], ensure_ascii=False)
    generation_prompts = []

    def answer(prompt, model):
        if _is_question_generation(prompt):
            generation_prompts.append(prompt)
            return json.dumps([_question(i) for i in _section_ids(prompt)])
        if "extrahuj merateľné vzdelávacie ciele" in prompt:
            return lo_answer
        if "LO ID:" in prompt:  # section faithfulness
            ids = [int(x) for x in re.findall(r"LO ID: (\d+)", prompt)]
            return json.dumps([{"lo_id": i, "skore": 5, "zdovodnenie": "ok"} for i in ids])
        return ScriptedModel(lambda ids, n: "[]")(prompt, model)

    llm = FakeLLM(default=answer)
    result = run_pipeline(SEGMENTS, llm=llm, embedder=FakeEmbedder(), language="sk", settings=settings,
                          per_section=4)
    sections = result.reports["section_diagnostics"]
    assert sections["sections_generated"] == 3
    assert sections["sections_accepted"] == len(result.learning_objects) >= 1
    assert sections["sections_after_clustering"] >= sections["sections_accepted"]
    assert generation_prompts and all("najmenej tento počet položiek: 4." in p for p in generation_prompts)
    assert result.reports["question_diagnostics"]["per_section_minimum"] == 4


# --- supplementary call ---------------------------------------------------------------

def test_missing_sections_get_one_supplementary_call(settings):
    def generate(ids, call_no):
        if call_no == 1:  # sections 1 and 2 only
            return json.dumps([_question(1, 1), _question(1, 2), _question(2, 1)])
        return json.dumps([_question(i) for i in ids])

    model = ScriptedModel(generate)
    result, diagnostics = _run(model, settings)
    assert len(model.generation_prompts) == 2
    assert _section_ids(model.generation_prompts[0]) == [1, 2, 3]
    assert _section_ids(model.generation_prompts[1]) == [3]  # only the missing section
    assert diagnostics["supplementary_call"] == {"sections_missing": [3], "calls": 1, "sections_filled": [3]}
    assert {q["lo_id"] for q in result.questions} == {1, 2, 3}
    assert diagnostics["warnings"] == []
    assert diagnostics["questions_per_section"]["generated"] == {"min": 1, "avg": 1.33, "max": 2}


def test_no_supplementary_call_when_every_section_has_a_question(settings):
    model = ScriptedModel(lambda ids, n: json.dumps([_question(i) for i in ids]))
    _, diagnostics = _run(model, settings)
    assert len(model.generation_prompts) == 1
    assert diagnostics["supplementary_call"] == {"sections_missing": [], "calls": 0, "sections_filled": []}


def test_section_still_without_question_gets_a_warning_with_the_model_reason(settings):
    def generate(ids, call_no):
        if call_no == 1:
            return json.dumps([_question(1), _question(2)])
        # supplementary call: the model says section 3 has no subject matter
        return json.dumps([{"lo_id": 3, "reason": "Text obsahuje iba obsah kapitoly."}], ensure_ascii=False)

    model = ScriptedModel(generate)
    result, diagnostics = _run(model, settings)
    assert len(model.generation_prompts) == 2
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
    assert all(q["otazka"] for q in result.all_questions)
    assert diagnostics["items_dropped"]["total"] == 0


def test_warning_without_model_reason(settings):
    model = ScriptedModel(lambda ids, n: json.dumps([_question(1)]) if n == 1 else "[]")
    _, diagnostics = _run(model, settings)
    warned = {(w["section_id"], w["stage"]) for w in diagnostics["warnings"]}
    assert warned == {(2, "generation"), (3, "generation")}
    assert all("also not in the supplementary call" in w["reason"] for w in diagnostics["warnings"])


# --- dropped items --------------------------------------------------------------------

def test_items_dropped_by_reason(settings):
    def generate(ids, call_no):
        no_lo_id = {k: v for k, v in _question(1).items() if k != "lo_id"}
        return json.dumps([_question(1), _question(2), _question(3), no_lo_id, "not an object", 42])

    _, diagnostics = _run(ScriptedModel(generate), settings)
    by_reason = diagnostics["items_dropped"]["by_reason"]
    assert by_reason[DROP_MISSING_LO_ID] == 1
    assert by_reason[DROP_INVALID_ITEM_SHAPE] == 2
    assert by_reason["validation"] == 0 and by_reason["low_score"] == 0
    assert diagnostics["items_dropped"]["total"] == 3


def test_items_dropped_by_the_filter_are_counted(settings):
    model = ScriptedModel(lambda ids, n: json.dumps([_question(i) for i in ids]))

    def low_scores(prompt, model_name):
        if "ITEM ID:" in prompt:
            ids = [int(x) for x in re.findall(r"ITEM ID: (\d+)", prompt)]
            return json.dumps([{"item_id": i, "skore": 1, "zdovodnenie": "zle"} for i in ids])
        return model(prompt, model_name)

    llm = FakeLLM(default=low_scores)
    result = generate_all_items(SECTIONS, SEGMENTS, llm=llm, embedder=FakeEmbedder(), verbose=False,
                                document_language="sk", settings=settings)
    diagnostics = result.reports["question_diagnostics"]
    assert result.questions == []
    assert diagnostics["items_dropped"]["by_reason"]["low_score"] == 3
    assert {w["stage"] for w in diagnostics["warnings"]} == {"filter"}
    assert diagnostics["questions_per_section"]["accepted"] == {"min": 0, "avg": 0.0, "max": 0}


# --- wrong JSON shape = invalid JSON ----------------------------------------------------

WRAPPED = json.dumps({"items": [_question(1)]})


def test_question_generation_object_instead_of_array_raises(settings):
    model = ScriptedModel(lambda ids, n: WRAPPED)
    with pytest.raises(LLMError) as info:
        _run(model, settings)
    assert info.value.invalid_json and info.value.role == "questions"
    assert "wrong shape" in str(info.value)
    assert len(model.generation_prompts) == 2  # first answer + one repeated call


def test_question_generation_object_then_array_succeeds(settings):
    model = ScriptedModel(lambda ids, n: WRAPPED if n == 1 else json.dumps([_question(i) for i in ids]))
    result, _ = _run(model, settings)
    assert len(model.generation_prompts) == 2 and len(result.questions) == 3


def test_section_generation_object_instead_of_array_raises(settings):
    llm = FakeLLM(default=json.dumps({"vzdelavacie_ciele": [{"vzdelávací_objekt": "X"}]}))
    with pytest.raises(LLMError) as info:
        generate_learning_objects(SEGMENTS, llm=llm, verbose=False, document_language="sk", settings=settings)
    assert info.value.invalid_json and info.value.role == "lo"
    assert len(llm.calls) == 2


def test_missing_sources_object_instead_of_array_raises(settings):
    without_sources = json.dumps([{"vzdelávací_objekt": "X", "bloom_level": "Pochopiť", "citovane_zdroje": []}],
                                 ensure_ascii=False)
    llm = FakeLLM([without_sources], default=json.dumps({"id": 1, "citovane_zdroje": ["D1:1"]}))
    with pytest.raises(LLMError) as info:
        generate_learning_objects(SEGMENTS, llm=llm, verbose=False, document_language="sk", settings=settings)
    assert info.value.invalid_json and len(llm.calls) == 3  # generation + missing sources twice


def test_generate_json_expect_list(settings):
    assert retry.generate_json(FakeLLM(['{"a": 1}']), "p", role="lo", model="m", settings=settings) == {"a": 1}
    llm = FakeLLM(['{"a": 1}', "[1]"])
    assert retry.generate_json(llm, "p", role="lo", model="m", settings=settings, expect_list=True) == [1]
    with pytest.raises(LLMError) as info:
        retry.generate_json(FakeLLM(default='"text"'), "p", role="lo", model="m", settings=settings, expect_list=True)
    assert info.value.invalid_json and "str" in str(info.value)
    # Truly invalid JSON keeps its own message.
    with pytest.raises(LLMError) as info:
        retry.generate_json(FakeLLM(default="no json"), "p", role="lo", model="m", settings=settings, expect_list=True)
    assert "invalid JSON" in str(info.value)
