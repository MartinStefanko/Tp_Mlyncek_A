"""Number of questions and coverage of sections (tematické celky), with FakeLLM, no network.

- per_section (options.questions.per_section) is a lower bound in the generation prompt,
  never a target; the main instruction is "as many different questions as possible".
- Coverage: after the filter, sections with fewer accepted questions than per_section get
  a coverage call, in up to QGEN_QUESTION_COVERAGE_ROUNDS rounds (default 2); a section
  without an accepted question gets its own call, first; the next round only for the
  sections still below the minimum. The prompt lists the existing questions with their
  status (accepted, or dropped and why). After the last round: section_without_questions
  (0 questions) and section_below_minimum (1 to per_section - 1) warnings.
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
    WARNING_SECTION_BELOW_MINIMUM,
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
    # all sections in one generation call, no extra round: one call per phase; coverage
    # rounds: code default (2)
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

def test_sections_below_the_minimum_get_a_coverage_call(settings):
    def generate(ids, call_no):
        if call_no == 1:  # section 1 gets 2 questions, section 2 one, section 3 none
            return items([question(1, 1), question(1, 2), question(2, 1)])
        return items(question(i, 10 * call_no + n) for i in ids for n in range(2))

    model = ScriptedModel(generate)
    result, diagnostics = _run(model, settings, per_section=2)
    assert len(model.generation_prompts) == 3  # first call + one round (2 calls); all filled: no round 2
    assert section_ids(model.generation_prompts[0]) == [1, 2, 3]
    zero, partial = model.generation_prompts[1:]
    assert is_coverage_call(zero) and is_coverage_call(partial)
    # the section without a question first, in its own call; only sections below the minimum
    assert section_ids(zero) == [3] and section_ids(partial) == [2]
    assert "- lo_id 3 (chýba: 2, prijaté: 0):" in zero
    assert "- lo_id 2 (chýba: 1, prijaté: 1):" in partial
    assert "1. [choice] Otázka 1 k celku 2? — prijatá" in partial  # existing questions with status
    assert diagnostics["supplementary_call"] == {
        "rounds_configured": 2,
        "sections_missing": [2, 3],
        "calls": 2,
        "sections_filled": [2, 3],
        "sections_still_below": [],
        "rounds": [{"round": 1, "sections": [2, 3], "sections_without_questions": [3], "calls": 2,
                    "questions_added": 4, "sections_filled": [2, 3], "sections_still_below": []}],
    }
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
    assert diagnostics["supplementary_call"] == {
        "rounds_configured": 2, "sections_missing": [], "calls": 0, "sections_filled": [],
        "sections_still_below": [], "rounds": [],
    }


def test_section_still_without_question_gets_a_warning_with_the_model_reason(settings):
    def generate(ids, call_no):
        if call_no == 1:
            return items([question(1), question(2),
                          {"lo_id": 3, "reason": "Text obsahuje iba obsah kapitoly."}])
        return EMPTY_ITEMS

    model = ScriptedModel(generate)
    result, diagnostics = _run(model, settings, per_section=1)
    # first call + coverage for section 3 in both rounds
    assert len(model.generation_prompts) == 3
    assert [section_ids(p) for p in model.generation_prompts[1:]] == [[3], [3]]
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
    assert all("also not in the coverage rounds" in w["reason"] for w in diagnostics["warnings"])


def _scored_by_marks(prompt, model_name):
    """Check answers by marks in the question text: "[q2]" quality 2, "[f2]" faithfulness 2,
    "[a1]" answerability 1, "[mark]" wrong choice marking; everything else 5."""

    def score(block, tag):
        found = re.search(rf"\[{tag}(\d)\]", block)
        return int(found.group(1)) if found else 5

    if "Item id:" in prompt:
        blocks = [b for b in re.split(r"(?=Item id: \d+)", prompt) if b.startswith("Item id:")]
        return items({"id": int(re.match(r"Item id: (\d+)", b).group(1)), "skore": score(b, "q"),
                      "zdovodnenie": "x"} for b in blocks)
    if "ITEM ID:" in prompt:
        answerability = "answerability" in prompt.lower()
        blocks = [b for b in re.split(r"(?=ITEM ID: \d+)", prompt) if b.startswith("ITEM ID:")]
        return items({"item_id": int(re.match(r"ITEM ID: (\d+)", b).group(1)),
                      "skore": score(b, "a" if answerability else "f"), "zdovodnenie": "x",
                      "choices_ok": "[mark]" not in b, "wrong_choices": ["A"] if "[mark]" in b else []}
                     for b in blocks)
    return None


def _run_scored(model, settings, per_section, sections=SECTIONS, **kwargs):
    def answer(prompt, model_name):
        return _scored_by_marks(prompt, model_name) or model(prompt, model_name)

    result = generate_all_items(
        sections, SEGMENTS, llm=FakeLLM(default=answer), embedder=FakeEmbedder(default=OrthogonalVectors()),
        verbose=False, document_language="sk", settings=settings, per_section=per_section, **kwargs,
    )
    return result, result.reports["question_diagnostics"]


def test_second_round_only_for_sections_still_below_the_minimum(settings):
    """Round 1 fills section 2; section 3 gets only an unanswerable question, so round 2 is
    only for section 3."""

    def generate(ids, call_no):
        if call_no == 1:
            return items([question(1, 1), question(1, 2), question(2, 1)])
        rows = []
        for i in ids:
            if i == 3 and call_no == 2:
                rows.append(question(3, call_no, description="Zlá [a1] k celku 3?"))
            else:
                rows.append(question(i, call_no))
        return items(rows)

    model = ScriptedModel(generate)
    result, diagnostics = _run_scored(model, settings, per_section=2)
    prompts = model.generation_prompts
    assert [section_ids(p) for p in prompts[1:]] == [[3], [2], [3]]
    second = prompts[3]
    assert "- lo_id 3 (chýba: 2, prijaté: 0):" in second
    assert "Zlá [a1] k celku 3? — VYRADENÁ: nezodpovedateľná zo zdroja" in second
    supplementary = diagnostics["supplementary_call"]
    assert [(r["round"], r["sections"], r["sections_without_questions"], r["sections_filled"],
             r["sections_still_below"]) for r in supplementary["rounds"]] == [
        (1, [2, 3], [3], [2], [3]),
        (2, [3], [3], [3], [3]),
    ]
    assert supplementary["sections_still_below"] == [3] and supplementary["calls"] == 3
    assert {q["origin"] for q in result.questions if q["lo_id"] == 3} == {"coverage_2"}
    # section 3 ends with 1 of 2 questions: section_below_minimum
    assert diagnostics["warnings"] == [{
        "code": WARNING_SECTION_BELOW_MINIMUM,
        "message": "Section 3 has 1 question(s), fewer than per_section 2, after 2 coverage round(s)",
        "section_id": 3,
        "questions": 1,
        "per_section": 2,
    }]


def test_sections_without_questions_come_first_each_in_its_own_call():
    settings = Settings.model_construct(question_generation_batch_size=2, question_extra_rounds=0,
                                        question_generation_attempts=1, llm_max_concurrency=1)
    sections = SECTIONS + [
        {"id": 4, "vzdelávací_objekt": "Osmóza", "bloom_level": "Pochopiť", "citovane_zdroje": ["D1:1"]},
        {"id": 5, "vzdelávací_objekt": "Klíčenie", "bloom_level": "Pochopiť", "citovane_zdroje": ["D1:2"]},
    ]

    def generate(ids, call_no):
        if not is_coverage_call(model.generation_prompts[-1]):
            # first calls (batch size 2): sections 2 and 4 get one question, 1, 3 and 5 none
            return items(question(i, 1, source=["D1:1"]) for i in ids if i in (2, 4))
        return items(question(i, 10 + call_no, source=["D1:1"]) for i in ids)

    model = ScriptedModel(generate)
    result, diagnostics = _run_scored(model, settings, per_section=2, sections=sections)
    coverage = [section_ids(p) for p in model.generation_prompts if is_coverage_call(p)]
    # round 1: sections without a question alone and first (in section order), then the others
    # by batch size; round 2: 1, 3 and 5 got one question each, so they are batched normally
    assert coverage == [[1], [3], [5], [2, 4], [1, 3], [5]]
    rounds = diagnostics["supplementary_call"]["rounds"]
    assert [r["sections_without_questions"] for r in rounds] == [[1, 3, 5], []]
    assert diagnostics["supplementary_call"]["sections_still_below"] == []


@pytest.mark.parametrize("rounds", [0, 1, 2, 5])
def test_number_of_coverage_rounds(settings, rounds):
    """A section the model never fills: one call per round; 1 = the former single call, 0 = off."""
    model = ScriptedModel(lambda ids, n: items([question(1), question(2)]) if n == 1 else EMPTY_ITEMS)
    _, diagnostics = _run(model, settings.model_copy(update={"question_coverage_rounds": rounds}), per_section=1)
    assert sum(is_coverage_call(p) for p in model.generation_prompts) == rounds
    assert diagnostics["supplementary_call"]["calls"] == rounds
    assert len(diagnostics["supplementary_call"]["rounds"]) == rounds
    assert diagnostics["supplementary_call"]["sections_missing"] == [3]
    assert [w["code"] for w in diagnostics["warnings"]] == [WARNING_SECTION_WITHOUT_QUESTIONS]


def test_coverage_rounds_argument_overrides_the_setting(settings):
    model = ScriptedModel(lambda ids, n: items([question(1), question(2)]) if n == 1 else EMPTY_ITEMS)
    _, diagnostics = _run(model, settings, per_section=1, coverage_rounds=1)
    assert diagnostics["supplementary_call"]["calls"] == 1


def test_rounds_stop_when_nothing_is_missing(settings):
    model = ScriptedModel(lambda ids, n: items([question(1), question(2)]) if n == 1 else _one_each(ids, n))
    _, diagnostics = _run(model, settings.model_copy(update={"question_coverage_rounds": 5}), per_section=1)
    assert sum(is_coverage_call(p) for p in model.generation_prompts) == 1
    assert len(diagnostics["supplementary_call"]["rounds"]) == 1
    assert diagnostics["warnings"] == []


def test_coverage_prompt_lists_the_drop_reasons(settings):
    """Every existing question is listed with its status: accepted, or dropped with a short reason."""

    def generate(ids, call_no):
        if call_no == 1:
            return items([
                question(1, 1), question(1, 2), question(3, 1), question(3, 2),
                question(2, 1, description="Dobrá k celku 2?"),
                question(2, 2, description="Nejasná [q2] k celku 2?"),
                question(2, 3, description="Mimo textu [f2] k celku 2?"),
                question(2, 4, description="Bez odpovede [a1] k celku 2?"),
                question(2, 5, description="Zle označená [mark] k celku 2?"),
                question(2, 1, description="Dobrá k celku 2?"),  # duplicate of the first
                question(2, 7, description="Bez zdroja k celku 2?", source=["D9:99"]),  # validation
            ])
        return _one_each(ids, call_no)

    model = ScriptedModel(generate)
    _, diagnostics = _run_scored(model, settings, per_section=2)
    coverage = [p for p in model.generation_prompts if is_coverage_call(p)]
    assert [section_ids(p) for p in coverage] == [[2]]
    prompt = coverage[0]
    assert "- lo_id 2 (chýba: 1, prijaté: 1):" in prompt
    assert "1. [choice] Dobrá k celku 2? — prijatá" in prompt
    assert "Nejasná [q2] k celku 2? — VYRADENÁ: nízke skóre kvality" in prompt
    assert "Mimo textu [f2] k celku 2? — VYRADENÁ: nevychádza z textu" in prompt
    assert "Bez odpovede [a1] k celku 2? — VYRADENÁ: nezodpovedateľná zo zdroja" in prompt
    assert "Zle označená [mark] k celku 2? — VYRADENÁ: zle označená možnosť" in prompt
    assert "6. [choice] Dobrá k celku 2? — VYRADENÁ: duplicita" in prompt
    assert "Bez zdroja k celku 2? — VYRADENÁ: neplatný formát" in prompt
    # the instructions: correctness and answerability from the source first
    assert "aby sa jej správna odpoveď dala jednoznačne zistiť iba z textu tematického celku" in prompt
    assert "Vyhni sa chybám vyradených otázok." in prompt
    # the drop reason in the report stays low_score
    assert diagnostics["items_dropped"]["by_reason"]["low_score"] == 3


def test_coverage_section_in_english():
    from qgen.questions.prompts import build_coverage_section

    existing = {7: [{"id": 1, "type": "open", "description": "What is X?"},
                    {"id": 2, "type": "choice", "description": "Which Y?"},
                    {"id": 3, "type": "choice", "description": "Which Z?"}]}
    statuses = {1: ["accepted"], 2: ["faithfulness", "answerability"], 3: ["wrong_choice_marking"]}
    text = build_coverage_section(existing, {7: 2}, "en", statuses)
    assert text.lstrip().startswith("COVERAGE SUPPLEMENT:")
    assert "- lo_id 7 (missing: 2, accepted: 1):" in text
    assert "1. [open] What is X? — accepted" in text
    assert "2. [choice] Which Y? — DROPPED: not supported by the text, not answerable from the source" in text
    assert "3. [choice] Which Z? — DROPPED: wrongly marked option" in text
    assert "its correct answer must be clearly found in the text of its section alone" in text


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
    # first call + 2 coverage rounds, each section without a question in its own call
    assert len(model.generation_prompts) == 1 + 3 + 3
    assert diagnostics["items_dropped"]["by_reason"]["low_score"] == 9
    assert {w["stage"] for w in diagnostics["warnings"]} == {"filter"}
    assert diagnostics["questions_per_section"]["accepted"] == {"min": 0, "avg": 0.0, "max": 0}


# --- wrong JSON shape = invalid JSON ----------------------------------------------------
# The repeated call gets SHAPE_CORRECTION; a non-empty bare array of objects in the repeated
# answer is taken as the items (gemini-3.1-flash-lite returned a bare array twice).

ARRAY = json.dumps([question(1)], ensure_ascii=False)


@pytest.mark.parametrize("answer", ['{"questions": []}', '{"items": {"lo_id": 1}}', "[]", "[1, 2]"])
def test_question_generation_wrong_shape_raises(settings, answer):
    model = ScriptedModel(lambda ids, n: answer)
    with pytest.raises(LLMError) as info:
        _run(model, settings)
    assert info.value.invalid_json and info.value.role == "questions"
    assert "wrong shape" in str(info.value)
    assert len(model.generation_prompts) == 2  # first answer + one repeated call
    assert "IMPORTANT: your previous answer had the wrong shape" not in model.generation_prompts[0]
    assert "IMPORTANT: your previous answer had the wrong shape" in model.generation_prompts[1]


def test_question_generation_bare_array_twice_is_taken_as_items(settings):
    model = ScriptedModel(lambda ids, n: ARRAY if n <= 2 else EMPTY_ITEMS)
    result, _ = _run(model, settings, per_section=1)
    assert [q["description"] for q in result.questions] == ["Otázka 1 k celku 1?"]
    assert 'Return ONLY a JSON object {"items": [...]}' in model.generation_prompts[1]


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
    llm = FakeLLM(default=json.dumps({"objects": [{"vzdelávací_objekt": "X"}]}))
    with pytest.raises(LLMError) as info:
        generate_learning_objects(SEGMENTS, llm=llm, verbose=False, document_language="sk", settings=settings)
    assert info.value.invalid_json and info.value.role == "lo"
    assert len(llm.calls) == 2


def test_missing_sources_wrong_shape_raises(settings):
    without_sources = items([{"vzdelávací_objekt": "X", "bloom_level": "Pochopiť", "citovane_zdroje": []}])
    llm = FakeLLM([without_sources], default=json.dumps({"sources": [{"id": 1, "citovane_zdroje": ["D1:1"]}]}))
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
    # A bare array of objects: the repeated call gets the correction; if the answer is again
    # a bare array of objects, it is taken as the items. A bare array in the first answer only
    # is never taken.
    llm = FakeLLM(default='[{"id": 1}]')
    assert retry.generate_json(llm, "p", role="check", model="m", settings=settings, items_key="items") == [{"id": 1}]
    assert [prompt for _, prompt in llm.calls] == ["p", "p" + retry.SHAPE_CORRECTION.format(shape="list", key="items")]
    llm = FakeLLM(['[{"id": 1}]', '{"items": [{"id": 2}]}'])
    assert retry.generate_json(llm, "p", role="check", model="m", settings=settings, items_key="items") == [{"id": 2}]
    with pytest.raises(LLMError):  # an empty bare array never becomes an empty result
        retry.generate_json(FakeLLM(default="[]"), "p", role="check", model="m", settings=settings, items_key="items")
    # Truly invalid JSON keeps its own message and repeats the same prompt.
    llm = FakeLLM(default="no json")
    with pytest.raises(LLMError) as info:
        retry.generate_json(llm, "p", role="lo", model="m", settings=settings, items_key="items")
    assert "invalid JSON" in str(info.value)
    assert [prompt for _, prompt in llm.calls] == ["p", "p"]
