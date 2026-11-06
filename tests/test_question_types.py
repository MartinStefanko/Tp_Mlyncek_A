"""Step 13: question types choice / open / no_answer, extra rounds, duplicates, rules per
type, hints, times, choice marking, split of a cut-off answer, progress. FakeLLM only."""

import json
import re

import pytest

from qgen.config import Settings
from qgen.contract.options import QuestionOptions
from qgen.llm import retry
from qgen.llm.adapters.replay import QUESTION_PROMPT_MARKERS
from qgen.llm.errors import LLMError
from qgen.questions.pipeline import generate_all_items
from qgen.questions.prompts import GENERATION_MARKERS, build_item_generation_prompt
from qgen.questions.types import (
    QuestionSettings,
    clamp_times,
    contract_part,
    normalize_generated_item,
    type_rules_text,
)
from qgen.questions.validation import validate_items
from qgen.contract.units import Question
from tests.fakes import (
    EMPTY_ITEMS,
    FakeEmbedder,
    FakeLLM,
    OrthogonalVectors,
    check_answer,
    is_extra_round,
    is_question_generation,
    items,
    question,
    section_ids,
    truncated,
)

SEGMENTS = [
    {"page": p, "source_id": "D1", "source_name": "a.pdf", "text": f"Učivo strany {p}: pojem {p} a jeho vlastnosti."}
    for p in range(1, 5)
]
SECTIONS = [
    {"id": 1, "vzdelávací_objekt": "Pojem jedna", "bloom_level": "Pochopiť", "citovane_zdroje": ["D1:1"]},
    {"id": 2, "vzdelávací_objekt": "Pojem dva", "bloom_level": "Aplikovať", "citovane_zdroje": ["D1:2"]},
    {"id": 3, "vzdelávací_objekt": "Pojem tri", "bloom_level": "Hodnotiť", "citovane_zdroje": ["D1:3"]},
    {"id": 4, "vzdelávací_objekt": "Pojem štyri", "bloom_level": "Zapamätať si", "citovane_zdroje": ["D1:4"]},
]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(retry, "_sleep", lambda seconds: None)


def _settings(**values):
    defaults = {"question_generation_batch_size": 10, "question_extra_rounds": 0}
    defaults.update(values)
    return Settings.model_construct(**defaults)


class Model:
    """first(ids, prompt) answers the first call, extra(ids, round_no, prompt) the extra
    rounds; coverage calls answer nothing; checks give 5 (or `check` overrides)."""

    def __init__(self, first, extra=None, check=None):
        self.first, self.extra, self.check = first, extra, check
        self.prompts, self.extra_prompts = [], []

    def __call__(self, prompt, model):
        if is_question_generation(prompt):
            self.prompts.append(prompt)
            if is_extra_round(prompt):
                self.extra_prompts.append(prompt)
                if self.extra is None:
                    return EMPTY_ITEMS
                return self.extra(section_ids(prompt), len(self.extra_prompts), prompt)
            if "DOPLNENIE POKRYTIA" in prompt:
                return EMPTY_ITEMS
            return self.first(section_ids(prompt), prompt)
        if self.check:
            answer = self.check(prompt)
            if answer is not None:
                return answer
        return check_answer(prompt) or EMPTY_ITEMS


def _run(model, settings=None, embed=None, sections=SECTIONS, **kwargs):
    kwargs.setdefault("per_section", 1)
    result = generate_all_items(
        sections, SEGMENTS, llm=FakeLLM(default=model),
        embedder=FakeEmbedder(default=embed or OrthogonalVectors()), verbose=False,
        document_language=kwargs.pop("language", "sk"), settings=settings or _settings(), **kwargs,
    )
    return result, result.reports["question_diagnostics"]


# --- prompts and type rules -----------------------------------------------------------------


def test_default_type_rules_prefer_choice():
    text = type_rules_text(QuestionSettings(), _settings())
    assert 'Povolené typy sú iba "choice", "open" a "no_answer".' in text
    assert 'Prednostný typ je "choice": aspoň 60 % otázok' in text
    assert "bloom_level Aplikovať, Analyzovať, Hodnotiť alebo Vytvoriť má aj aspoň jednu praktickú úlohu" in text
    assert '"no_answer" použi iba pri otázke s bloom_level Hodnotiť alebo Vytvoriť.' in text


def test_type_rules_follow_the_config():
    settings = _settings(question_choice_min_share=0.75, question_no_answer_bloom="create",
                         question_practical_bloom="analyze")
    text = type_rules_text(QuestionSettings(), settings, "en")
    assert "at least 75 % of the questions" in text
    assert "bloom_level Analyzovať also gets at least one practical task" in text
    assert 'Use "no_answer" only for a question with bloom_level Vytvoriť.' in text


def test_only_choice_allowed_gives_a_choice_only_prompt():
    qs = QuestionSettings.from_options(QuestionOptions(types=["choice"]))
    text = type_rules_text(qs, _settings())
    assert text == '- Povolený je iba typ "choice". Všetky otázky sú "choice", aj výpočty a postupy (výsledky sú možnosti).'
    prompt = build_item_generation_prompt("x", "sk", qs, _settings())
    assert "praktickú úlohu: typ \"open\"" not in prompt


def test_types_without_open_and_no_answer():
    qs = QuestionSettings.from_options(QuestionOptions(types=["no_answer", "choice"]))
    assert qs.types == ("choice", "no_answer")  # contract order
    text = type_rules_text(qs, _settings())
    assert '"practical": true' in text and "ako \"choice\" s výsledkami" in text
    text = type_rules_text(QuestionSettings.from_options(QuestionOptions(types=["choice", "open"])), _settings())
    assert 'Typ "no_answer" nepoužívaj.' in text


def test_prompt_uses_options_and_never_refers_to_pages():
    qs = QuestionSettings(choice_options=5, hints=1)
    prompt = build_item_generation_prompt("x", "sk", qs, _settings())
    assert 'v "choices" presne 5 možností' in prompt
    assert 'presne 1 nápovedu' in prompt
    assert "nesmie odkazovať na strany, kapitoly ani časti dokumentu" in prompt
    assert "„všetky uvedené“ ani „žiadna z uvedených“" in prompt
    none = build_item_generation_prompt("x", "en", QuestionSettings(hints=0), _settings())
    assert '"hints" is an empty array [].' in none and "must not reveal the answer" not in none


def test_replay_markers_match_the_prompts():
    assert QUESTION_PROMPT_MARKERS == GENERATION_MARKERS
    for language, marker in zip(("sk", "en"), GENERATION_MARKERS):
        assert marker in build_item_generation_prompt("x", language)


# --- ratio of types in the pipeline ------------------------------------------------------------


def _mixed(ids, prompt):
    rows = []
    for i in ids:
        rows += [question(i, 1), question(i, 2), question(i, 3, "open"),
                 question(i, 4, "no_answer", bloom_level="Hodnotiť"), question(i, 5, "no_answer")]
    return items(rows)


def test_type_rules_are_enforced_on_the_answer():
    result, diagnostics = _run(Model(_mixed))
    types = diagnostics["question_types"]["accepted"]
    # no_answer at Pochopiť (question 5) is dropped; no_answer at Hodnotiť (question 4) stays
    assert types == {"choice": 8, "open": 4, "no_answer": 4}
    assert diagnostics["items_dropped"]["by_reason"]["type_rule"] == 4
    assert all(q["bloom_level"] == "evaluate" for q in result.questions if q["type"] == "no_answer")


def test_only_choice_drops_the_other_types():
    model = Model(_mixed)
    result, diagnostics = _run(model, question_options=QuestionOptions(types=["choice"]))
    assert {q["type"] for q in result.questions} == {"choice"}
    assert diagnostics["items_dropped"]["by_reason"]["type_not_allowed"] == 12
    assert 'Povolený je iba typ "choice"' in model.prompts[0]
    assert diagnostics["question_options"]["types"] == ["choice"]


def test_only_no_answer_lifts_the_bloom_rule():
    result, diagnostics = _run(Model(_mixed), question_options=QuestionOptions(types=["no_answer"]))
    assert diagnostics["question_types"]["accepted"]["no_answer"] == 8
    assert diagnostics["items_dropped"]["by_reason"]["type_rule"] == 0


def test_low_choice_share_is_a_warning():
    open_only = lambda ids, prompt: items(question(i, n, "open") for i in ids for n in (1, 2))
    result, diagnostics = _run(Model(open_only))
    assert len(result.questions) == 8
    assert [w["code"] for w in diagnostics["warnings"]] == ["choice_share_low"]


# --- extra rounds ---------------------------------------------------------------------------------


def _round_answers(ids, round_no, prompt):
    return items(question(i, 10 * round_no + n) for i in ids for n in (1, 2))


def test_one_extra_round_by_default_with_the_existing_questions():
    model = Model(lambda ids, prompt: items(question(i, 1) for i in ids), extra=_round_answers)
    result, diagnostics = _run(model, _settings(question_extra_rounds=1))
    assert len(model.extra_prompts) == 1
    extra = model.extra_prompts[0]
    assert "Vytvor ďalšie otázky, ktoré sa od nich líšia obsahom alebo uhlom pohľadu" in extra
    assert "ak pre tematický celok už nič nové nie je" in extra.lower()
    assert '{"items": []}' in extra
    assert "- lo_id 1:\n  1. [choice] Otázka 1 k celku 1?" in extra
    assert len(result.questions) == 4 * 3
    assert diagnostics["extra_rounds"]["rounds"] == [
        {"round": 1, "calls": 1, "sections": 4, "questions_added": 8, "sections_continuing": 4}]
    assert diagnostics["questions_by_origin"] == {"initial": 4, "extra_1": 8}
    assert diagnostics["generation_calls"] == {"initial": 1, "extra": 1, "coverage": 0}


def test_extra_rounds_off():
    model = Model(lambda ids, prompt: items(question(i, 1) for i in ids), extra=_round_answers)
    _, diagnostics = _run(model, _settings(question_extra_rounds=0))
    assert model.extra_prompts == [] and diagnostics["extra_rounds"]["rounds_run"] == 0


def test_three_rounds_when_every_round_brings_something():
    model = Model(lambda ids, prompt: items(question(i, 1) for i in ids), extra=_round_answers)
    result, diagnostics = _run(model, _settings(question_extra_rounds=3))
    assert len(model.extra_prompts) == 3
    assert diagnostics["extra_rounds"]["rounds_run"] == 3
    assert len(result.questions) == 4 * (1 + 3 * 2)


def test_rounds_stop_for_a_section_without_anything_new():
    def extra(ids, round_no, prompt):
        # round 1: only section 1 and 2 get something; round 2: nothing at all
        if round_no == 1:
            return items(question(i, 11) for i in ids if i in (1, 2))
        return EMPTY_ITEMS

    model = Model(lambda ids, prompt: items(question(i, 1) for i in ids), extra=extra)
    _, diagnostics = _run(model, _settings(question_extra_rounds=5))
    assert [section_ids(p) for p in model.extra_prompts] == [[1, 2, 3, 4], [1, 2]]
    assert diagnostics["extra_rounds"]["rounds_run"] == 2


def test_round_with_only_duplicates_ends_the_rounds():
    same = {}

    def extra(ids, round_no, prompt):
        rows = [question(i, 1, description=f"Iná formulácia {i}?") for i in ids]
        for i in ids:  # the embedder sees the new wording as the old one
            same[f"Iná formulácia {i}? Správna 1/{i}"] = f"Otázka 1 k celku {i}? Správna 1/{i}"
        return items(rows)

    model = Model(lambda ids, prompt: items(question(i, 1) for i in ids), extra=extra)
    result, diagnostics = _run(model, _settings(question_extra_rounds=3), embed=OrthogonalVectors(same=same))
    assert len(model.extra_prompts) == 1
    assert diagnostics["duplicates_removed"] == {"total": 4, "by_origin": {"extra_1": 4}}
    assert diagnostics["items_dropped"]["by_reason"]["duplicate"] == 4
    assert len(result.questions) == 4


# --- duplicates -------------------------------------------------------------------------------------


def test_duplicates_are_removed_only_within_a_section():
    rows = [question(1, 1), question(1, 2, description="Otázka 1 k celku 1?"),
            question(2, 1, description="Otázka 1 k celku 1?")]
    # same text of question 1/1 and 1/2 -> same vector; section 2 has the same text too
    same = {"Otázka 1 k celku 1? Správna 2/1": "Otázka 1 k celku 1? Správna 1/1",
            "Otázka 1 k celku 1? Správna 1/2": "Otázka 1 k celku 1? Správna 1/1"}
    result, diagnostics = _run(Model(lambda ids, prompt: items(rows)), embed=OrthogonalVectors(same=same),
                               sections=SECTIONS[:2])
    assert [(q["lo_id"], q["choices"][0]["text"]) for q in result.questions] == [(1, "Správna 1/1"), (2, "Správna 1/2")]
    assert diagnostics["duplicates_removed"]["total"] == 1
    assert diagnostics["dedup_threshold"] == 0.94
    dropped = [q for q in result.all_questions if q["drop_reason"] == "duplicate"]
    assert [q["choices"][0]["text"] for q in dropped] == ["Správna 2/1"]


def test_threshold_from_config():
    rows = [question(1, 1), question(1, 2)]
    near = lambda texts, model: [[1.0, 0.0] if "1/1" in t else [0.9, 0.436] for t in texts]  # cosine 0.9
    _, kept = _run(Model(lambda ids, prompt: items(rows)), _settings(question_dedup_threshold=0.95),
                   embed=near, sections=SECTIONS[:1])
    _, dropped = _run(Model(lambda ids, prompt: items(rows)), _settings(question_dedup_threshold=0.85),
                      embed=near, sections=SECTIONS[:1])
    assert kept["duplicates_removed"]["total"] == 0 and dropped["duplicates_removed"]["total"] == 1


# --- rules per type (validation) ---------------------------------------------------------------------


def _validated(**fields):
    raw = question(1, 1, **fields)
    item, _ = normalize_generated_item(raw, SECTIONS[0], QuestionSettings())
    return validate_items([{"id": 1, **item}], allowed_pages=["D1:1"], valid_lo_ids={1})


@pytest.mark.parametrize("choices, message", [
    ([{"text": "A", "is_correct": True}], "at least 2 choices"),
    ([{"text": "A", "is_correct": False}, {"text": "B", "is_correct": False}], "at least 1 correct"),
    ([{"text": "A", "is_correct": True}, {"text": " a ", "is_correct": False}], "duplicate choice text"),
    ([{"text": "A", "is_correct": True}, {"text": "Žiadna z uvedených možností", "is_correct": False}], "vsetky/ziadna"),
    ([{"text": "A", "is_correct": True}, {"text": "None of the above", "is_correct": False}], "vsetky/ziadna"),
    ([{"text": "A", "is_correct": "true"}, {"text": "B", "is_correct": False}], "nie je true/false"),
])
def test_choice_rules(choices, message):
    report = _validated(choices=choices)
    assert not report["is_valid"] and any(message in e for e in report["errors"]), report["errors"]


def test_choice_with_more_correct_options_is_valid_and_count_mismatch_is_a_warning():
    choices = [{"text": "A", "is_correct": True}, {"text": "B", "is_correct": True}, {"text": "C", "is_correct": False}]
    report = _validated(choices=choices)
    assert report["is_valid"] and report["warning_counts"] == {"choice_count_mismatch": 1}


@pytest.mark.parametrize("question_type, choices", [
    ("open", [{"text": "A", "is_correct": True}, {"text": "B", "is_correct": True}]),
    ("open", [{"text": "A", "is_correct": False}]),
    ("no_answer", [{"text": "A", "is_correct": True}]),
    ("essay", []),
])
def test_open_and_no_answer_rules(question_type, choices):
    assert not _validated(type=question_type, choices=choices)["is_valid"]


def test_rules_are_the_contract_rules():
    item, _ = normalize_generated_item(question(1, 1), SECTIONS[0], QuestionSettings())
    Question.model_validate(contract_part(item))  # a valid pipeline question is a valid contract question
    assert list(contract_part(item)) == list(Question.model_fields)


# --- hints, times, Bloom, keywords --------------------------------------------------------------------


def test_hints_are_html_with_level_names_and_trimmed():
    raw = question(1, 1, hints=["Zamerajte sa na <vzťah> medzi pojmami.", "<p>Druhá</p>", "<p>Tretia</p>"])
    item, _ = normalize_generated_item(raw, SECTIONS[0], QuestionSettings(hints=2))
    assert item["hints"] == [
        {"name": "Ľahká nápoveda", "url": "<p>Zamerajte sa na &lt;vzťah&gt; medzi pojmami.</p>"},
        {"name": "Ťažká nápoveda", "url": "<p>Druhá</p>"},
    ]
    item, _ = normalize_generated_item(raw, SECTIONS[0], QuestionSettings(hints=1), "en")
    assert [h["name"] for h in item["hints"]] == ["Hint"]
    item, _ = normalize_generated_item(raw, SECTIONS[0], QuestionSettings(hints=0))
    assert item["hints"] == []


@pytest.mark.parametrize("hint", ["<p>Pozrite si stranu 14.</p>", "<p>Pozri kapitolu 2.</p>", "<p>Strana 3</p>"])
def test_hint_with_a_page_reference_is_a_warning(hint):
    model = Model(lambda ids, prompt: items([question(1, 1, hints=[hint, "<p>ok</p>"])]))
    result, diagnostics = _run(model, sections=SECTIONS[:1])
    assert len(result.questions) == 1  # the question stays
    assert diagnostics["validation_warnings"] == {"hint_page_reference": 1}
    assert [w["code"] for w in diagnostics["warnings"]] == ["hint_page_reference"]


@pytest.mark.parametrize("question_type, raw, expected", [
    ("choice", (5, 500), (30, 90)),
    ("choice", (None, "x"), (120, 30)),
    ("open", (10_000, -5), (900, 0)),
    ("open", (None, None), (300, 0)),
    ("no_answer", (2400.4, 700), (1800, 600)),
    ("no_answer", ("600", "60"), (600, 60)),
])
def test_times_are_clamped_per_type(question_type, raw, expected):
    assert clamp_times(question_type, *raw)[:2] == expected


def test_clamped_times_are_counted():
    model = Model(lambda ids, prompt: items([question(1, 1, time_to_handle=1), question(1, 2)]))
    result, diagnostics = _run(model, sections=SECTIONS[:1])
    assert result.questions[0]["time_to_handle"] == 30 and diagnostics["times_clamped"] == 1


def test_bloom_and_keywords_from_the_llm_or_inherited_from_the_section():
    item, _ = normalize_generated_item(question(2, 1, bloom_level="Analyzovať", keywords=["a", "a", " b "]),
                                       SECTIONS[1], QuestionSettings())
    assert (item["bloom_level"], item["keywords"]) == ("analyze", ["a", "b"])
    item, _ = normalize_generated_item(question(2, 1, bloom_level="", keywords=[]), SECTIONS[1], QuestionSettings())
    assert (item["bloom_level"], item["keywords"]) == ("apply", ["Pojem dva"])
    item, _ = normalize_generated_item(question(2, 1, bloom_level="Zázrak", name=""), SECTIONS[1], QuestionSettings())
    assert item["bloom_level"] == "apply" and item["name"] == "Pojem dva"


# --- choice marking (faithfulness check) ---------------------------------------------------------------


def test_wrongly_marked_choice_is_dropped_with_a_warning():
    def check(prompt):
        if "ITEM ID:" in prompt and "Otázka a odpoveď" in prompt:
            rows = []
            for block in prompt.split("ITEM ID: ")[1:]:
                item_id = int(block.split("\n", 1)[0])
                wrong = "Otázka 2 k celku 1?" in block
                if "typ: choice" in block:
                    assert "[správna]" in block and "[nesprávna]" in block  # options with marks
                rows.append({"item_id": item_id, "skore": 5, "zdovodnenie": "ok",
                             "choices_ok": not wrong, "wrong_choices": ["B"] if wrong else []})
            return items(rows)
        if "ITEM ID:" in prompt:
            assert "[správna]" not in prompt  # answerability: options without marks
        return None

    model = Model(lambda ids, prompt: items([question(1, 1), question(1, 2), question(1, 3, "open")]), check=check)
    result, diagnostics = _run(model, sections=SECTIONS[:1])
    assert [q["description"] for q in result.questions] == ["Otázka 1 k celku 1?", "Otázka 3 k celku 1?"]
    assert diagnostics["items_dropped"]["by_reason"]["wrong_choice_marking"] == 1
    assert diagnostics["wrong_choice_marking"] == 1
    warning = next(w for w in diagnostics["warnings"] if w["code"] == "choice_marking_wrong")
    assert "option(s) B marked wrongly" in warning["message"]
    rows = result.reports["item_faithfulness"]["items"]
    assert [row["choices_ok"] for row in rows] == [True, False, True]


# --- output cut off at the token limit ------------------------------------------------------------------


def test_cut_off_answer_splits_the_batch():
    calls = []

    def answer(prompt, model):
        if is_question_generation(prompt) and not is_extra_round(prompt):
            ids = section_ids(prompt)
            calls.append(ids)
            if len(ids) > 1:
                raise truncated()
            return items([question(ids[0], 1)])
        return check_answer(prompt) or EMPTY_ITEMS

    result = generate_all_items(SECTIONS, SEGMENTS, llm=FakeLLM(default=answer),
                                embedder=FakeEmbedder(default=OrthogonalVectors()), verbose=False,
                                document_language="sk", settings=_settings(question_generation_batch_size=4),
                                per_section=1)
    diagnostics = result.reports["question_diagnostics"]
    assert calls == [[1, 2, 3, 4], [1, 2], [1], [2], [3, 4], [3], [4]]
    assert diagnostics["batch_splits"] == 3
    assert len(result.questions) == 4


def test_single_section_that_does_not_fit_is_an_llm_error():
    def answer(prompt, model):
        if is_question_generation(prompt):
            raise truncated()
        return check_answer(prompt) or EMPTY_ITEMS

    with pytest.raises(LLMError) as info:
        generate_all_items(SECTIONS[:2], SEGMENTS, llm=FakeLLM(default=answer), embedder=FakeEmbedder(),
                           verbose=False, document_language="sk", settings=_settings(), per_section=1)
    assert info.value.truncated and info.value.role == "questions"


def test_generation_asks_for_the_output_token_limit():
    seen = []

    class Client(FakeLLM):
        def generate(self, prompt, *, model, json_mode=True, temperature=0.2, max_tokens=None):
            if is_question_generation(prompt):
                seen.append(max_tokens)
            return super().generate(prompt, model=model, json_mode=json_mode, temperature=temperature,
                                    max_tokens=max_tokens)

    model = Model(lambda ids, prompt: items(question(i, 1) for i in ids))
    generate_all_items(SECTIONS[:1], SEGMENTS, llm=Client(default=model), embedder=FakeEmbedder(), verbose=False,
                       document_language="sk", settings=_settings(llm_max_output_tokens=1234), per_section=1)
    assert seen and set(seen) == {1234}


def test_generation_batch_size_from_config():
    model = Model(lambda ids, prompt: items(question(i, 1) for i in ids))
    _run(model, Settings.model_construct(question_extra_rounds=0))  # default: 3 sections per call
    assert [section_ids(p) for p in model.prompts] == [[1, 2, 3], [4]]


# --- progress -------------------------------------------------------------------------------------------


def test_progress_is_well_formed_with_rounds_and_coverage():
    calls = []
    model = Model(lambda ids, prompt: items(question(i, 1) for i in ids if i != 4), extra=_round_answers)
    generate_all_items(SECTIONS, SEGMENTS, llm=FakeLLM(default=model), embedder=FakeEmbedder(default=OrthogonalVectors()),
                       verbose=False, document_language="sk",
                       settings=Settings.model_construct(question_extra_rounds=2),
                       per_section=1, progress_callback=lambda *c: calls.append(c))
    for stage in ("questions", "q_evaluation"):
        own = [(d, t) for s, d, t in calls if s == stage]
        assert own[0][0] == 0 and own[-1][0] == own[-1][1], (stage, own)
        assert [d for d, _ in own] == sorted(d for d, _ in own)
        assert all(0 <= d <= t for d, t in own)
    questions = [(d, t) for s, d, t in calls if s == "questions"]
    assert questions[0] == (0, 2 * 3)  # 2 batches x (1 + 2 rounds)
    assert len(questions) > 3
