"""Step 13c: one hint per question with the source document and pages, made by the code.

FakeLLM only; no real API.
"""

import re

import pytest

from qgen.config import Settings
from qgen.context_builder import build_allowed_source_refs, build_source_name_map
from qgen.contract.units import HINT_NAME_MAX, HINT_URL_MAX, Question
from qgen.llm import retry
from qgen.questions.hints import build_source_hints, page_runs
from qgen.questions.pipeline import generate_all_items
from qgen.questions.prompts import (
    build_coverage_section,
    build_extra_round_section,
    build_item_evaluation_prompt,
    build_item_generation_prompt,
    evaluation_items_text,
)
from qgen.questions.types import QuestionSettings, contract_part, normalize_generated_item
from tests.fakes import (
    EMPTY_ITEMS,
    FakeEmbedder,
    FakeLLM,
    OrthogonalVectors,
    check_answer,
    is_question_generation,
    items,
    question,
    section_ids,
)


def _segments(source_id, name, pages):
    return [{"page": p, "source_id": source_id, "source_name": name, "text": f"Strana {p}"} for p in pages]


SEGMENTS = _segments("D1", "eko.pdf", range(1, 40)) + _segments("D2", "uvod.pdf", range(1, 10))
NAMES = build_source_name_map(SEGMENTS)
ALLOWED = build_allowed_source_refs(SEGMENTS)
SETTINGS = Settings.model_construct()


def hint(source, section_sources=(), *, language="sk", count=1, segments=None, settings=SETTINGS):
    names = build_source_name_map(segments) if segments else NAMES
    allowed = build_allowed_source_refs(segments) if segments else ALLOWED
    return build_source_hints(
        {"source": list(source)}, {"citovane_zdroje": list(section_sources)},
        source_name_map=names, allowed_refs=allowed, language=language, count=count, settings=settings,
    )


@pytest.mark.parametrize("source, url", [
    (["D1:14"], "<p>eko.pdf, strana 14</p>"),
    (["D1:14", "D1:15"], "<p>eko.pdf, strany 14 – 15</p>"),
    (["D1:16", "D1:14"], "<p>eko.pdf, strany 14, 16</p>"),
    (["D1:3", "D1:9", "D1:4", "D1:5", "D1:3"], "<p>eko.pdf, strany 3 – 5, 9</p>"),
    (["D2:3", "D1:4", "D2:2"], "<p>eko.pdf, strana 4; uvod.pdf, strany 2 – 3</p>"),
])
def test_format_sk(source, url):
    assert hint(source) == [{"name": "Zdroj", "url": url}]


@pytest.mark.parametrize("source, url", [
    (["D1:14"], "<p>eko.pdf, page 14</p>"),
    (["D1:14", "D1:15"], "<p>eko.pdf, pages 14 – 15</p>"),
    (["D1:14", "D1:16"], "<p>eko.pdf, pages 14, 16</p>"),
])
def test_format_en(source, url):
    assert hint(source, language="en") == [{"name": "Source", "url": url}]


def test_document_name_is_the_uploaded_name_not_d1():
    segments = _segments("D1", "Ekonómia – úvod.pdf", range(1, 5))
    assert hint(["D1:2"], segments=segments)[0]["url"] == "<p>Ekonómia – úvod.pdf, strana 2</p>"
    # one document: bare page numbers are pages of that document, merged with "D1:n"
    assert hint(["2", "D1:3"], segments=segments)[0]["url"] == "<p>Ekonómia – úvod.pdf, strany 2 – 3</p>"


def test_html_of_the_file_name_is_escaped():
    segments = _segments("D1", "a&b<c>.pdf", [1])
    assert hint(["D1:1"], segments=segments)[0]["url"] == "<p>a&amp;b&lt;c&gt;.pdf, strana 1</p>"


def test_fallback_to_the_pages_of_the_section():
    # own source missing, unknown document or a page that does not exist -> section pages
    for source in ([], ["D1:999"], ["D7:3"], ["bez strany"]):
        assert hint(source, ["D1:20", "D1:21"])[0]["url"] == "<p>eko.pdf, strany 20 – 21</p>"
    # a valid own source wins
    assert hint(["D1:5"], ["D1:20"])[0]["url"] == "<p>eko.pdf, strana 5</p>"


def test_no_valid_pages_no_hint():
    assert hint([], []) == []
    assert hint(["D1:999"], ["D9:1"]) == []


def test_hints_option():
    assert hint(["D1:14"], count=0) == []
    assert len(hint(["D1:14"], count=2)) == 1  # >= 1: exactly one hint with the source


def test_long_page_list_is_shortened_within_the_ecech_limit():
    segments = _segments("D1", "eko.pdf", range(1, 3001)) + _segments("D2", "uvod.pdf", range(1, 3001))
    source = [f"D1:{p}" for p in range(1, 3001, 2)] + [f"D2:{p}" for p in range(1, 3001, 2)]
    (result,) = hint(source, segments=segments)
    assert len(result["url"]) <= HINT_URL_MAX
    assert result["url"].startswith("<p>eko.pdf, strany 1, 3, 5, ")
    assert result["url"].endswith(", …</p>")
    assert "uvod.pdf, strany 1, 3, " in result["url"]
    Question.model_validate({"name": "q", "description": "d", "type": "no_answer", "time_to_handle": 0,
                             "time_to_explain": 0, "hints": [result], "choices": []})


def test_many_documents_and_a_huge_file_name_fit():
    segments = []
    for i in range(1, 11):
        segments += _segments(f"D{i}", f"{'dlhy_nazov_' * 25}{i}.pdf", range(1, 400))
    source = [f"D{i}:{p}" for i in range(1, 11) for p in range(1, 400, 2)]
    url = hint(source, segments=segments)[0]["url"]
    assert len(url) <= HINT_URL_MAX and "…" in url
    huge = _segments("D1", "x" * 3000 + ".pdf", [1])
    url = hint(["D1:1"], segments=huge)[0]["url"]
    assert len(url) <= HINT_URL_MAX and url.endswith("…</p>")


def test_templates_from_the_configuration_and_name_limit():
    settings = Settings.model_construct(
        hint_name_sk="Pozri si zdroj tejto otázky v pôvodnom dokumente, prosím pekne",
        hint_page_sk="Pozri: {file}, s. {pages}", hint_pages_sk="Pozri: {file}, s. {pages}",
    )
    (result,) = hint(["D1:14", "D1:15"], settings=settings)
    assert result["url"] == "<p>Pozri: eko.pdf, s. 14 – 15</p>"
    assert len(result["name"]) <= HINT_NAME_MAX and result["name"].startswith("Pozri si zdroj")


def test_template_without_placeholders_is_rejected():
    with pytest.raises(ValueError):
        Settings(hint_page_sk="strana {pages}")


def test_page_runs():
    assert page_runs([5, 1, 2, 3, 9, 10, 2]) == [(1, 3), (5, 5), (9, 10)]


# --- prompts --------------------------------------------------------------------------------

HINT_WORDS = re.compile(r"hint|nápoved|napoved", re.IGNORECASE)


@pytest.mark.parametrize("language", ["sk", "en"])
def test_prompts_say_nothing_about_hints(language):
    record = normalize_generated_item(question(1, 1), {"vzdelávací_objekt": "x"}, QuestionSettings())[0]
    record = {**record, "id": 1, "hints": [{"name": "Zdroj", "url": "<p>eko.pdf, strana 1</p>"}]}
    existing = {1: [record]}
    for hints in (0, 1, 2):
        qs = QuestionSettings(hints=hints)
        for extra in ("", build_extra_round_section(existing, language), build_coverage_section(existing, {1: 2}, language)):
            prompt = build_item_generation_prompt("blok", language, qs, SETTINGS, extra)
            assert not HINT_WORDS.search(prompt), prompt
    evaluation = build_item_evaluation_prompt(evaluation_items_text([record], language), language)
    assert not HINT_WORDS.search(evaluation), evaluation
    assert "eko.pdf" not in evaluation


# --- whole question pipeline ----------------------------------------------------------------

PIPELINE_SEGMENTS = _segments("D1", "bio.pdf", range(1, 4)) + _segments("D2", "chem.pdf", range(1, 3))
PIPELINE_SECTIONS = [
    {"id": 1, "vzdelávací_objekt": "Fotosyntéza", "bloom_level": "Pochopiť", "citovane_zdroje": ["D1:1", "D1:2"]},
    {"id": 2, "vzdelávací_objekt": "Reakcie", "bloom_level": "Pochopiť", "citovane_zdroje": ["D2:1"]},
]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(retry, "_sleep", lambda seconds: None)


def _model(prompt, model):
    if is_question_generation(prompt):
        if "DOPLŇUJÚCE KOLO" in prompt or "DOPLNENIE POKRYTIA" in prompt:
            return EMPTY_ITEMS
        rows = []
        for lo_id in section_ids(prompt):
            if lo_id == 1:
                rows.append(question(1, 1, source=["D1:1", "D1:2", "D2:2"], hints=["<p>Stará LLM nápoveda</p>"]))
            else:
                rows.append(question(2, 1, source=["D2:1"]))
        return items(rows)
    return check_answer(prompt) or EMPTY_ITEMS


@pytest.mark.parametrize("hints, expected", [
    (2, [[{"name": "Zdroj", "url": "<p>bio.pdf, strany 1 – 2; chem.pdf, strana 2</p>"}],
         [{"name": "Zdroj", "url": "<p>chem.pdf, strana 1</p>"}]]),
    (0, [[], []]),
])
def test_pipeline_builds_the_source_hint_and_passes_the_contract(hints, expected):
    from qgen.contract.options import QuestionOptions

    llm = FakeLLM(default=_model)
    result = generate_all_items(
        PIPELINE_SECTIONS, PIPELINE_SEGMENTS, llm=llm, embedder=FakeEmbedder(default=OrthogonalVectors()),
        verbose=False, document_language="sk", per_section=1,
        question_options=QuestionOptions(hints=hints),
        settings=Settings.model_construct(question_generation_batch_size=10, question_extra_rounds=0),
    )
    assert [q["hints"] for q in result.questions] == expected
    assert [q["hints"] for q in result.all_questions] == expected
    for item in result.questions:
        Question.model_validate(contract_part(item))
    diagnostics = result.reports["question_diagnostics"]
    assert diagnostics["validation_warnings"] == {}
