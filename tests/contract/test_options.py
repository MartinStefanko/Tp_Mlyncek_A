import pytest
from pydantic import ValidationError

from qgen.contract import (
    JobOptions,
    LessonsSplit,
    OptionDefaults,
    SingleSplit,
    SyllabusSplit,
    TopicsSplit,
    resolve_options,
)

# Example from Príloha A – Vstupné parametre (options).
APPENDIX_EXAMPLE = {
    "external_ref": "gen-01JB7Q3K",
    "callback_url": "http://eceh-backend/api/internal/qgen/webhook",
    "language": "auto",
    "extraction": {"ocr": False},
    "questions": {
        "types": ["choice", "open", "no_answer"],
        "per_section": 3,
        "choice_options": 4,
        "hints": 2,
    },
    "split": {"mode": "topics", "lesson_count": None, "lesson_minutes": 45, "syllabus": None, "syllabus_text": None},
}


def test_missing_values_are_none_except_split():
    options = JobOptions()
    assert options.external_ref is None
    assert options.callback_url is None
    assert options.language is None
    assert options.extraction is None
    assert options.questions is None
    assert isinstance(options.split, SingleSplit)
    assert options.split.lesson_minutes is None


def test_appendix_example_passes():
    options = JobOptions.model_validate(APPENDIX_EXAMPLE)
    assert isinstance(options.split, TopicsSplit)
    assert options.split.lesson_minutes == 45
    assert str(options.callback_url) == APPENDIX_EXAMPLE["callback_url"]


@pytest.mark.parametrize(
    ("split", "cls"),
    [
        # Examples from kap. 8.
        ({"mode": "single"}, SingleSplit),
        ({"mode": "lessons", "lesson_count": 6, "lesson_minutes": 45}, LessonsSplit),
        ({"mode": "topics", "lesson_minutes": 45}, TopicsSplit),
        (
            {
                "mode": "syllabus",
                "syllabus": [
                    {"title": "Jednoduchý úrok a diskont", "pages": "D1:1-10"},
                    {"title": "Zložené úrokovanie", "description": "efektívna úroková miera, ekvivalencia sadzieb"},
                    {"title": "Anuity a perpetuita"},
                ],
            },
            SyllabusSplit,
        ),
        ({"mode": "lessons", "lesson_count": 3}, LessonsSplit),
        (
            {
                "mode": "syllabus",
                "syllabus_text": "3 hodiny: najprv teória úrokovania, potom príklady, perpetuity vynechaj",
            },
            SyllabusSplit,
        ),
        ({"mode": "syllabus", "syllabus": [{"title": "A", "pages": "D1:1-10, D2:3"}], "syllabus_text": None}, SyllabusSplit),
        ({"mode": "topics", "syllabus_text": None}, TopicsSplit),
        ({"mode": "lessons", "lesson_minutes": 90}, LessonsSplit),
        ({"mode": "topics"}, TopicsSplit),
    ],
)
def test_valid_split_modes(split, cls):
    assert isinstance(JobOptions.model_validate({"split": split}).split, cls)


@pytest.mark.parametrize(
    "split",
    [
        {"mode": "syllabus"},
        {"mode": "syllabus", "syllabus": None},
        {"mode": "syllabus", "syllabus": []},
        {"mode": "syllabus", "syllabus": [{"description": "bez názvu"}]},
        {"mode": "syllabus", "syllabus": [{"title": "A", "pages": "1-10"}]},
        {"mode": "syllabus", "syllabus": [{"title": "A", "pages": "D1:10-3"}]},
        {"mode": "syllabus", "syllabus": [{"title": "A", "pages": "D0:1"}]},
        {"mode": "syllabus", "syllabus": [{"title": "A", "pages": "D1:0-4"}]},
        {"mode": "syllabus", "syllabus": [{"title": "A"}], "syllabus_text": "2 hodiny"},
        {"mode": "syllabus", "syllabus_text": None},
        {"mode": "syllabus", "syllabus_text": ""},
        {"mode": "syllabus", "syllabus_text": "   "},
        {"mode": "syllabus", "syllabus_text": "x" * 4001},
        {"mode": "syllabus", "syllabus": None, "syllabus_text": None},
        {"mode": "topics", "syllabus_text": "2 hodiny"},
        {"mode": "lessons", "lesson_count": 2, "syllabus_text": "2 hodiny"},
        {"mode": "single", "syllabus_text": "2 hodiny"},
        {"mode": "lessons"},
        {"mode": "lessons", "lesson_count": None, "lesson_minutes": None},
        {"mode": "lessons", "lesson_count": 0},
        {"mode": "lessons", "lesson_count": 101},
        {"mode": "lessons", "lesson_minutes": 0},
        {"mode": "lessons", "lesson_minutes": 241},
        {"mode": "topics", "lesson_count": 4},
        {"mode": "topics", "syllabus": [{"title": "A"}]},
        {"mode": "single", "lesson_count": 2},
        {"mode": "chapters"},
        {"lesson_count": 3},
        {"mode": "topics", "unknown": 1},
    ],
)
def test_invalid_split_rejected(split):
    with pytest.raises(ValidationError):
        JobOptions.model_validate({"split": split})


@pytest.mark.parametrize(
    "options",
    [
        {"language": "de"},
        {"external_ref": ""},
        {"external_ref": "x" * 256},
        {"callback_url": "ftp://eceh-backend/hook"},
        {"callback_url": "not a url"},
        {"extraction": {"ocr": False, "dpi": 300}},
        {"questions": {"types": []}},
        {"questions": {"types": ["essay"]}},
        {"questions": {"types": ["open", "open"]}},
        {"questions": {"per_section": 0}},
        {"questions": {"per_section": 11}},
        {"questions": {"per_learning_object": 3}},
        {"questions": {"choice_options": 1}},
        {"questions": {"choice_options": 11}},
        {"questions": {"hints": -1}},
        {"questions": {"hints": 3}},
        {"unknown_option": True},
    ],
)
def test_invalid_options_rejected(options):
    with pytest.raises(ValidationError):
        JobOptions.model_validate(options)


# --- resolve_options ------------------------------------------------------------------------

SPEC_DEFAULTS = OptionDefaults()


def test_spec_defaults():
    assert SPEC_DEFAULTS == OptionDefaults(
        language="auto",
        ocr=False,
        question_types=("choice", "open", "no_answer"),
        per_section=3,
        choice_options=4,
        hints=2,
        lesson_minutes=45,
    )


def test_resolve_empty_options():
    resolved = resolve_options(JobOptions(), SPEC_DEFAULTS)
    assert resolved.model_dump(mode="json") == {
        "external_ref": None,
        "callback_url": None,
        "language": "auto",
        "extraction": {"ocr": False},
        "questions": {"types": ["choice", "open", "no_answer"], "per_section": 3, "choice_options": 4, "hints": 2},
        "split": {
            "mode": "single",
            "lesson_count": None,
            "lesson_minutes": 45,
            "syllabus": None,
            "syllabus_text": None,
        },
    }


def test_resolve_keeps_sent_values():
    resolved = resolve_options(JobOptions.model_validate(APPENDIX_EXAMPLE), SPEC_DEFAULTS)
    assert resolved.model_dump(mode="json") == APPENDIX_EXAMPLE


def test_resolve_partial_questions_fills_the_rest():
    resolved = resolve_options(JobOptions.model_validate({"questions": {"types": ["open"], "hints": 0}}), SPEC_DEFAULTS)
    assert resolved.questions.types == ["open"]
    assert resolved.questions.hints == 0
    assert resolved.questions.per_section == 3
    assert resolved.questions.choice_options == 4


def test_resolve_uses_given_defaults():
    defaults = OptionDefaults(language="sk", ocr=True, question_types=("open",), per_section=5, lesson_minutes=90)
    resolved = resolve_options(JobOptions.model_validate({"split": {"mode": "lessons", "lesson_count": 4}}), defaults)
    assert resolved.language == "sk"
    assert resolved.extraction.ocr is True
    assert resolved.questions.types == ["open"]
    assert resolved.questions.per_section == 5
    assert isinstance(resolved.split, LessonsSplit)
    assert (resolved.split.lesson_count, resolved.split.lesson_minutes) == (4, 90)


def test_resolve_does_not_override_explicit_false_or_lesson_minutes():
    options = JobOptions.model_validate({"extraction": {"ocr": False}, "split": {"mode": "topics", "lesson_minutes": 30}})
    resolved = resolve_options(options, OptionDefaults(ocr=True))
    assert resolved.extraction.ocr is False
    assert resolved.split.lesson_minutes == 30


def test_resolved_options_survive_json_round_trip():
    resolved = resolve_options(JobOptions.model_validate(APPENDIX_EXAMPLE), SPEC_DEFAULTS)
    assert JobOptions.model_validate_json(resolved.model_dump_json()) == resolved


def test_resolve_rejects_invalid_defaults():
    with pytest.raises(ValidationError):
        resolve_options(JobOptions(), OptionDefaults(lesson_minutes=0))


def test_syllabus_text_survives_resolve():
    text = "5 hodín, perpetuity vynechaj"
    resolved = resolve_options(JobOptions.model_validate({"split": {"mode": "syllabus", "syllabus_text": text}}), SPEC_DEFAULTS)
    assert (resolved.split.syllabus, resolved.split.syllabus_text, resolved.split.lesson_minutes) == (None, text, 45)
