import pytest
from pydantic import ValidationError

from qgen.contract import JobOptions, LessonsSplit, SingleSplit, SyllabusSplit, TopicsSplit

# Example from Príloha A – Vstupné parametre (options).
APPENDIX_EXAMPLE = {
    "external_ref": "gen-01JB7Q3K",
    "callback_url": "http://eceh-backend/api/internal/qgen/webhook",
    "language": "auto",
    "extraction": {"ocr": False},
    "questions": {
        "types": ["choice", "open", "no_answer"],
        "per_learning_object": 3,
        "choice_options": 4,
        "hints": 2,
    },
    "split": {"mode": "topics", "lesson_count": None, "lesson_minutes": 45, "syllabus": None},
}


def test_defaults():
    options = JobOptions()
    assert options.external_ref is None
    assert options.callback_url is None
    assert options.language == "auto"
    assert options.extraction.ocr is False
    assert options.questions.types == ["choice", "open", "no_answer"]
    assert options.questions.per_learning_object == 3
    assert options.questions.choice_options == 4
    assert options.questions.hints == 2
    assert isinstance(options.split, SingleSplit)


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
        {"questions": {"per_learning_object": 0}},
        {"questions": {"per_learning_object": 11}},
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


def test_partial_questions_keep_other_defaults():
    options = JobOptions.model_validate({"questions": {"types": ["open"]}})
    assert options.questions.types == ["open"]
    assert options.questions.per_learning_object == 3
    assert options.questions.hints == 2
