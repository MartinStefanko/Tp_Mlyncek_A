import copy
import json
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from qgen.contract import Question, UnitsResult

FIXTURES = Path(__file__).parent / "fixtures"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "lesson.schema.json"


@pytest.fixture
def example() -> dict:
    return json.loads((FIXTURES / "units_example.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def schema_validator():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


def assert_rejected(data: dict, schema_validator, *, by_schema: bool = True) -> None:
    with pytest.raises(ValidationError):
        UnitsResult.model_validate(data)
    if by_schema:
        assert not schema_validator.is_valid(data)


# --- the example from kap. 6 -----------------------------------------------------------


def test_example_passes_model(example):
    result = UnitsResult.model_validate(example)
    assert len(result.units) == 2
    assert [q.type for q in result.units[0].questions] == ["choice", "open", "no_answer"]


def test_example_passes_json_schema(example, schema_validator):
    schema_validator.validate(example)


def test_round_trip_is_exact(example):
    dumped = UnitsResult.model_validate(example).model_dump(mode="json")
    assert dumped == example


# --- rejected outputs -----------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        (),                                   # top level
        ("units", 0),                         # unit
        ("units", 0, "questions", 0),         # question
        ("units", 0, "questions", 0, "hints", 0),
        ("units", 0, "questions", 0, "choices", 0),
    ],
)
def test_extra_field_rejected(example, schema_validator, path):
    data = copy.deepcopy(example)
    target = data
    for key in path:
        target = target[key]
    target["lo_id"] = "LO-1"
    assert_rejected(data, schema_validator)


@pytest.mark.parametrize(
    ("path", "field", "limit"),
    [
        (("units", 0), "title", 255),
        (("units", 0), "description", 4000),
        (("units", 0, "questions", 0), "name", 255),
        (("units", 0, "questions", 0), "description", 4000),
        (("units", 0, "questions", 0, "hints", 0), "name", 50),
        (("units", 0, "questions", 0, "hints", 0), "url", 2000),
        (("units", 0, "questions", 0, "choices", 0), "text", 4000),
    ],
)
def test_too_long_text_rejected(example, schema_validator, path, field, limit):
    data = copy.deepcopy(example)
    target = data
    for key in path:
        target = target[key]

    target[field] = "x" * limit
    UnitsResult.model_validate(data)  # exactly at the limit is fine
    schema_validator.validate(data)

    target[field] = "x" * (limit + 1)
    assert_rejected(data, schema_validator)


def test_choice_without_correct_answer_rejected(example, schema_validator):
    data = copy.deepcopy(example)
    for choice in data["units"][0]["questions"][0]["choices"]:
        choice["is_correct"] = False
    assert_rejected(data, schema_validator)


def test_choice_with_single_option_rejected(example, schema_validator):
    data = copy.deepcopy(example)
    question = data["units"][0]["questions"][0]
    question["choices"] = question["choices"][:1]
    assert_rejected(data, schema_validator)


def test_choice_with_duplicate_text_rejected(example):
    data = copy.deepcopy(example)
    choices = data["units"][0]["questions"][0]["choices"]
    choices[2]["text"] = "  je to súčet istiny a  prislúchajúceho úroku "
    # Not expressible in JSON Schema – only the model catches it.
    with pytest.raises(ValidationError, match="duplicate"):
        UnitsResult.model_validate(data)


def test_open_with_two_answers_rejected(example, schema_validator):
    data = copy.deepcopy(example)
    question = data["units"][0]["questions"][1]
    question["choices"].append({"text": "Iná vzorová odpoveď.", "is_correct": True})
    assert_rejected(data, schema_validator)


def test_open_with_incorrect_answer_rejected(example, schema_validator):
    data = copy.deepcopy(example)
    data["units"][0]["questions"][1]["choices"][0]["is_correct"] = False
    assert_rejected(data, schema_validator)


def test_open_without_answer_rejected(example, schema_validator):
    data = copy.deepcopy(example)
    data["units"][0]["questions"][1]["choices"] = []
    assert_rejected(data, schema_validator)


def test_no_answer_with_choices_rejected(example, schema_validator):
    data = copy.deepcopy(example)
    data["units"][0]["questions"][2]["choices"] = [{"text": "Odpoveď", "is_correct": True}]
    assert_rejected(data, schema_validator)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("type", "essay"),
        ("time_to_handle", -1),
        ("time_to_handle", 1.5),
        ("time_to_explain", "30"),
        ("time_to_explain", True),
    ],
)
def test_invalid_type_or_time_rejected(example, schema_validator, field, value):
    data = copy.deepcopy(example)
    data["units"][0]["questions"][0][field] = value
    # JSON Schema treats true as boolean, not integer, so all cases fail there too.
    assert_rejected(data, schema_validator)


def test_empty_units_rejected(schema_validator):
    assert_rejected({"units": []}, schema_validator)


def test_missing_field_rejected(example, schema_validator):
    data = copy.deepcopy(example)
    del data["units"][0]["questions"][0]["hints"]
    assert_rejected(data, schema_validator)


def test_question_model_directly():
    q = Question(
        name="N",
        description="D",
        type="no_answer",
        time_to_handle=0,
        time_to_explain=0,
        hints=[],
        choices=[],
    )
    assert q.model_dump()["choices"] == []


# --- the official example of the format from ECEH ----------------------------------------
# tests/contract/fixtures/units_official_example.json is the file delivered by ECEH, stored
# unchanged (also its test texts and its formatting).

OFFICIAL_EXAMPLE = FIXTURES / "units_official_example.json"


def test_official_example_passes_model_and_schema(schema_validator):
    data = json.loads(OFFICIAL_EXAMPLE.read_text(encoding="utf-8"))
    result = UnitsResult.model_validate(data)
    schema_validator.validate(data)
    assert [len(unit.questions) for unit in result.units] == [3, 4]
    assert [q.type for q in result.units[1].questions] == ["choice", "choice", "no_answer", "no_answer"]


def test_official_example_round_trip_gives_the_same_json():
    data = json.loads(OFFICIAL_EXAMPLE.read_text(encoding="utf-8"))
    serialized = UnitsResult.model_validate(data).model_dump_json()
    assert json.loads(serialized) == data
    # also key order: the model writes the fields in the order of the official example
    assert serialized == json.dumps(data, ensure_ascii=False, separators=(",", ":"))
