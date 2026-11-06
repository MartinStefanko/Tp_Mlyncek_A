"""Output format `units` (kap. 6) – GET /v1/jobs/{id}/result.

The format must match exactly: no extra fields anywhere (extra="forbid").
String limits are the "Limit v ECEH" column (DB column sizes in ECEH).
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

# Limits from the "Limit v ECEH" column (kap. 6).
UNIT_TITLE_MAX = 255          # units.title
UNIT_DESCRIPTION_MAX = 4000   # units.description
QUESTION_NAME_MAX = 255       # events.header
QUESTION_DESCRIPTION_MAX = 4000  # events.message
HINT_NAME_MAX = 50            # helps.name
HINT_URL_MAX = 2000           # helps.url (HTML text of the hint, not a web address)
CHOICE_TEXT_MAX = 4000        # options.answer_data

QuestionType = Literal["choice", "open", "no_answer"]

Seconds = Annotated[StrictInt, Field(ge=0, description="Time in seconds.")]


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Hint(_ContractModel):
    name: str = Field(min_length=1, max_length=HINT_NAME_MAX, description="Hint level, e.g. 'Ľahká nápoveda'.")
    url: str = Field(max_length=HINT_URL_MAX, description="HTML text of the hint, e.g. '<p>…</p>'.")


class Choice(_ContractModel):
    text: str = Field(min_length=1, max_length=CHOICE_TEXT_MAX)
    is_correct: StrictBool


# Type rules expressed in JSON Schema as well, so Laravel / Mlynček B can check them
# without our Python code. Duplicate choice texts cannot be expressed in JSON Schema
# (uniqueItems compares whole objects), so that rule lives only in the validator.
_TYPE_RULES_SCHEMA: list[dict[str, Any]] = [
    {
        "if": {"properties": {"type": {"const": "choice"}}, "required": ["type"]},
        "then": {
            "properties": {
                "choices": {
                    "minItems": 2,
                    "contains": {"properties": {"is_correct": {"const": True}}, "required": ["is_correct"]},
                }
            }
        },
    },
    {
        "if": {"properties": {"type": {"const": "open"}}, "required": ["type"]},
        "then": {
            "properties": {
                "choices": {
                    "minItems": 1,
                    "maxItems": 1,
                    "items": {"properties": {"is_correct": {"const": True}}},
                }
            }
        },
    },
    {
        "if": {"properties": {"type": {"const": "no_answer"}}, "required": ["type"]},
        "then": {"properties": {"choices": {"maxItems": 0}}},
    },
]


class Question(_ContractModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"allOf": _TYPE_RULES_SCHEMA})

    name: str = Field(min_length=1, max_length=QUESTION_NAME_MAX, description="Short question name.")
    description: str = Field(min_length=1, max_length=QUESTION_DESCRIPTION_MAX, description="Question / task text.")
    type: QuestionType
    time_to_handle: Seconds
    time_to_explain: Seconds
    hints: list[Hint]
    choices: list[Choice]

    @model_validator(mode="after")
    def _check_type_rules(self) -> "Question":
        errors = type_rule_errors(self.type, [(c.text, c.is_correct) for c in self.choices])
        if errors:
            raise ValueError(errors[0])
        return self


def choice_key(text: str) -> str:
    """Text of a choice for the duplicate check: whitespace collapsed, case folded."""
    return " ".join(text.split()).casefold()


def type_rule_errors(question_type: str, choices: list[tuple[str, bool]]) -> list[str]:
    """Rules of the question types (kap. 7) on (text, is_correct) pairs; [] when they hold.

    Used by the Question validator and by the validation of the question pipeline, so the
    rules exist only once.
    """
    if question_type == "choice":
        if len(choices) < 2:
            return ["choice question needs at least 2 choices"]
        if not any(correct for _, correct in choices):
            return ["choice question needs at least 1 correct choice"]
        seen: set[str] = set()
        for text, _ in choices:
            key = choice_key(text)
            if key in seen:
                return [f"choice question has duplicate choice text: {text!r}"]
            seen.add(key)
    elif question_type == "open":
        if len(choices) != 1 or not choices[0][1]:
            return ["open question needs exactly 1 choice with is_correct=true"]
    elif question_type == "no_answer":
        if choices:
            return ["no_answer question must have empty choices"]
    return []


class Unit(_ContractModel):
    title: str = Field(min_length=1, max_length=UNIT_TITLE_MAX)
    description: str = Field(max_length=UNIT_DESCRIPTION_MAX)
    questions: list[Question]


class UnitsResult(_ContractModel):
    model_config = ConfigDict(extra="forbid", title="UnitsResult")

    units: list[Unit] = Field(min_length=1, description="Lessons; exactly one in split mode 'single'.")
