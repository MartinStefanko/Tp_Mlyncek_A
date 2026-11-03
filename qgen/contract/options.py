"""Job input parameters `options` (Príloha A – Vstupné parametre, kap. 8 – split).

All parameters are optional. A missing value is None in the model; the API fills it
from the service configuration with `resolve_options` when the job is created and
stores the complete options with the job (a retry then uses the same settings even
if the configuration changed). Without `split` everything goes into one lesson
(mode "single").

Checks that need configuration (QGEN_WEBHOOK_ALLOWED_HOSTS, OCR not installed)
are API logic, not part of these models.
"""

from dataclasses import dataclass
from typing import Annotated, Literal, Optional, Union

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator

from qgen.contract.units import UNIT_DESCRIPTION_MAX, UNIT_TITLE_MAX, QuestionType

LESSON_MINUTES_MIN = 5
LESSON_MINUTES_MAX = 240
LESSON_COUNT_MAX = 100
SYLLABUS_MAX_LESSONS = 100

# One or more page ranges, e.g. "D1:1-10" or "D1:1-10, D2:3".
PAGE_RANGES_PATTERN = r"^D\d+:\d+(-\d+)?(\s*,\s*D\d+:\d+(-\d+)?)*$"

LessonMinutes = Annotated[int, Field(ge=LESSON_MINUTES_MIN, le=LESSON_MINUTES_MAX, description="Lesson length in minutes.")]
LessonCount = Annotated[int, Field(ge=1, le=LESSON_COUNT_MAX, description="Number of lessons.")]
Language = Literal["auto", "sk", "en"]


class _OptionsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractionOptions(_OptionsModel):
    ocr: Optional[bool] = Field(default=None, description="OCR of images in the PDF. Default: QGEN_OCR_ENABLED.")


class QuestionOptions(_OptionsModel):
    types: Optional[list[QuestionType]] = Field(
        default=None,
        min_length=1,
        description="Question types to generate. Default: all three.",
    )
    per_section: Optional[int] = Field(
        default=None, ge=1, le=10, description="Questions per section (tematický celok). Default: 3."
    )
    choice_options: Optional[int] = Field(
        default=None, ge=2, le=10, description="Number of options for a choice question. Default: 4."
    )
    hints: Optional[int] = Field(default=None, ge=0, le=2, description="Number of hints (e.g. easy and hard). Default: 2.")

    @model_validator(mode="after")
    def _unique_types(self) -> "QuestionOptions":
        if self.types is not None and len(set(self.types)) != len(self.types):
            raise ValueError("questions.types contains duplicates")
        return self


# --- split: discriminated union on `mode` -------------------------------------------
# Every variant accepts the other variants' keys only as null, so the full-shape
# example from Príloha A ({"mode": "topics", "lesson_count": null, ..., "syllabus": null})
# stays valid while a value in a field that the mode does not use is rejected.
# A missing lesson_minutes is filled by resolve_options (QGEN_DEFAULT_LESSON_MINUTES).


class SingleSplit(_OptionsModel):
    mode: Literal["single"] = "single"
    lesson_count: None = None
    lesson_minutes: Optional[LessonMinutes] = None
    syllabus: None = None


class LessonsSplit(_OptionsModel):
    mode: Literal["lessons"]
    lesson_count: Optional[LessonCount] = None
    lesson_minutes: Optional[LessonMinutes] = None
    syllabus: None = None

    @model_validator(mode="after")
    def _count_or_minutes(self) -> "LessonsSplit":
        if self.lesson_count is None and self.lesson_minutes is None:
            raise ValueError("split mode 'lessons' needs lesson_count or lesson_minutes")
        return self


class TopicsSplit(_OptionsModel):
    mode: Literal["topics"]
    lesson_count: None = None
    lesson_minutes: Optional[LessonMinutes] = None
    syllabus: None = None


class SyllabusLesson(_OptionsModel):
    title: str = Field(min_length=1, max_length=UNIT_TITLE_MAX)
    description: Optional[str] = Field(default=None, max_length=UNIT_DESCRIPTION_MAX)
    pages: Optional[str] = Field(
        default=None,
        pattern=PAGE_RANGES_PATTERN,
        description="Page ranges, e.g. 'D1:1-10' or 'D1:1-10, D2:3'.",
    )


class SyllabusSplit(_OptionsModel):
    mode: Literal["syllabus"]
    lesson_count: None = None
    lesson_minutes: Optional[LessonMinutes] = None
    syllabus: list[SyllabusLesson] = Field(min_length=1, max_length=SYLLABUS_MAX_LESSONS)


SplitOptions = Annotated[
    Union[SingleSplit, LessonsSplit, TopicsSplit, SyllabusSplit],
    Field(discriminator="mode"),
]


class JobOptions(_OptionsModel):
    external_ref: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Caller's own identifier; unique per API key; required for ECEH.",
    )
    callback_url: Optional[AnyHttpUrl] = Field(
        default=None,
        description="Webhook address; host must be in QGEN_WEBHOOK_ALLOWED_HOSTS (checked by the API).",
    )
    language: Optional[Language] = Field(default=None, description="Document language. Default: 'auto'.")
    extraction: Optional[ExtractionOptions] = None
    questions: Optional[QuestionOptions] = None
    split: SplitOptions = Field(default_factory=SingleSplit)


# --- defaults --------------------------------------------------------------------------


@dataclass(frozen=True)
class OptionDefaults:
    """Values used for options the caller did not send (Príloha A, Príloha D).

    The defaults here are the documented ones; the service builds this object from
    its configuration (QGEN_OCR_ENABLED, QGEN_DEFAULT_LESSON_MINUTES, ...).
    """

    language: Language = "auto"
    ocr: bool = False
    question_types: tuple[QuestionType, ...] = ("choice", "open", "no_answer")
    per_section: int = 3
    choice_options: int = 4
    hints: int = 2
    lesson_minutes: int = 45


def resolve_options(options: JobOptions, defaults: OptionDefaults) -> JobOptions:
    """Return complete options: every None replaced by the value from `defaults`.

    The result is what gets stored with the job. external_ref and callback_url stay
    as sent (None means "not used"); unused split fields stay null.
    """
    extraction = options.extraction or ExtractionOptions()
    questions = options.questions or QuestionOptions()
    split = options.split
    if split.lesson_minutes is None:
        split = split.model_copy(update={"lesson_minutes": defaults.lesson_minutes})

    resolved = JobOptions(
        external_ref=options.external_ref,
        callback_url=options.callback_url,
        language=options.language if options.language is not None else defaults.language,
        extraction=ExtractionOptions(ocr=extraction.ocr if extraction.ocr is not None else defaults.ocr),
        questions=QuestionOptions(
            types=list(questions.types) if questions.types is not None else list(defaults.question_types),
            per_section=_or(questions.per_section, defaults.per_section),
            choice_options=_or(questions.choice_options, defaults.choice_options),
            hints=_or(questions.hints, defaults.hints),
        ),
        split=split,
    )
    # Validate once more so that bad defaults (e.g. lesson_minutes=0 in .env) fail here.
    return JobOptions.model_validate(resolved.model_dump())


def _or(value: Optional[int], default: int) -> int:
    return value if value is not None else default
