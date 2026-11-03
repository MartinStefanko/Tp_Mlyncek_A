"""Job input parameters `options` (Príloha A – Vstupné parametre, kap. 8 – split).

All parameters are optional. Defaults below equal the documented configuration
defaults (Príloha D); filling them from the live configuration is not done here.
Without `split` everything goes into one lesson (mode "single").

Checks that need configuration (QGEN_WEBHOOK_ALLOWED_HOSTS, OCR not installed)
are API logic, not part of these models.
"""

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


class _OptionsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractionOptions(_OptionsModel):
    ocr: bool = Field(default=False, description="OCR of images in the PDF (QGEN_OCR_ENABLED).")


class QuestionOptions(_OptionsModel):
    types: list[QuestionType] = Field(
        default_factory=lambda: ["choice", "open", "no_answer"],
        min_length=1,
        description="Question types to generate.",
    )
    per_learning_object: int = Field(default=3, ge=1, le=10, description="Questions per learning object.")
    choice_options: int = Field(default=4, ge=2, le=10, description="Number of options for a choice question.")
    hints: int = Field(default=2, ge=0, le=2, description="Number of hints (e.g. easy and hard).")

    @model_validator(mode="after")
    def _unique_types(self) -> "QuestionOptions":
        if len(set(self.types)) != len(self.types):
            raise ValueError("questions.types contains duplicates")
        return self


# --- split: discriminated union on `mode` -------------------------------------------
# Every variant accepts the other variants' keys only as null, so the full-shape
# example from Príloha A ({"mode": "topics", "lesson_count": null, ..., "syllabus": null})
# stays valid while a value in a field that the mode does not use is rejected.


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
        description="Caller's own identifier; unique; required for ECEH.",
    )
    callback_url: Optional[AnyHttpUrl] = Field(
        default=None,
        description="Webhook address; host must be in QGEN_WEBHOOK_ALLOWED_HOSTS (checked by the API).",
    )
    language: Literal["auto", "sk", "en"] = "auto"
    extraction: ExtractionOptions = Field(default_factory=ExtractionOptions)
    questions: QuestionOptions = Field(default_factory=QuestionOptions)
    split: SplitOptions = Field(default_factory=SingleSplit)
