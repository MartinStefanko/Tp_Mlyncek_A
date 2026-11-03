"""Report (Príloha A – GET /v1/jobs/{id}/report, kap. 6 – Report pre DB ECEH).

Data outside the `units` format:
- sections: Mlynček's sections (tematické celky; "LO" in the original code). Context
  only, Laravel does not have to store them.
- question_meta: one entry per question in units (Bloom level, keywords, cited pages,
  topic). In ECEH a learning object is one question, Laravel fills its tables from this.
- quality, warnings (split warnings, dropped questions), timings.
"""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SECTION_ID_PATTERN = r"^S-\d+$"

SectionId = Annotated[str, Field(pattern=SECTION_ID_PATTERN, description="Section id, e.g. 'S-12'.")]
Score01 = Annotated[float, Field(ge=0.0, le=1.0)]

BloomLevel = Literal["remember", "understand", "apply", "analyze", "evaluate", "create"]


class _ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Section(_ReportModel):
    """Section (tematický celok) of the document."""

    id: SectionId
    title: str
    statement: str
    lo_type: str = Field(description="Kind of section, e.g. 'definition'.")
    prerequisites: list[SectionId] = Field(description="Ids of sections this one builds on.")
    topic: Optional[str] = None


class SourceRef(_ReportModel):
    file: str = Field(min_length=1, description="Source file name, e.g. 'eko.pdf'.")
    pages: list[Annotated[int, Field(ge=1)]] = Field(description="Cited pages (1-based).")


class QuestionMeta(_ReportModel):
    """Metadata of one question; the question is units[unit].questions[question]."""

    unit: int = Field(ge=0, description="Index in units.")
    question: int = Field(ge=0, description="Index in units[unit].questions.")
    section_id: SectionId
    bloom_level: BloomLevel
    keywords: list[str]
    source: list[SourceRef] = Field(description="Cited files and pages; a question can cite more PDFs.")
    topic: Optional[str] = None


class Quality(_ReportModel):
    """Averages in the range 0–1, or null when not measured.

    The pipeline scores on a 1–5 scale; the service converts a score x as (x − 1) / 4.
    Topic coverage is a percentage in the pipeline; the service divides it by 100.
    """

    section_faithfulness_avg: Optional[Score01] = None
    question_faithfulness_avg: Optional[Score01] = None
    answerability_avg: Optional[Score01] = None
    topic_coverage: Optional[Score01] = None


class ReportWarning(_ReportModel):
    code: str = Field(min_length=1, description="Machine-readable code, e.g. 'field_too_long'.")
    message: str


class Report(_ReportModel):
    model_config = ConfigDict(extra="forbid", title="Report")

    sections: list[Section]
    question_meta: list[QuestionMeta]
    quality: Quality
    warnings: list[ReportWarning]
    timings: dict[str, Annotated[float, Field(ge=0.0)]] = Field(
        description="Seconds per step, e.g. {'extract': 1.5, 'learning_objects': 56.3}.",
    )
