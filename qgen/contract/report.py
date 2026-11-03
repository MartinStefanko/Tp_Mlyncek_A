"""Report (Príloha A – GET /v1/jobs/{id}/report, kap. 6 – Report pre DB ECEH).

Data outside the `units` format: learning objects, question → LO mapping,
quality scores, split warnings / dropped questions, step timings.
"""

from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field

LO_ID_PATTERN = r"^LO-\d+$"

LoId = Annotated[str, Field(pattern=LO_ID_PATTERN, description="Learning object id, e.g. 'LO-12'.")]
Score01 = Annotated[float, Field(ge=0.0, le=1.0)]


class _ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LearningObjectSource(_ReportModel):
    file: str = Field(min_length=1, description="Source file name, e.g. 'eko.pdf'.")
    pages: list[Annotated[int, Field(ge=1)]] = Field(description="Cited pages (1-based).")


class LearningObject(_ReportModel):
    id: LoId
    title: str
    statement: str
    lo_type: str = Field(description="Kind of learning object, e.g. 'definition'.")
    source: LearningObjectSource
    keywords: list[str]
    bloom_level: str
    topic: Optional[str] = None


class QuestionMapEntry(_ReportModel):
    unit: int = Field(ge=0, description="Index in units.")
    question: int = Field(ge=0, description="Index in units[unit].questions.")
    lo_id: LoId


class Quality(_ReportModel):
    faithfulness_avg: Optional[Score01] = None
    answerability_avg: Optional[Score01] = None
    topic_coverage: Optional[Score01] = None


class ReportWarning(_ReportModel):
    code: str = Field(min_length=1, description="Machine-readable code, e.g. 'field_too_long'.")
    message: str


class Report(_ReportModel):
    model_config = ConfigDict(extra="forbid", title="Report")

    learning_objects: list[LearningObject]
    question_map: list[QuestionMapEntry]
    quality: Quality
    warnings: list[ReportWarning]
    timings: dict[str, Annotated[float, Field(ge=0.0)]] = Field(
        description="Seconds per step, e.g. {'extract': 12.4, 'learning_objects': 95.1}.",
    )
