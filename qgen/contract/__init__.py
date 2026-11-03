"""Data contract of the QGen service (Príloha A, kap. 6 and 8).

Models only – no pipeline logic. Everything here is what Laravel (ECEH) sees.
"""

from qgen.contract.jobs import (
    API_ERROR_CODES,
    ApiErrorCode,
    ErrorBody,
    ErrorResponse,
    JobError,
    JobStage,
    JobStatus,
    JobStatusValue,
    WebhookEvent,
    WebhookPayload,
)
from qgen.contract.options import (
    ExtractionOptions,
    JobOptions,
    LessonsSplit,
    QuestionOptions,
    SingleSplit,
    SplitOptions,
    SyllabusLesson,
    SyllabusSplit,
    TopicsSplit,
)
from qgen.contract.report import (
    LearningObject,
    LearningObjectSource,
    Quality,
    QuestionMapEntry,
    Report,
    ReportWarning,
)
from qgen.contract.units import Choice, Hint, Question, QuestionType, Unit, UnitsResult

__all__ = [
    "API_ERROR_CODES",
    "ApiErrorCode",
    "Choice",
    "ErrorBody",
    "ErrorResponse",
    "ExtractionOptions",
    "Hint",
    "JobError",
    "JobOptions",
    "JobStage",
    "JobStatus",
    "JobStatusValue",
    "LearningObject",
    "LearningObjectSource",
    "LessonsSplit",
    "Quality",
    "Question",
    "QuestionMapEntry",
    "QuestionOptions",
    "QuestionType",
    "Report",
    "ReportWarning",
    "SingleSplit",
    "SplitOptions",
    "SyllabusLesson",
    "SyllabusSplit",
    "TopicsSplit",
    "Unit",
    "UnitsResult",
    "WebhookEvent",
    "WebhookPayload",
]
