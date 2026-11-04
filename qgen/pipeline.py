"""Whole pipeline after extraction: LO (tematické celky) and questions.

Same order as the former app.py: generate_lo_pipeline, then generate_all_items only
when some LO passed the filter. Nothing is written to disk; the caller (CLI, worker)
decides what to store.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from qgen.config import Settings, get_settings
from qgen.lo.pipeline import LOPipelineResult, generate_lo_pipeline
from qgen.progress import ProgressCallback
from qgen.questions.pipeline import QuestionPipelineResult, generate_all_items


@dataclass
class PipelineResult:
    """learning_objects / questions: accepted results; reports: all report dicts by name.

    Report keys: lo_validation, document_topics, topic_coverage, lo_relevance_to_segment,
    lo_faithfulness, lo_processing_time and, when questions ran, item_validation,
    item_relevance_to_lo, item_faithfulness, item_answerability, item_processing_time.
    """

    learning_objects: list[dict]
    questions: list[dict]
    reports: dict[str, Any]
    lo_result: LOPipelineResult
    question_result: Optional[QuestionPipelineResult] = None

    @property
    def all_learning_objects(self) -> list[dict]:
        return self.lo_result.all_learning_objects

    @property
    def all_questions(self) -> list[dict]:
        return self.question_result.all_questions if self.question_result else []

    @property
    def language_info(self) -> dict[str, str]:
        return self.lo_result.language_info


def run_pipeline(
    segments,
    client=None,
    language: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
    settings: Optional[Settings] = None,
    verbose: bool = False,
) -> PipelineResult:
    """language: None / "auto" = detect (both pipelines detect, as today); "sk" / "en" = use it."""
    settings = settings or get_settings()
    lo_result = generate_lo_pipeline(
        segments,
        client=client,
        verbose=verbose,
        document_language=language,
        progress_callback=progress_callback,
        settings=settings,
    )
    reports = dict(lo_result.reports)
    question_result = None
    if lo_result.learning_objects:
        question_result = generate_all_items(
            lo_result.learning_objects,
            segments,
            client=client,
            verbose=verbose,
            document_language=language,
            progress_callback=progress_callback,
            settings=settings,
        )
        reports.update(question_result.reports)
    return PipelineResult(
        learning_objects=lo_result.learning_objects,
        questions=question_result.questions if question_result else [],
        reports=reports,
        lo_result=lo_result,
        question_result=question_result,
    )
