"""JobRunner: one run of a job in the worker, from queued to a final state.

    1. read job.json; not queued (cancelled, already taken, duplicate entry) → end quietly
    2. queued → running (the store adds 1 to attempts), check_config
    3. extraction of input/*.pdf (options: OCR), check that some page has text
    4. run_pipeline (options: language, per_section) with a progress callback
    5. output: temporary mapping to units + report (service.jobs.legacy_output, step 14)
    6. save result.json and report.json, running → completed, delete the PDFs

Progress (ProgressWriter): stage and progress go to job.json at most every
PROGRESS_INTERVAL_S or when the step changes. The same locked write checks for
cancellation: a cancelled job raises JobCancelled and the run stops. The callback also
checks the job's own time limit (QGEN_JOB_TIMEOUT_S).

Errors → failed (error.code): LLMError → llm_error; no text / OCR missing / unreadable PDF
→ extraction_failed; time limit, configuration and every other error → internal_error.
The message never has a prompt or document text. The PDFs of a failed job stay for retry
(QGEN_FAILED_INPUT_TTL_HOURS). A cancelled job (also InvalidTransition when writing the
final state after a cancel) is a normal end: logged, PDFs deleted.

Webhooks (job.stage_changed, job.completed, …) come in step 11.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pypdf.errors import PyPdfError

from qgen.config import Settings
from qgen.contract import JobError
from qgen.extraction.text_extraction import OcrUnavailableError, pdfs_to_text
from qgen.llm.base import EmbeddingClient, LLMClient
from qgen.llm.errors import LLMConfigError, LLMError
from qgen.llm.factory import check_config, get_embedder, get_llm
from qgen.pipeline import run_pipeline
from qgen.progress import PIPELINE_STAGES
from service.jobs.legacy_output import build_output
from service.jobs.models import JobRecord
from service.jobs.store import InvalidTransition, JobNotFound, JobStore

logger = logging.getLogger("qgen.worker")

PROGRESS_INTERVAL_S = 2.0
PROGRESS_BEFORE_COMPLETED = 0.99

# Share of each step in the whole run (split is skipped in mode "single").
STAGE_WEIGHTS: dict[str, float] = {
    "extract": 0.05,
    "language": 0.02,
    "learning_objects": 0.38,
    "lo_evaluation": 0.15,
    "questions": 0.25,
    "q_evaluation": 0.13,
    "output": 0.02,
}
STAGE_ORDER: tuple[str, ...] = (*PIPELINE_STAGES, "output")


# --- stop signals ---------------------------------------------------------------------------
# BaseException on purpose: `except Exception` in the pipeline or the retry layer must not
# swallow them (same reason as tests.replay_client.UnrecordedCallError).


class JobStop(BaseException):
    """The run must stop now."""


class JobCancelled(JobStop):
    """The job was cancelled (DELETE) during the run."""


class JobLost(JobStop):
    """job.json is no longer running (e.g. recovery put it back to the queue)."""


class JobTimedOut(JobStop):
    def __init__(self, timeout_s: float) -> None:
        super().__init__(f"job time limit of {timeout_s:g} s exceeded")
        self.timeout_s = timeout_s


class NoTextExtracted(Exception):
    """No page of the PDFs has text."""


class ExtractionError(Exception):
    """The PDF could not be read (passed the API check, failed in the extraction)."""


# --- progress -------------------------------------------------------------------------------


def stage_start(stage: str) -> float:
    total = 0.0
    for name in STAGE_ORDER:
        if name == stage:
            return total
        total += STAGE_WEIGHTS.get(name, 0.0)
    return total


def overall_progress(stage: str, done: int, total: int) -> float:
    fraction = 1.0 if total <= 0 else min(1.0, max(0.0, done / total))
    return stage_start(stage) + STAGE_WEIGHTS.get(stage, 0.0) * fraction


class ProgressWriter:
    """progress_callback(stage, done, total) of the pipeline → job.json, throttled."""

    def __init__(
        self,
        store: JobStore,
        job_id: str,
        *,
        deadline: float,
        timeout_s: float,
        interval_s: float = PROGRESS_INTERVAL_S,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store
        self.job_id = job_id
        self.deadline = deadline
        self.timeout_s = timeout_s
        self.interval_s = interval_s
        self.monotonic = monotonic
        self.stage: Optional[str] = None
        self.progress = 0.0
        self.written_stage: Optional[str] = None
        self.last_write: Optional[float] = None

    def __call__(self, stage: str, done: int, total: int) -> None:
        self.check_deadline()
        if stage not in STAGE_ORDER:
            return
        if self.stage is None or STAGE_ORDER.index(stage) >= STAGE_ORDER.index(self.stage):
            self.stage = stage  # the step only moves forward (language is detected twice)
        self.progress = max(self.progress, min(overall_progress(stage, done, total), PROGRESS_BEFORE_COMPLETED))
        now = self.monotonic()
        if self.stage != self.written_stage or self.last_write is None or now - self.last_write >= self.interval_s:
            self.write()

    def check_deadline(self) -> None:
        if self.monotonic() >= self.deadline:
            raise JobTimedOut(self.timeout_s)

    def write(self) -> None:
        stage, progress = self.stage, round(self.progress, 4)

        def change(record: JobRecord) -> None:
            if record.status == "cancelled" or record.cancel_requested:
                raise JobCancelled()
            if record.status != "running":
                raise JobLost()
            record.stage = stage  # type: ignore[assignment]
            record.progress = progress

        try:
            self.store.update(self.job_id, change)
        except JobNotFound:
            raise JobLost() from None
        self.written_stage = stage
        self.last_write = self.monotonic()


# --- runner ---------------------------------------------------------------------------------


@dataclass
class JobRunner:
    """`llm` / `embedder`: None = clients from qgen.llm.factory (checked by check_config)."""

    store: JobStore
    settings: Settings
    llm: Optional[LLMClient] = None
    embedder: Optional[EmbeddingClient] = None
    progress_interval_s: float = PROGRESS_INTERVAL_S
    monotonic: Callable[[], float] = time.monotonic
    extract: Callable[..., list] = field(default=pdfs_to_text)
    pipeline: Callable[..., Any] = field(default=run_pipeline)

    def run(self, job_id: str) -> Optional[str]:
        """Run the job; returns its final status, or None when the run did not start."""
        record = self._start(job_id)
        if record is None:
            return None
        started = self.monotonic()
        progress = ProgressWriter(
            self.store,
            job_id,
            deadline=started + self.settings.job_timeout_s,
            timeout_s=self.settings.job_timeout_s,
            interval_s=self.progress_interval_s,
            monotonic=self.monotonic,
        )
        try:
            self._process(record, progress)
        except JobCancelled:
            return self._cancelled(job_id)
        except JobLost:
            logger.warning("job no longer running, run stopped job_id=%s", job_id)
            return None
        except JobTimedOut as exc:
            return self._fail(job_id, _timeout_error(exc.timeout_s, progress.stage))
        except Exception as exc:  # noqa: BLE001 – every error ends as a failed job
            return self._fail(job_id, self._error_for(exc, progress, record))
        return self._complete_or_cancelled(job_id)

    # --- steps ------------------------------------------------------------------------

    def _start(self, job_id: str) -> Optional[JobRecord]:
        try:
            record = self.store.get(job_id)
        except JobNotFound:
            logger.info("job not found, skipped job_id=%s", job_id)
            return None
        if record.status != "queued":
            logger.info("job not queued, skipped job_id=%s status=%s", job_id, record.status)
            return None
        try:
            record = self.store.update(job_id, _to_running)
        except (InvalidTransition, JobNotFound):
            logger.info("job taken or changed meanwhile, skipped job_id=%s", job_id)
            return None
        logger.info("job started job_id=%s external_ref=%s attempt=%d", job_id, record.external_ref, record.attempts)
        return record

    def _process(self, record: JobRecord, progress: ProgressWriter) -> None:
        job_id, options = record.job_id, record.options
        check_config(self.settings, llm=self.llm is None, embedder=self.embedder is None)
        llm = self.llm or get_llm("lo", self.settings)
        embedder = self.embedder or get_embedder(self.settings)

        timings: dict[str, float] = {}
        step_started = time.perf_counter()
        segments = self._extract(record, progress)
        timings["extract"] = round(time.perf_counter() - step_started, 4)

        result = self.pipeline(
            segments,
            llm=llm,
            embedder=embedder,
            language=options.language,
            progress_callback=progress,
            settings=self.settings,
            per_section=options.questions.per_section if options.questions else None,
        )

        progress("output", 0, 1)
        step_started = time.perf_counter()
        output = build_output(result, record, timings=timings)
        output.report.timings["output"] = round(time.perf_counter() - step_started, 4)
        progress.check_deadline()
        self.store.save_result(job_id, output.result)
        self.store.save_report(job_id, output.report)
        logger.info(
            "job output job_id=%s sections=%d questions=%d dropped=%d",
            job_id,
            len(output.report.sections),
            len(output.result.units[0].questions),
            output.dropped,
        )

    def _extract(self, record: JobRecord, progress: ProgressWriter) -> list:
        paths = self.store.input_paths(record.job_id)
        if len(paths) != len(record.input_files):
            raise ExtractionError("input PDF missing")
        names = {f.stored_as: f.name for f in record.input_files}
        inputs = [{"path": str(path), "name": names[path.name]} for path in paths]
        ocr = bool(record.options.extraction and record.options.extraction.ocr)
        try:
            segments = self.extract(inputs, ocr=ocr, progress_callback=progress, settings=self.settings)
        except (OcrUnavailableError, JobStop):
            raise
        except (PyPdfError, RuntimeError, ValueError, OSError) as exc:
            raise ExtractionError(type(exc).__name__) from exc
        if not any(page_has_text(segment) for segment in segments):
            raise NoTextExtracted()
        return segments

    # --- final states -----------------------------------------------------------------

    def _complete_or_cancelled(self, job_id: str) -> Optional[str]:
        try:
            self.store.update(job_id, _to_completed)
        except InvalidTransition:
            return self._final_after_conflict(job_id)
        except JobNotFound:
            logger.info("job deleted during the run job_id=%s", job_id)
            return None
        self._delete_inputs(job_id)
        logger.info("job completed job_id=%s", job_id)
        return "completed"

    def _fail(self, job_id: str, error: JobError) -> Optional[str]:
        def change(record: JobRecord) -> None:
            if record.status != "running":
                raise InvalidTransition(job_id, record.status, "failed", "no longer running")
            record.status = "failed"
            record.error = error

        try:
            self.store.update(job_id, change)
        except InvalidTransition:
            return self._final_after_conflict(job_id)
        except JobNotFound:
            logger.info("job deleted during the run job_id=%s", job_id)
            return None
        logger.warning("job failed job_id=%s code=%s", job_id, error.code)
        return "failed"

    def _final_after_conflict(self, job_id: str) -> Optional[str]:
        """The final write was refused: normal when the job was cancelled meanwhile."""
        try:
            status = self.store.get(job_id).status
        except JobNotFound:
            return None
        if status == "cancelled":
            return self._cancelled(job_id)
        logger.warning("job changed during the run job_id=%s status=%s", job_id, status)
        return None

    def _cancelled(self, job_id: str) -> str:
        logger.info("job cancelled during the run, stopped job_id=%s", job_id)
        self._delete_inputs(job_id)
        return "cancelled"

    def _delete_inputs(self, job_id: str) -> None:
        try:
            self.store.delete_inputs(job_id)
        except (JobNotFound, OSError) as exc:  # cleanup deletes them later
            logger.error("input delete failed job_id=%s error=%s", job_id, type(exc).__name__)

    # --- errors -----------------------------------------------------------------------

    def _error_for(self, exc: Exception, progress: ProgressWriter, record: JobRecord) -> JobError:
        stage = progress.stage
        if isinstance(exc, LLMError):
            return JobError(
                code="llm_error",
                message=str(exc),
                details={
                    "stage": stage,
                    "role": exc.role,
                    "model": exc.model,
                    "attempts": exc.attempts,
                    "invalid_json": exc.invalid_json,
                },
            )
        if isinstance(exc, NoTextExtracted):
            ocr = bool(record.options.extraction and record.options.extraction.ocr)
            hint = "" if ocr else " Try again with extraction.ocr=true."
            return JobError(
                code="extraction_failed",
                message=f"No text could be extracted from the PDF.{hint}",
                details={"ocr": ocr},
            )
        if isinstance(exc, OcrUnavailableError):
            return JobError(
                code="extraction_failed",
                message="OCR was requested, but it is not installed in this service.",
                details={"ocr": True},
            )
        if isinstance(exc, ExtractionError):
            return JobError(
                code="extraction_failed", message="The PDF could not be read.", details={"error_type": str(exc)}
            )
        if _is_rq_timeout(exc):
            return _timeout_error(self.settings.job_timeout_s, stage)
        if isinstance(exc, LLMConfigError):
            logger.error("service configuration error job_id=%s", record.job_id)
            return JobError(
                code="internal_error",
                message="The service configuration is not valid.",
                details={"stage": stage, "error_type": type(exc).__name__},
            )
        logger.exception("unexpected error job_id=%s stage=%s error=%s", record.job_id, stage, type(exc).__name__)
        return JobError(
            code="internal_error",
            message=f"Unexpected error in step {stage or 'start'}.",
            details={"stage": stage, "error_type": type(exc).__name__},
        )


def page_has_text(segment: dict) -> bool:
    """True if the page has text besides the header added by the extraction."""
    lines = str(segment.get("text", "")).splitlines()
    header = 2 if segment.get("source_id") else 1
    return any(line.strip() for line in lines[header:])


def _timeout_error(timeout_s: float, stage: Optional[str]) -> JobError:
    return JobError(
        code="internal_error",
        message=f"The job exceeded its time limit of {timeout_s:g} s.",
        details={"reason": "timeout", "timeout_s": timeout_s, "stage": stage},
    )


def _is_rq_timeout(exc: Exception) -> bool:
    try:
        from rq.timeouts import JobTimeoutException
    except ImportError:  # pragma: no cover
        return False
    return isinstance(exc, JobTimeoutException)


def _to_running(record: JobRecord) -> None:
    if record.status != "queued":
        raise InvalidTransition(record.job_id, record.status, "running", "no longer queued")
    record.status = "running"
    record.error = None


def _to_completed(record: JobRecord) -> None:
    if record.status != "running":
        raise InvalidTransition(record.job_id, record.status, "completed", "no longer running")
    record.status = "completed"
