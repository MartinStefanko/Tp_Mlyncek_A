"""MockBackend: in-memory jobs that pass through the steps in time.

Purpose: Laravel can integrate against the real API before the pipeline exists.

Timeline of one attempt (D = QGEN_MOCK_DURATION_S, 5×D for "mock-slow-"):
- first 5 % of D: status queued, stage null, progress 0;
- the rest is split evenly among the 8 JobStage steps (status running);
  progress grows from 0 to 1 over the running part;
- at D: completed, stage "output", progress 1.0; result/report = fixtures.

external_ref starting with "mock-fail-": the first attempt fails in the middle of
step "questions" with error.code llm_error; retryable_until = failure + 24 h
(QGEN_FAILED_INPUT_TTL_HOURS). A retry runs normally and completes.

The state is computed from the clock: every call first advances the jobs, and a
background ticker does the same so that webhooks go out without polling.
"""

import asyncio
import copy
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Literal, Optional, Protocol, get_args

from qgen.contract import (
    JobError,
    JobListItem,
    JobOptions,
    JobStage,
    JobStatus,
    JobStatusValue,
    Report,
    UnitsResult,
    WebhookEvent,
    WebhookPayload,
)
from service.backend import CreateJobRequest, CreateJobResult
from service.errors import ApiError
from service.settings import ServiceSettings

logger = logging.getLogger("qgen.mock")

FIXTURES_DIR = Path(__file__).parent / "fixtures"

FAIL_PREFIX = "mock-fail-"
SLOW_PREFIX = "mock-slow-"
SLOW_FACTOR = 5
QUEUED_FRACTION = 0.05
FAIL_STAGE: JobStage = "questions"
RETRY_AFTER_S = 60

STAGES: tuple[JobStage, ...] = get_args(JobStage)
ACTIVE: tuple[JobStatusValue, ...] = ("queued", "running")

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class WebhookNotifier(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def submit(self, url: str, payload: WebhookPayload) -> None: ...


@dataclass(frozen=True)
class MockFixtures:
    result: UnitsResult
    report: Report

    @classmethod
    def load(cls, directory: Path = FIXTURES_DIR) -> "MockFixtures":
        units = json.loads((directory / "units_example.json").read_text(encoding="utf-8"))
        report = json.loads((directory / "report_example.json").read_text(encoding="utf-8"))
        return cls(result=UnitsResult.model_validate(units), report=Report.model_validate(report))


@dataclass
class _MockJob:
    job_id: str
    owner: str
    options: JobOptions
    files: list[tuple[str, int]]  # (name, size); content is not kept
    created_at: datetime
    started_at: datetime  # start of the current attempt
    updated_at: datetime
    behavior: Literal["normal", "fail", "slow"]
    attempt: int = 1
    status: JobStatusValue = "queued"
    stage: Optional[JobStage] = None
    progress: float = 0.0
    error: Optional[JobError] = None
    retryable_until: Optional[datetime] = None
    reported_stages: set[str] = field(default_factory=set)

    @property
    def external_ref(self) -> Optional[str]:
        return self.options.external_ref

    def to_status(self) -> JobStatus:
        return JobStatus(
            job_id=self.job_id,
            external_ref=self.external_ref,
            status=self.status,
            stage=self.stage,
            progress=self.progress,
            error=self.error,
            created_at=self.created_at,
            updated_at=self.updated_at,
            retryable_until=self.retryable_until,
        )


@dataclass(frozen=True)
class _Event:
    url: str
    payload: WebhookPayload


class MockBackend:
    def __init__(
        self,
        settings: ServiceSettings,
        notifier: Optional[WebhookNotifier] = None,
        *,
        fixtures: Optional[MockFixtures] = None,
        clock: Clock = utc_now,
        tick_interval_s: Optional[float] = 0.5,
    ) -> None:
        self._settings = settings
        self._notifier = notifier
        self._fixtures = fixtures or MockFixtures.load()
        self._clock = clock
        self._tick_interval_s = tick_interval_s
        self._lock = threading.RLock()
        self._jobs: dict[str, _MockJob] = {}
        self._refs: dict[tuple[str, str], str] = {}  # (owner, external_ref) -> job_id
        self._ticker: Optional[asyncio.Task] = None

    # --- lifecycle ------------------------------------------------------------------------

    async def start(self) -> None:
        if self._notifier is not None:
            await self._notifier.start()
        if self._tick_interval_s:
            self._ticker = asyncio.create_task(self._tick_loop(self._tick_interval_s))

    async def stop(self) -> None:
        if self._ticker is not None:
            self._ticker.cancel()
            try:
                await self._ticker
            except asyncio.CancelledError:
                pass
            self._ticker = None
        if self._notifier is not None:
            await self._notifier.stop()

    async def _tick_loop(self, interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            try:
                self.tick()
            except Exception:  # keep ticking; one bad job must not stop the others
                logger.exception("mock tick failed")

    def tick(self) -> None:
        """Advance all jobs to the current time and send due webhooks."""
        with self._lock:
            events = [event for job in self._jobs.values() for event in self._advance(job)]
        self._emit(events)

    def health(self) -> dict[str, str]:
        return {"api": "ok", "redis": "mock", "worker": "mock"}

    # --- operations -------------------------------------------------------------------------

    def create(self, request: CreateJobRequest) -> CreateJobResult:
        external_ref = request.options.external_ref
        with self._lock:
            events = [event for job in self._jobs.values() for event in self._advance(job)]
            existing = self._refs.get((request.owner, external_ref)) if external_ref else None
            if existing is not None:
                result = CreateJobResult(status=self._jobs[existing].to_status(), replayed=True)
            else:
                self._check_capacity()
                now = self._clock()
                job = _MockJob(
                    job_id=uuid.uuid4().hex,
                    owner=request.owner,
                    options=request.options,
                    files=[(f.filename, len(f.content)) for f in request.files],
                    created_at=now,
                    started_at=now,
                    updated_at=now,
                    behavior=_behavior(external_ref),
                )
                self._jobs[job.job_id] = job
                if external_ref:
                    self._refs[(request.owner, external_ref)] = job.job_id
                logger.info("job created job_id=%s external_ref=%s files=%d", job.job_id, external_ref, len(job.files))
                result = CreateJobResult(status=job.to_status(), replayed=False)
        self._emit(events)
        return result

    def get(self, job_id: str, owner: str) -> JobStatus:
        with self._lock:
            job, events = self._find(job_id, owner)
            status = job.to_status()
        self._emit(events)
        return status

    def list_by_external_ref(self, external_ref: str, owner: str) -> list[JobListItem]:
        self.tick()
        with self._lock:
            jobs = [j for j in self._jobs.values() if j.owner == owner and j.external_ref == external_ref]
            jobs.sort(key=lambda j: j.created_at, reverse=True)
            return [JobListItem(job_id=j.job_id, status=j.status, created_at=j.created_at) for j in jobs]

    def result(self, job_id: str, owner: str) -> UnitsResult:
        self._require_completed(job_id, owner)
        return self._fixtures.result.model_copy(deep=True)

    def report(self, job_id: str, owner: str) -> Report:
        self._require_completed(job_id, owner)
        return self._fixtures.report.model_copy(deep=True)

    def cancel_or_delete(self, job_id: str, owner: str) -> Optional[JobStatus]:
        with self._lock:
            job, events = self._find(job_id, owner)
            if job.status in ACTIVE:
                now = self._clock()
                job.status = "cancelled"
                job.updated_at = now
                events.extend(self._event(job, "job.cancelled", now))
                logger.info("job cancelled job_id=%s", job.job_id)
                status: Optional[JobStatus] = job.to_status()
            else:
                del self._jobs[job.job_id]
                if job.external_ref and self._refs.get((job.owner, job.external_ref)) == job.job_id:
                    del self._refs[(job.owner, job.external_ref)]
                logger.info("job deleted job_id=%s", job.job_id)
                status = None
        self._emit(events)
        return status

    def retry(self, job_id: str, owner: str) -> JobStatus:
        with self._lock:
            job, events = self._find(job_id, owner)
            now = self._clock()
            if job.status != "failed":
                error = ApiError(409, "job_not_retryable", f"Job is {job.status}; only a failed job can be retried.")
            elif job.retryable_until is not None and now > job.retryable_until:
                error = ApiError(409, "job_not_retryable", "Input PDF was already deleted; create a new job.")
            else:
                error = None
            if error is None:
                try:
                    self._check_capacity()
                except ApiError as exc:
                    error = exc
            if error is None:
                job.attempt += 1
                job.status = "queued"
                job.stage = None
                job.progress = 0.0
                job.error = None
                job.retryable_until = None
                job.started_at = now
                job.updated_at = now
                job.reported_stages = set()
                logger.info("job retried job_id=%s attempt=%d", job.job_id, job.attempt)
            status = job.to_status()
        self._emit(events)
        if error is not None:
            raise error
        return status

    # --- internals --------------------------------------------------------------------------

    def _find(self, job_id: str, owner: str) -> tuple[_MockJob, list[_Event]]:
        job = self._jobs.get(job_id)
        if job is None or job.owner != owner:
            raise ApiError(404, "job_not_found", "Job does not exist or was deleted.", {"job_id": job_id})
        return job, self._advance(job)

    def _require_completed(self, job_id: str, owner: str) -> None:
        with self._lock:
            job, events = self._find(job_id, owner)
            status = job.status
        self._emit(events)
        if status in ACTIVE:
            raise ApiError(409, "result_not_ready", f"Job is {status}; the result is not ready yet.")
        if status != "completed":
            raise ApiError(409, "job_not_completed", f"Job is {status}; it has no result.")

    def _check_capacity(self) -> None:
        active = sum(1 for j in self._jobs.values() if j.status in ACTIVE)
        if active >= self._settings.max_active_jobs:
            raise ApiError(
                429,
                "too_many_jobs",
                f"Too many active jobs (limit {self._settings.max_active_jobs}).",
                {"max_active_jobs": self._settings.max_active_jobs},
                headers={"Retry-After": str(RETRY_AFTER_S)},
            )

    def _duration_s(self, job: _MockJob) -> float:
        base = self._settings.mock_duration_s
        return base * SLOW_FACTOR if job.behavior == "slow" else base

    def _advance(self, job: _MockJob) -> list[_Event]:
        """Move the job to the state given by the clock; return the webhooks to send."""
        if job.status not in ACTIVE:
            return []
        duration = self._duration_s(job)
        queued_s = duration * QUEUED_FRACTION
        running_s = duration - queued_s
        stage_s = running_s / len(STAGES)
        elapsed = (self._clock() - job.started_at).total_seconds()

        end_s = duration
        fail = job.behavior == "fail" and job.attempt == 1
        if fail:
            end_s = queued_s + (STAGES.index(FAIL_STAGE) + 0.5) * stage_s
        t = min(elapsed, end_s)

        events: list[_Event] = []
        if t >= queued_s:
            job.status = "running"
            current = min(int((t - queued_s) / stage_s), len(STAGES) - 1)
            for index in range(current + 1):
                stage = STAGES[index]
                if stage in job.reported_stages:
                    continue
                job.reported_stages.add(stage)
                at_s = queued_s + index * stage_s
                job.stage = stage
                job.progress = _progress(at_s - queued_s, running_s)
                events.extend(self._event(job, "job.stage_changed", job.started_at + timedelta(seconds=at_s)))
            job.stage = STAGES[current]
            job.progress = min(_progress(t - queued_s, running_s), 0.99)
            job.updated_at = job.started_at + timedelta(seconds=t)

        if elapsed >= end_s:
            ended_at = job.started_at + timedelta(seconds=end_s)
            job.updated_at = ended_at
            if fail:
                job.status = "failed"
                job.error = JobError(
                    code="llm_error",
                    message="LLM provider failed after retries (simulated by mock-fail-).",
                    details={"stage": FAIL_STAGE},
                )
                job.retryable_until = ended_at + timedelta(hours=self._settings.failed_input_ttl_hours)
                events.extend(self._event(job, "job.failed", ended_at))
            else:
                job.status = "completed"
                job.stage = STAGES[-1]
                job.progress = 1.0
                events.extend(self._event(job, "job.completed", ended_at))
            logger.info("job finished job_id=%s status=%s", job.job_id, job.status)
        return events

    def _event(self, job: _MockJob, event: WebhookEvent, occurred_at: datetime) -> list[_Event]:
        if job.options.callback_url is None:
            return []
        payload = WebhookPayload(
            event=event,
            job_id=job.job_id,
            external_ref=job.external_ref,
            status=job.status,
            stage=job.stage,
            progress=job.progress,
            error=copy.deepcopy(job.error),
            occurred_at=occurred_at,
        )
        return [_Event(url=str(job.options.callback_url), payload=payload)]

    def _emit(self, events: list[_Event]) -> None:
        if self._notifier is None:
            return
        for event in events:
            self._notifier.submit(event.url, event.payload)


def _behavior(external_ref: Optional[str]) -> Literal["normal", "fail", "slow"]:
    if external_ref and external_ref.startswith(FAIL_PREFIX):
        return "fail"
    if external_ref and external_ref.startswith(SLOW_PREFIX):
        return "slow"
    return "normal"


def _progress(done_s: float, total_s: float) -> float:
    return round(max(0.0, min(done_s / total_s, 1.0)), 2)
