"""Helpers for API tests: fake clock, recording webhook notifier, app factory for both backends.

Every test that uses the `api` / `api_factory` fixture runs twice: over MockBackend and over
RealBackend (JobStore in tmp_path, fakeredis, InMemoryJobQueue). Markers:
    @pytest.mark.mock_only  – behavior that exists only in the mock (time simulation, webhooks)
    @pytest.mark.real_only  – checks of the real backend (job.json, Redis reservations, PDF checks)

The helpers start / finish / fail move a job forward independent of the backend: the mock
advances its clock, the real backend changes job.json as a worker would.
"""

import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator, Optional, Union

import fakeredis
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from qgen.contract import ErrorResponse, JobError, JobOptions, WebhookPayload
from service.jobs import JobStore
from service.jobs.events import InMemoryJobEvents
from service.jobs.queue import InMemoryJobQueue
from service.jobs.refs import ExternalRefs
from service.main import create_app
from service.mock import MockBackend
from service.mock.backend import MockFixtures
from service.real_backend import RealBackend
from service.settings import ServiceSettings

API_KEY = "test-key"
CALLBACK_URL = "http://eceh-backend/api/internal/qgen/webhook"
DURATION_S = 30.0
BACKENDS = ("mock", "real")
LLM_ERROR = JobError(code="llm_error", message="LLM provider failed after retries.")


def make_pdf(pages: int = 1, pad: int = 0, password: Optional[str] = None) -> bytes:
    """A valid PDF; `pad` adds bytes (metadata) to reach a size, `password` encrypts it."""
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    if pad:
        writer.add_metadata({"/Subject": "x" * pad})
    if password:
        writer.encrypt(password)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


PDF = make_pdf()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "backend_kind" not in metafunc.fixturenames:
        return
    if metafunc.definition.get_closest_marker("mock_only"):
        kinds: tuple[str, ...] = ("mock",)
    elif metafunc.definition.get_closest_marker("real_only"):
        kinds = ("real",)
    else:
        kinds = BACKENDS
    metafunc.parametrize("backend_kind", kinds)


class FakeClock:
    def __init__(self, start: datetime = datetime(2026, 11, 3, 10, 0, 0, tzinfo=timezone.utc)) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@dataclass
class RecordingNotifier:
    sent: list[tuple[str, WebhookPayload]] = field(default_factory=list)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def submit(self, url: str, payload: WebhookPayload) -> None:
        self.sent.append((url, payload))

    def events(self, job_id: Optional[str] = None) -> list[str]:
        return [p.event for _, p in self.sent if job_id is None or p.job_id == job_id]


def make_settings(**overrides: Any) -> ServiceSettings:
    values: dict[str, Any] = {
        "api_keys": API_KEY,
        "webhook_allowed_hosts": "eceh-backend",
        "webhook_secret": "test-secret",
        "mock_duration_s": DURATION_S,
    }
    values.update(overrides)
    return ServiceSettings(_env_file=None, **values)


@dataclass
class RealParts:
    store: JobStore
    redis: fakeredis.FakeRedis
    queue: InMemoryJobQueue
    refs: ExternalRefs
    events: InMemoryJobEvents = field(default_factory=InMemoryJobEvents)


@dataclass
class Api:
    kind: str
    client: TestClient
    backend: Union[MockBackend, RealBackend]
    clock: FakeClock
    notifier: RecordingNotifier
    settings: ServiceSettings
    headers: dict[str, str]
    real: Optional[RealParts] = None

    def create(
        self,
        options: Optional[dict[str, Any]] = None,
        files: Optional[list[tuple[str, tuple[str, bytes, str]]]] = None,
        headers: Optional[dict[str, str]] = None,
    ):
        data = {"options": json.dumps(options)} if options is not None else {}
        return self.client.post(
            "/v1/jobs",
            headers=self.headers if headers is None else headers,
            files=files if files is not None else [("files", ("eko.pdf", PDF, "application/pdf"))],
            data=data,
        )

    def create_id(self, options: Optional[dict[str, Any]] = None) -> str:
        response = self.create(options)
        assert response.status_code == 202, response.text
        return response.json()["job_id"]

    def status(self, job_id: str) -> dict[str, Any]:
        response = self.client.get(f"/v1/jobs/{job_id}", headers=self.headers)
        assert response.status_code == 200, response.text
        return response.json()

    # --- mock only: time simulation -----------------------------------------------------

    def run_to_end(self, factor: float = 1.0) -> None:
        assert isinstance(self.backend, MockBackend), "run_to_end is mock only; use finish/fail"
        self.clock.advance(DURATION_S * factor + 1)
        self.backend.tick()

    # --- both backends ------------------------------------------------------------------

    def start(self, job_id: str) -> None:
        """queued → running."""
        if self.real is None:
            self.clock.advance(DURATION_S / 2)
        else:
            self.real.store.update(job_id, _set_status("running"))
        assert self.status(job_id)["status"] == "running"

    def finish(self, job_id: str) -> None:
        """→ completed with the example result and report."""
        if self.real is None:
            self.run_to_end()
        else:
            store = self.real.store
            if store.get(job_id).status == "queued":
                store.update(job_id, _set_status("running"))
            fixtures = MockFixtures.load()
            store.save_result(job_id, fixtures.result)
            store.save_report(job_id, fixtures.report)
            store.update(job_id, _set_status("completed"))
        assert self.status(job_id)["status"] == "completed"

    def fail(self, job_id: str) -> None:
        """→ failed with llm_error (mock: the job needs external_ref "mock-fail-…")."""
        if self.real is None:
            self.run_to_end()
        else:
            store = self.real.store
            if store.get(job_id).status == "queued":
                store.update(job_id, _set_status("running"))
            store.update(job_id, _set_status("failed", LLM_ERROR))
        assert self.status(job_id)["status"] == "failed"

    def job_count(self) -> int:
        if self.real is None:
            return len(self.backend._jobs)  # type: ignore[union-attr]
        return len(list(self.real.store.iter_jobs()))

    def stored_options(self, job_id: str) -> JobOptions:
        if self.real is None:
            return self.backend._jobs[job_id].options  # type: ignore[union-attr]
        return self.real.store.get(job_id).options

    def stored_files(self, job_id: str) -> list[tuple[str, int]]:
        if self.real is None:
            return sorted(self.backend._jobs[job_id].files)  # type: ignore[union-attr]
        return sorted((f.name, f.size) for f in self.real.store.get(job_id).input_files)


def _set_status(status: str, error: Optional[JobError] = None):
    def change(record) -> None:
        record.status = status
        if error is not None:
            record.error = error

    return change


def assert_error(response, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status, response.text
    body = response.json()
    ErrorResponse.model_validate(body)
    assert body["error"]["code"] == code
    return body["error"]


@pytest.fixture
def api_factory(backend_kind: str, tmp_path, monkeypatch) -> Iterator[Callable[..., Api]]:
    # The real backend checks OCR availability; make it independent of the local easyocr install.
    from qgen.extraction import text_extraction

    monkeypatch.setattr(text_extraction, "ocr_available", lambda: True)
    clients: list[TestClient] = []

    def factory(**overrides: Any) -> Api:
        clock = FakeClock()
        notifier = RecordingNotifier()
        real: Optional[RealParts] = None
        if backend_kind == "mock":
            settings = make_settings(**overrides)
            backend: Union[MockBackend, RealBackend] = MockBackend(settings, notifier, clock=clock, tick_interval_s=None)
        else:
            overrides.setdefault("data_dir", tmp_path / f"data-{len(clients)}")
            settings = make_settings(backend="real", **overrides)
            store = JobStore.from_settings(settings, clock=clock, lock_timeout_s=5)
            redis_client = fakeredis.FakeRedis()
            real = RealParts(store, redis_client, InMemoryJobQueue(), ExternalRefs(redis_client))
            backend = RealBackend(settings, store, real.queue, real.refs, real.events)
        client = TestClient(create_app(settings, backend), raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)
        return Api(backend_kind, client, backend, clock, notifier, settings, {"X-API-Key": API_KEY}, real)

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def api(api_factory) -> Api:
    return api_factory()
