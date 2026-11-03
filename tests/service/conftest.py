"""Helpers for API tests: fake clock, recording webhook notifier, app factory."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator, Optional

import pytest
from fastapi.testclient import TestClient

from qgen.contract import ErrorResponse, WebhookPayload
from service.main import create_app
from service.mock import MockBackend
from service.settings import ServiceSettings

PDF = b"%PDF-1.7\n%mock\n"
API_KEY = "test-key"
CALLBACK_URL = "http://eceh-backend/api/internal/qgen/webhook"
DURATION_S = 30.0


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
class Api:
    client: TestClient
    backend: MockBackend
    clock: FakeClock
    notifier: RecordingNotifier
    settings: ServiceSettings
    headers: dict[str, str]

    def create(
        self,
        options: Optional[dict[str, Any]] = None,
        files: Optional[list[tuple[str, tuple[str, bytes, str]]]] = None,
        headers: Optional[dict[str, str]] = None,
    ):
        import json

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

    def run_to_end(self, factor: float = 1.0) -> None:
        self.clock.advance(DURATION_S * factor + 1)
        self.backend.tick()


def assert_error(response, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status, response.text
    body = response.json()
    ErrorResponse.model_validate(body)
    assert body["error"]["code"] == code
    return body["error"]


@pytest.fixture
def api_factory() -> Iterator[Callable[..., Api]]:
    clients: list[TestClient] = []

    def factory(**overrides: Any) -> Api:
        settings = make_settings(**overrides)
        clock = FakeClock()
        notifier = RecordingNotifier()
        backend = MockBackend(settings, notifier, clock=clock, tick_interval_s=None)
        client = TestClient(create_app(settings, backend), raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)
        return Api(client, backend, clock, notifier, settings, {"X-API-Key": API_KEY})

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def api(api_factory) -> Api:
    return api_factory()
