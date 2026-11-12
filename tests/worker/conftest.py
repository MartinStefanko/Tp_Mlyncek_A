"""Worker tests: API (RealBackend + RqJobQueue) and RQ SimpleWorker over fakeredis.

No real LLM / embedding API: FakeLLM with a scripted model, or the replay client over
baseline_v6 (skipped when the recording is missing). PDF extraction is replaced by fixed
segments unless a test needs the real one.

Webhooks: the API and the worker queue them on "qgen-webhooks" (RqWebhookNotifier);
`env.hooks.deliver()` runs the webhook worker (SimpleWorker over that queue) against a
fake Laravel (httpx.MockTransport) with a fake Unix clock. Retries planned by enqueue_in
wait in RQ's ScheduledJobRegistry; `env.hooks.release_scheduled()` moves them to the queue.
"""

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import fakeredis
import httpx
import pytest
from fastapi.testclient import TestClient
from rq import Queue, SimpleWorker
from rq.job import Job
from rq.registry import ScheduledJobRegistry

from qgen.llm import retry
from qgen.progress import STAGE_EXTRACT
from service.jobs import JobStore
from service.jobs import tasks
from service.jobs.refs import ExternalRefs
from service.jobs.rq_queue import RqJobQueue
from service.jobs.runner import JobRunner
from service.main import create_app
from service.real_backend import RealBackend
from service.settings import ServiceSettings
from service.webhooks import rq_delivery
from service.webhooks.rq_delivery import RqWebhookNotifier
from tests.fakes import (
    EMPTY_ITEMS,
    FakeEmbedder,
    FakeLLM,
    check_answer,
    is_coverage_call,
    is_extra_round,
    is_question_generation,
    items,
    question,
    section_ids,
)
from tests.service.conftest import make_pdf

API_KEY = "test-key"
HEADERS = {"X-API-Key": API_KEY}
WEBHOOK_SECRET = "test-secret"
CALLBACK_URL = "http://eceh-backend/api/internal/qgen/webhook"
EKO_PDF = Path(__file__).resolve().parent.parent / "fixtures" / "eko.pdf"

SEGMENTS = [
    {"page": 1, "source_id": "D1", "source_name": "bio.pdf",
     "text": "Dokument D1: bio.pdf\nStrana 1\nFotosyntéza je proces, pri ktorom rastliny tvoria cukor."},
    {"page": 2, "source_id": "D1", "source_name": "bio.pdf",
     "text": "Dokument D1: bio.pdf\nStrana 2\nDýchanie rastlín uvoľňuje energiu z cukru."},
    {"page": 3, "source_id": "D1", "source_name": "bio.pdf",
     "text": "Dokument D1: bio.pdf\nStrana 3\nTranspirácia je odparovanie vody listami."},
]
SECTION_NAMES = ((1, "Fotosyntéza"), (2, "Dýchanie"), (3, "Transpirácia"))


# --- scripted model (same prompt markers as tests/test_generation_coverage.py) -------------


def scripted_answer(prompt: str, model: str) -> str:
    """Three sections; per section 3 questions (choice, choice, open) in the first call,
    nothing new in the extra round; every check scores 5."""
    if is_question_generation(prompt):
        if is_extra_round(prompt) or is_coverage_call(prompt):
            return EMPTY_ITEMS
        rows = []
        for i in section_ids(prompt):
            rows += [question(i, 1, hints=["<p>Pozri <b>definíciu</b>.</p>"]), question(i, 2),
                     question(i, 3, "open")]
        return items(rows)
    if "extrahuj merateľné vzdelávacie ciele" in prompt:  # section generation
        return items(
            {"vzdelávací_objekt": name, "bloom_level": "Pochopiť", "odporúčané_aktivity": ["čítanie"],
             "odporúčané_zadania": "Vysvetli.", "citovane_zdroje": [f"D1:{page}"]}
            for page, name in SECTION_NAMES
        )
    return check_answer(prompt) or EMPTY_ITEMS


class OrthogonalVectors:
    """Embedding answer: every distinct text gets its own one-hot vector (no clustering)."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.index: dict[str, int] = {}

    def __call__(self, texts, model):
        vectors = []
        for text in texts:
            position = self.index.setdefault(text, len(self.index) % self.dim)
            vectors.append([1.0 if i == position else 0.0 for i in range(self.dim)])
        return vectors


def fixed_extraction(segments: list[dict]) -> Callable[..., list[dict]]:
    def extract(inputs, ocr=None, progress_callback=None, settings=None):
        if progress_callback:
            progress_callback(STAGE_EXTRACT, 0, len(segments))
            progress_callback(STAGE_EXTRACT, len(segments), len(segments))
        return [dict(segment) for segment in segments]

    return extract


# --- webhooks ------------------------------------------------------------------------------


class UnixClock:
    def __init__(self, start: float = 1_793_701_125.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


@dataclass
class FakeLaravel:
    """Records every webhook request; answers with `statuses` in turn, then 200."""

    statuses: list[int] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.statuses.pop(0) if self.statuses else 200)

    def payloads(self) -> list[dict]:
        return [json.loads(request.content) for request in self.requests]

    def events(self) -> list[str]:
        return [request.headers["X-QGen-Event"] for request in self.requests]


@dataclass
class WebhookEnv:
    redis: fakeredis.FakeRedis
    laravel: FakeLaravel
    clock: UnixClock
    context: rq_delivery.DeliveryContext

    @property
    def queue(self) -> Queue:
        return self.context.queue

    def deliver(self) -> None:
        """The webhook worker processes everything that waits in "qgen-webhooks"."""
        SimpleWorker([self.queue], connection=self.redis).work(burst=True)

    def release_scheduled(self) -> list[float]:
        """Move planned retries to the queue (as the RQ scheduler does); their delays in s."""
        registry = ScheduledJobRegistry(queue=self.queue)
        delays = []
        for job_id in registry.get_job_ids():
            job = Job.fetch(job_id, connection=self.redis)
            # RQ stores the time in whole seconds (rounded down): ceil gives the planned delay
            delays.append(math.ceil(registry.get_scheduled_time(job).timestamp() - job.created_at.timestamp()))
            registry.remove(job)
            self.queue._enqueue_job(job)
        return delays


# --- environment ---------------------------------------------------------------------------


@dataclass
class WorkerEnv:
    client: TestClient
    settings: ServiceSettings
    redis: fakeredis.FakeRedis
    store: JobStore
    queue: RqJobQueue
    refs: ExternalRefs
    context: tasks.WorkerContext
    hooks: WebhookEnv

    @property
    def runner(self) -> JobRunner:
        return self.context.runner

    def use(self, **runner_fields: Any) -> None:
        """Change fields of the runner (llm, embedder, extract, monotonic, settings …)."""
        for name, value in runner_fields.items():
            setattr(self.context.runner, name, value)

    def create_response(self, options: Optional[dict] = None, pdf: bytes = b"", name: str = "bio.pdf") -> httpx.Response:
        data = {"options": json.dumps(options)} if options is not None else {}
        return self.client.post(
            "/v1/jobs",
            headers=HEADERS,
            files=[("files", (name, pdf or make_pdf(3), "application/pdf"))],
            data=data,
        )

    def create(self, options: Optional[dict] = None, pdf: bytes = b"", name: str = "bio.pdf") -> str:
        response = self.create_response(options, pdf, name)
        assert response.status_code == 202, response.text
        return response.json()["job_id"]

    def work(self) -> None:
        worker = SimpleWorker([self.queue.queue], connection=self.redis)
        worker.work(burst=True)

    def status(self, job_id: str) -> dict:
        response = self.client.get(f"/v1/jobs/{job_id}", headers=HEADERS)
        assert response.status_code == 200, response.text
        return response.json()


def worker_settings(tmp_path: Path, **overrides: Any) -> ServiceSettings:
    values: dict[str, Any] = {
        "backend": "real",
        "api_keys": API_KEY,
        "data_dir": tmp_path / "data",
        "webhook_allowed_hosts": "eceh-backend",
        "webhook_secret": WEBHOOK_SECRET,
    }
    values.update(overrides)
    return ServiceSettings(_env_file=None, **values)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(retry, "_sleep", lambda seconds: None)


@pytest.fixture
def env_factory(tmp_path, monkeypatch) -> Iterator[Callable[..., WorkerEnv]]:
    from qgen.extraction import text_extraction

    monkeypatch.setattr(text_extraction, "ocr_available", lambda: True)
    clients: list[TestClient] = []

    def factory(**settings_overrides: Any) -> WorkerEnv:
        settings = worker_settings(tmp_path, **settings_overrides)
        client_redis = fakeredis.FakeRedis()
        store = JobStore.from_settings(settings, lock_timeout_s=10)
        queue = RqJobQueue(client_redis, job_timeout_s=settings.job_timeout_s)
        refs = ExternalRefs(client_redis)
        events = RqWebhookNotifier(client_redis, settings)
        runner = JobRunner(
            store,
            settings,
            llm=FakeLLM(default=scripted_answer),
            embedder=FakeEmbedder(default=OrthogonalVectors()),
            progress_interval_s=0,
            extract=fixed_extraction(SEGMENTS),
            events=events,
        )
        context = tasks.WorkerContext(settings, client_redis, store, queue, refs, runner, events)
        tasks.configure(context)
        laravel, clock = FakeLaravel(), UnixClock()
        delivery = rq_delivery.build_context(
            settings, client_redis, transport=httpx.MockTransport(laravel), unix_time=clock
        )
        rq_delivery.configure(delivery)
        backend = RealBackend(settings, store, queue, refs, events)
        client = TestClient(create_app(settings, backend), raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)
        hooks = WebhookEnv(client_redis, laravel, clock, delivery)
        return WorkerEnv(client, settings, client_redis, store, queue, refs, context, hooks)

    yield factory
    for client in clients:
        client.__exit__(None, None, None)
    tasks.configure(None)
    rq_delivery.configure(None)


@pytest.fixture
def env(env_factory) -> WorkerEnv:
    return env_factory()
