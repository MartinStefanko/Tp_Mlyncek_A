"""Worker tests: API (RealBackend + RqJobQueue) and RQ SimpleWorker over fakeredis.

No real LLM / embedding API: FakeLLM with a scripted model, or the replay client over
baseline_v3 (skipped when the recording is missing). PDF extraction is replaced by fixed
segments unless a test needs the real one.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import fakeredis
import pytest
from fastapi.testclient import TestClient
from rq import SimpleWorker

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
from tests.fakes import FakeEmbedder, FakeLLM
from tests.service.conftest import make_pdf

API_KEY = "test-key"
HEADERS = {"X-API-Key": API_KEY}
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


def _section_ids(prompt: str) -> list[int]:
    return [int(x) for x in re.findall(r"- lo_id: (\d+)", prompt)]


def scripted_answer(prompt: str, model: str) -> str:
    """Three sections, one question per section, every check scores 5."""
    if "LO 1:" in prompt and "- lo_id:" in prompt:  # question generation
        return json.dumps([
            {"lo_id": i, "typ": "teoreticka_otazka", "otazka": f"Otázka k celku {i}?", "odpoved": "Odpoveď.",
             "napoveda": "Pozri <b>definíciu</b>.", "citovane_zdroje": [f"D1:{i}"]}
            for i in _section_ids(prompt)
        ], ensure_ascii=False)
    if "extrahuj merateľné vzdelávacie ciele" in prompt:  # section generation
        return json.dumps([
            {"vzdelávací_objekt": name, "bloom_level": "Pochopiť", "odporúčané_aktivity": ["čítanie"],
             "odporúčané_zadania": "Vysvetli.", "citovane_zdroje": [f"D1:{page}"]}
            for page, name in SECTION_NAMES
        ], ensure_ascii=False)
    if "LO ID:" in prompt:  # section faithfulness
        ids = [int(x) for x in re.findall(r"LO ID: (\d+)", prompt)]
        return json.dumps([{"lo_id": i, "skore": 5, "zdovodnenie": "ok"} for i in ids])
    if "Item id:" in prompt:  # question quality score
        ids = [int(x) for x in re.findall(r"Item id: (\d+)", prompt)]
        return json.dumps([{"id": i, "skore": 5, "zdovodnenie": "ok"} for i in ids])
    if "ITEM ID:" in prompt:  # question faithfulness / answerability
        ids = [int(x) for x in re.findall(r"ITEM ID: (\d+)", prompt)]
        return json.dumps([{"item_id": i, "skore": 5, "zdovodnenie": "ok"} for i in ids])
    return "[]"


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

    @property
    def runner(self) -> JobRunner:
        return self.context.runner

    def use(self, **runner_fields: Any) -> None:
        """Change fields of the runner (llm, embedder, extract, monotonic, settings …)."""
        for name, value in runner_fields.items():
            setattr(self.context.runner, name, value)

    def create(self, options: Optional[dict] = None, pdf: bytes = b"", name: str = "bio.pdf") -> str:
        data = {"options": json.dumps(options)} if options is not None else {}
        response = self.client.post(
            "/v1/jobs",
            headers=HEADERS,
            files=[("files", (name, pdf or make_pdf(3), "application/pdf"))],
            data=data,
        )
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
    values: dict[str, Any] = {"backend": "real", "api_keys": API_KEY, "data_dir": tmp_path / "data"}
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
        runner = JobRunner(
            store,
            settings,
            llm=FakeLLM(default=scripted_answer),
            embedder=FakeEmbedder(default=OrthogonalVectors()),
            progress_interval_s=0,
            extract=fixed_extraction(SEGMENTS),
        )
        context = tasks.WorkerContext(settings, client_redis, store, queue, refs, runner)
        tasks.configure(context)
        client = TestClient(create_app(settings, RealBackend(settings, store, queue, refs)), raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)
        return WorkerEnv(client, settings, client_redis, store, queue, refs, context)

    yield factory
    for client in clients:
        client.__exit__(None, None, None)
    tasks.configure(None)


@pytest.fixture
def env(env_factory) -> WorkerEnv:
    return env_factory()
