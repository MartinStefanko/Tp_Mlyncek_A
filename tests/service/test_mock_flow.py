"""Whole flow POST → state → result → report, simulation of steps, list by external_ref."""

import json
from datetime import timedelta
from pathlib import Path

import pytest

from qgen.contract import CreateJobResponse, JobListResponse, JobStatus, Report, UnitsResult
from service.mock.backend import STAGES

from .conftest import CALLBACK_URL, DURATION_S, PDF, assert_error

CONTRACT_FIXTURES = Path(__file__).resolve().parents[1] / "contract" / "fixtures"
MOCK_FIXTURES = Path(__file__).resolve().parents[2] / "service" / "mock" / "fixtures"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_full_flow(api):
    response = api.create({"external_ref": "gen-1", "callback_url": CALLBACK_URL, "split": {"mode": "topics"}})
    assert response.status_code == 202
    body = CreateJobResponse.model_validate(response.json())
    assert body.status == "queued"
    assert body.replayed is False
    assert response.json()["links"] == {"self": f"/v1/jobs/{body.job_id}"}
    job_id = body.job_id

    status = JobStatus.model_validate(api.status(job_id))
    assert (status.status, status.stage, status.progress) == ("queued", None, 0.0)
    assert status.external_ref == "gen-1"

    assert_error(api.client.get(f"/v1/jobs/{job_id}/result", headers=api.headers), 409, "result_not_ready")
    assert_error(api.client.get(f"/v1/jobs/{job_id}/report", headers=api.headers), 409, "result_not_ready")

    seen_stages = []
    last_progress = 0.0
    for _ in range(60):
        api.clock.advance(DURATION_S / 50)
        state = api.status(job_id)
        if state["status"] == "completed":
            break
        assert state["status"] in ("queued", "running")
        if state["status"] == "queued":
            assert state["stage"] is None
        else:
            assert state["stage"] in STAGES
            if not seen_stages or seen_stages[-1] != state["stage"]:
                seen_stages.append(state["stage"])
        assert 0.0 <= state["progress"] < 1.0
        assert state["progress"] >= last_progress
        last_progress = state["progress"]
    assert seen_stages == list(STAGES)

    final = api.status(job_id)
    assert (final["status"], final["stage"], final["progress"], final["error"]) == ("completed", "output", 1.0, None)

    result = api.client.get(f"/v1/jobs/{job_id}/result", headers=api.headers)
    assert result.status_code == 200
    assert result.json() == _load(CONTRACT_FIXTURES / "units_example.json")
    UnitsResult.model_validate(result.json())

    report = api.client.get(f"/v1/jobs/{job_id}/report", headers=api.headers)
    assert report.status_code == 200
    assert report.json() == _load(CONTRACT_FIXTURES / "report_example.json")
    assert set(report.json()) == {"sections", "question_meta", "quality", "warnings", "timings"}
    Report.model_validate(report.json())

    assert api.notifier.events(job_id) == ["job.stage_changed"] * len(STAGES) + ["job.completed"]


def test_job_completes_after_mock_duration(api):
    job_id = api.create_id({"external_ref": "gen-time"})
    api.clock.advance(DURATION_S - 0.5)
    assert api.status(job_id)["status"] == "running"
    api.clock.advance(1)
    status = api.status(job_id)
    assert status["status"] == "completed"
    assert status["updated_at"] == "2026-11-03T10:00:30Z"


def test_slow_job_takes_five_times_longer(api):
    job_id = api.create_id({"external_ref": "mock-slow-1"})
    api.clock.advance(DURATION_S * 4.9)
    assert api.status(job_id)["status"] == "running"
    api.clock.advance(DURATION_S * 0.2)
    assert api.status(job_id)["status"] == "completed"


def test_mock_duration_setting(api_factory):
    api = api_factory(mock_duration_s=4)
    job_id = api.create_id()
    api.clock.advance(4.1)
    assert api.status(job_id)["status"] == "completed"


def test_job_without_external_ref(api):
    first = api.create_id()
    second = api.create_id()
    assert first != second
    assert api.status(first)["external_ref"] is None


def test_list_by_external_ref_newest_first(api):
    first = api.create_id({"external_ref": "gen-list"})
    api.run_to_end()
    assert api.client.delete(f"/v1/jobs/{first}", headers=api.headers).status_code == 204
    second = api.create_id({"external_ref": "gen-list"})
    api.create_id({"external_ref": "other"})

    response = api.client.get("/v1/jobs", params={"external_ref": "gen-list"}, headers=api.headers)
    assert response.status_code == 200
    jobs = JobListResponse.model_validate(response.json()).jobs
    assert [j.job_id for j in jobs] == [second]
    assert set(response.json()["jobs"][0]) == {"job_id", "status", "created_at"}

    empty = api.client.get("/v1/jobs", params={"external_ref": "none"}, headers=api.headers)
    assert empty.json() == {"jobs": []}


def test_stored_options_are_complete(api):
    job_id = api.create_id({"external_ref": "gen-opt", "questions": {"types": ["open"]}})
    stored = api.backend._jobs[job_id].options
    assert stored.model_dump(mode="json") == {
        "external_ref": "gen-opt",
        "callback_url": None,
        "language": "auto",
        "extraction": {"ocr": False},
        "questions": {"types": ["open"], "per_section": 3, "choice_options": 4, "hints": 2},
        "split": {"mode": "single", "lesson_count": None, "lesson_minutes": 45, "syllabus": None},
    }


def test_defaults_come_from_settings(api_factory):
    api = api_factory(default_lesson_minutes=90, ocr_enabled=True)
    job_id = api.create_id({"split": {"mode": "topics"}})
    stored = api.backend._jobs[job_id].options
    assert stored.split.lesson_minutes == 90
    assert stored.extraction.ocr is True


def test_multiple_files_and_files_brackets_field(api):
    files = [
        ("files[]", ("a.pdf", PDF, "application/pdf")),
        ("files[]", ("b.pdf", PDF + b"x", "application/pdf")),
        ("files", ("c.pdf", PDF, "application/pdf")),
    ]
    response = api.create({"external_ref": "gen-multi"}, files=files)
    assert response.status_code == 202, response.text
    stored = api.backend._jobs[response.json()["job_id"]]
    assert sorted(stored.files) == [("a.pdf", len(PDF)), ("b.pdf", len(PDF) + 1), ("c.pdf", len(PDF))]


def test_options_field_may_be_missing(api):
    response = api.client.post("/v1/jobs", headers=api.headers, files=[("files", ("eko.pdf", PDF, "application/pdf"))])
    assert response.status_code == 202


def test_mock_fixtures_equal_contract_fixtures():
    for name in ("units_example.json", "report_example.json"):
        assert _load(MOCK_FIXTURES / name) == _load(CONTRACT_FIXTURES / name), name


def test_status_timestamps_are_utc(api):
    job_id = api.create_id()
    status = api.status(job_id)
    assert status["created_at"].endswith("Z")
    created = JobStatus.model_validate(status).created_at
    assert created.utcoffset() == timedelta(0)


@pytest.mark.parametrize("path", ["/v1/schema/lesson", "/v1/schema/report"])
def test_schema_endpoints_equal_exported_files(api, path):
    name = path.rsplit("/", 1)[1]
    exported = _load(Path(__file__).resolve().parents[2] / "schemas" / f"{name}.schema.json")
    response = api.client.get(path, headers=api.headers)
    assert response.status_code == 200
    assert response.json() == exported


def test_health_info_docs(api):
    health = api.client.get("/health")  # no API key
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    info = api.client.get("/v1/info", headers=api.headers)
    assert info.status_code == 200
    body = info.json()
    assert body["backend"] == "mock"
    assert body["limits"]["max_files"] == 10
    assert body["option_defaults"]["lesson_minutes"] == 45
    assert "test-key" not in info.text and "test-secret" not in info.text

    assert api.client.get("/docs").status_code == 200
    paths = api.client.get("/openapi.json").json()["paths"]
    assert set(paths) == {
        "/health",
        "/v1/info",
        "/v1/jobs",
        "/v1/jobs/{job_id}",
        "/v1/jobs/{job_id}/result",
        "/v1/jobs/{job_id}/report",
        "/v1/jobs/{job_id}/retry",
        "/v1/schema/lesson",
        "/v1/schema/report",
    }
    assert set(paths["/v1/jobs/{job_id}"]) == {"get", "delete"}
