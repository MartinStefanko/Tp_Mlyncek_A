import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from qgen.contract import (
    API_ERROR_CODES,
    ErrorResponse,
    JobStatus,
    Report,
    WebhookPayload,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Example from Príloha A – Report.
REPORT_EXAMPLE = {
    "learning_objects": [
        {
            "id": "LO-12",
            "title": "Akumulovaná hodnota",
            "statement": "Akumulovaná hodnota je súčet istiny a úroku ...",
            "lo_type": "definition",
            "source": {"file": "eko.pdf", "pages": [4, 5]},
            "keywords": ["úrok", "istina"],
            "bloom_level": "understand",
            "topic": "Hodnota peňazí v čase",
        }
    ],
    "question_map": [{"unit": 0, "question": 0, "lo_id": "LO-12"}],
    "quality": {"faithfulness_avg": 0.91, "answerability_avg": 0.95, "topic_coverage": 0.78},
    "warnings": [{"code": "field_too_long", "message": "Otázka vyradená: zadanie má viac ako 4000 znakov."}],
    "timings": {"extract": 12.4, "learning_objects": 95.1, "questions": 210.7, "split": 3.2},
}

# Example from Príloha A – stav úlohy.
STATUS_EXAMPLE = {
    "job_id": "7f3c2a1e",
    "external_ref": "gen-01JB7Q3K",
    "status": "running",
    "stage": "questions",
    "progress": 0.62,
    "created_at": "2026-11-03T10:12:01Z",
    "updated_at": "2026-11-03T10:16:40Z",
    "error": None,
    "retryable_until": None,
}

# Example from Príloha A – Webhook body.
WEBHOOK_EXAMPLE = {
    "event": "job.completed",
    "job_id": "7f3c2a1e",
    "external_ref": "gen-01JB7Q3K",
    "status": "completed",
    "stage": "output",
    "progress": 1.0,
    "error": None,
    "occurred_at": "2026-11-03T10:21:45Z",
}


# --- report ---------------------------------------------------------------------------


def test_report_example_passes_model_and_schema():
    Report.model_validate(REPORT_EXAMPLE)
    schema = json.loads((REPO_ROOT / "schemas" / "report.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(REPORT_EXAMPLE)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["learning_objects"][0].update(id="12"),
        lambda r: r["learning_objects"][0].update(extra=1),
        lambda r: r["question_map"][0].update(unit=-1),
        lambda r: r["quality"].update(topic_coverage=77.78),
        lambda r: r["timings"].update(split=-0.1),
        lambda r: r.update(extra=[]),
    ],
)
def test_invalid_report_rejected(mutate):
    data = json.loads(json.dumps(REPORT_EXAMPLE))
    mutate(data)
    with pytest.raises(ValidationError):
        Report.model_validate(data)


# --- job status / webhook ---------------------------------------------------------------


def test_status_example_passes_and_serializes_utc_with_z():
    status = JobStatus.model_validate(STATUS_EXAMPLE)
    assert status.model_dump(mode="json") == STATUS_EXAMPLE


def test_failed_status_with_llm_error():
    status = JobStatus.model_validate(
        {
            **STATUS_EXAMPLE,
            "status": "failed",
            "error": {"code": "llm_error", "message": "LLM provider failed after retries."},
            "retryable_until": "2026-11-04T10:16:40Z",
        }
    )
    assert status.error.code == "llm_error"
    assert status.error.details == {}


@pytest.mark.parametrize(
    "change",
    [
        {"status": "done"},
        {"stage": "upload"},
        {"progress": 1.2},
        {"created_at": "2026-11-03T10:12:01"},  # naive datetime
        {"foo": "bar"},
    ],
)
def test_invalid_status_rejected(change):
    with pytest.raises(ValidationError):
        JobStatus.model_validate({**STATUS_EXAMPLE, **change})


def test_queued_status_without_stage():
    JobStatus.model_validate({**STATUS_EXAMPLE, "status": "queued", "stage": None, "progress": 0.0})


def test_webhook_example_passes():
    payload = WebhookPayload.model_validate(WEBHOOK_EXAMPLE)
    assert payload.model_dump(mode="json") == WEBHOOK_EXAMPLE


@pytest.mark.parametrize("change", [{"event": "job.started"}, {"result": {}}])
def test_invalid_webhook_rejected(change):
    with pytest.raises(ValidationError):
        WebhookPayload.model_validate({**WEBHOOK_EXAMPLE, **change})


# --- error response ---------------------------------------------------------------------


def test_error_response_shape():
    body = ErrorResponse.model_validate(
        {"error": {"code": "invalid_options", "message": "mode=syllabus needs a syllabus", "details": {"field": "split"}}}
    )
    assert body.model_dump() == {
        "error": {"code": "invalid_options", "message": "mode=syllabus needs a syllabus", "details": {"field": "split"}}
    }
    assert ErrorResponse.model_validate({"error": {"code": "job_not_found", "message": "x"}}).error.details == {}


def test_llm_error_is_not_an_api_error_code():
    assert "llm_error" not in API_ERROR_CODES
    with pytest.raises(ValidationError):
        ErrorResponse.model_validate({"error": {"code": "llm_error", "message": "x"}})


# --- exported schemas are up to date --------------------------------------------------------


def test_exported_schemas_are_up_to_date():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "export_schemas.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
