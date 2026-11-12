"""Every error code of Príloha A (plus not_found / method_not_allowed / internal_error)."""

import pytest

from qgen.contract import API_ERROR_CODES
from service.main import create_app

from .conftest import CALLBACK_URL, PDF, assert_error, make_pdf

PDF_FILE = [("files", ("eko.pdf", PDF, "application/pdf"))]


def _post_raw_options(api, raw: str):
    return api.client.post("/v1/jobs", headers=api.headers, files=PDF_FILE, data={"options": raw})


# --- 400 invalid_options --------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "{bad json",
        "[]",
        '"text"',
        '{"language": "de"}',
        '{"questions": {"per_section": "three"}}',
        '{"questions": {"per_learning_object": 3}}',
        '{"split": {"mode": "syllabus"}}',
        '{"split": {"mode": "syllabus", "syllabus": [{"title": "A"}], "syllabus_text": "2 hodiny"}}',
        '{"split": {"mode": "syllabus", "syllabus_text": " "}}',
        '{"split": {"mode": "syllabus", "syllabus": [{"title": "A", "pages": "D1:9-2"}]}}',
        '{"split": {"mode": "syllabus", "syllabus": [{"title": "A", "pages": "D2:1-5"}]}}',
        '{"split": {"mode": "topics", "lesson_count": 4}}',
        '{"callback_url": "ftp://eceh-backend/hook"}',
        '{"unknown": 1}',
    ],
)
def test_invalid_options(api, raw):
    error = assert_error(_post_raw_options(api, raw), 400, "invalid_options")
    assert error["details"]["errors"]
    for item in error["details"]["errors"]:
        assert set(item) == {"loc", "msg", "type"}
    assert api.job_count() == 0


def test_invalid_options_details_point_to_field(api):
    error = assert_error(_post_raw_options(api, '{"split": {"mode": "syllabus"}}'), 400, "invalid_options")
    assert error["details"]["errors"][0]["loc"][:2] == ["split", "syllabus"]


def test_syllabus_pages_must_name_uploaded_documents(api):
    raw = '{"split": {"mode": "syllabus", "syllabus": [{"title": "A", "pages": "D1:1-3"}, {"title": "B", "pages": "D1:4, D3:1"}]}}'
    error = assert_error(_post_raw_options(api, raw), 400, "invalid_options")
    assert error["details"]["errors"] == [
        {"loc": ["split", "syllabus", 1, "pages"], "msg": "Document D3 does not exist; 1 file(s) uploaded.", "type": "unknown_document"}
    ]
    two_files = [("files", ("a.pdf", PDF, "application/pdf")), ("files", ("b.pdf", PDF, "application/pdf"))]
    ok = '{"split": {"mode": "syllabus", "syllabus": [{"title": "A", "pages": "D2:1-3"}]}}'
    response = api.client.post("/v1/jobs", headers=api.headers, files=two_files, data={"options": ok})
    assert response.status_code == 202


def test_syllabus_text_accepted(api):
    raw = '{"split": {"mode": "syllabus", "syllabus_text": "3 hodiny: najprv teória, potom príklady"}}'
    assert _post_raw_options(api, raw).status_code == 202


def test_callback_host_not_allowed(api):
    error = assert_error(
        _post_raw_options(api, '{"callback_url": "http://evil.example.com/hook"}'), 400, "invalid_options"
    )
    assert error["details"]["errors"] == [
        {"loc": ["callback_url"], "msg": "Host 'evil.example.com' is not in QGEN_WEBHOOK_ALLOWED_HOSTS.", "type": "host_not_allowed"}
    ]


def test_callback_rejected_when_webhooks_disabled(api_factory):
    api = api_factory(webhook_allowed_hosts="")
    assert_error(_post_raw_options(api, f'{{"callback_url": "{CALLBACK_URL}"}}'), 400, "invalid_options")


def test_callback_host_with_port(api_factory):
    api = api_factory(webhook_allowed_hosts="eceh-backend:8080, Other-Host")
    assert _post_raw_options(api, '{"callback_url": "http://eceh-backend:8080/hook"}').status_code == 202
    assert _post_raw_options(api, '{"callback_url": "https://other-host/hook"}').status_code == 202
    assert_error(_post_raw_options(api, '{"callback_url": "http://eceh-backend:9090/hook"}'), 400, "invalid_options")


def test_options_are_checked_before_files(api):
    response = api.client.post(
        "/v1/jobs", headers=api.headers, files=[("files", ("a.txt", b"hello", "text/plain"))], data={"options": "{"}
    )
    assert_error(response, 400, "invalid_options")


# --- 401 unauthorized -----------------------------------------------------------------------


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}])
def test_unauthorized(api, headers):
    assert_error(api.create(headers=headers), 401, "unauthorized")
    assert_error(api.client.get("/v1/info", headers=headers), 401, "unauthorized")
    assert_error(api.client.get("/v1/schema/lesson", headers=headers), 401, "unauthorized")
    assert_error(api.client.get("/v1/jobs", params={"external_ref": "x"}, headers=headers), 401, "unauthorized")


def test_no_key_needed_when_api_keys_empty(api_factory):
    api = api_factory(api_keys="")
    assert api.create(headers={}).status_code == 202
    assert api.client.get("/v1/info").status_code == 200


def test_any_of_several_keys(api_factory):
    api = api_factory(api_keys="a, b")
    assert api.create(headers={"X-API-Key": "b"}).status_code == 202
    assert_error(api.create(headers={"X-API-Key": "a, b"}), 401, "unauthorized")


# --- 404 / 409 --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "suffix"),
    [("get", ""), ("get", "/result"), ("get", "/report"), ("delete", ""), ("post", "/retry")],
)
def test_job_not_found(api, method, suffix):
    error = assert_error(api.client.request(method, f"/v1/jobs/nope{suffix}", headers=api.headers), 404, "job_not_found")
    assert error["details"] == {"job_id": "nope"}


def test_result_not_ready_and_job_not_completed(api):
    running = api.create_id()
    assert_error(api.client.get(f"/v1/jobs/{running}/result", headers=api.headers), 409, "result_not_ready")
    api.client.delete(f"/v1/jobs/{running}", headers=api.headers)
    assert_error(api.client.get(f"/v1/jobs/{running}/result", headers=api.headers), 409, "job_not_completed")


def test_job_not_retryable(api):
    job_id = api.create_id()
    assert_error(api.client.post(f"/v1/jobs/{job_id}/retry", headers=api.headers), 409, "job_not_retryable")


# --- 413 / 415 / 422 --------------------------------------------------------------------------


def test_file_too_large(api_factory):
    ok = make_pdf(pad=1000)
    big = make_pdf(pad=2000)
    api = api_factory(max_upload_mb=(len(ok) + 10) / (1024 * 1024))
    error = assert_error(api.create(files=[("files", ("big.pdf", big, "application/pdf"))]), 413, "file_too_large")
    assert error["details"]["file"] == "big.pdf"
    assert api.create(files=[("files", ("ok.pdf", ok, "application/pdf"))]).status_code == 202


def test_too_many_files(api_factory):
    api = api_factory(max_files=2)
    files = [("files", (f"{i}.pdf", PDF, "application/pdf")) for i in range(3)]
    error = assert_error(api.create(files=files), 413, "file_too_large")
    assert error["details"] == {"max_files": 2, "files": 3}
    assert api.create(files=files[:2]).status_code == 202


@pytest.mark.parametrize(
    "content",
    [b"hello", b"", b" %PDF-1.7", b"PK\x03\x04", b"%PD"],
)
def test_unsupported_file(api, content):
    files = [("files", ("ok.pdf", PDF, "application/pdf")), ("files", ("x.pdf", content, "application/pdf"))]
    error = assert_error(api.create(files=files), 415, "unsupported_file")
    assert error["details"] == {"file": "x.pdf"}


def test_missing_files_is_validation_error(api):
    response = api.client.post("/v1/jobs", headers=api.headers, data={"options": "{}"})
    error = assert_error(response, 422, "validation_error")
    assert error["details"]["errors"][0]["loc"] == ["body", "files"]


# Swagger UI sends an empty text part for a file field left empty (curl -F 'files[]=').
# multipart parts with filename None are text parts.
EMPTY_FILES = ("files", (None, b""))
EMPTY_FILES_BRACKETS = ("files[]", (None, b""))


@pytest.mark.parametrize("empty", [EMPTY_FILES, EMPTY_FILES_BRACKETS, None], ids=["files", "files[]", "both"])
def test_only_empty_text_parts_is_missing_files(api, empty):
    parts = [empty] if empty else [EMPTY_FILES, EMPTY_FILES_BRACKETS]
    response = api.client.post("/v1/jobs", headers=api.headers, files=[*parts, ("options", (None, b"{}"))])
    error = assert_error(response, 422, "validation_error")
    assert error["details"]["errors"][0]["loc"] == ["body", "files"]
    assert api.job_count() == 0


def test_file_in_files_and_empty_files_brackets(api):
    response = api.client.post("/v1/jobs", headers=api.headers, files=[*PDF_FILE, EMPTY_FILES_BRACKETS])
    assert response.status_code == 202, response.text
    assert api.stored_files(response.json()["job_id"]) == [("eko.pdf", len(PDF))]


def test_file_in_files_brackets_and_empty_files(api):
    files = [EMPTY_FILES, ("files[]", ("eko.pdf", PDF, "application/pdf"))]
    response = api.client.post("/v1/jobs", headers=api.headers, files=files)
    assert response.status_code == 202, response.text


@pytest.mark.parametrize("field", ["files", "files[]"])
def test_text_instead_of_a_file_is_validation_error(api, field):
    response = api.client.post("/v1/jobs", headers=api.headers, files=[*PDF_FILE, (field, (None, b"not a file"))])
    error = assert_error(response, 422, "validation_error")
    loc = error["details"]["errors"][0]["loc"]
    assert loc[:2] == ["body", field]
    assert "Expected UploadFile" in error["details"]["errors"][0]["msg"]
    assert api.job_count() == 0


def test_missing_external_ref_query_is_validation_error(api):
    assert_error(api.client.get("/v1/jobs", headers=api.headers), 422, "validation_error")


# --- 429 --------------------------------------------------------------------------------------


def test_too_many_jobs(api_factory):
    api = api_factory(max_active_jobs=2)
    first = api.create_id()
    api.create_id()
    response = api.create()
    assert_error(response, 429, "too_many_jobs")
    assert int(response.headers["Retry-After"]) > 0

    # A finished job frees a slot.
    api.client.delete(f"/v1/jobs/{first}", headers=api.headers)
    assert api.create().status_code == 202


# --- generic errors ---------------------------------------------------------------------------


def test_unknown_path_and_method(api):
    assert_error(api.client.get("/v1/unknown", headers=api.headers), 404, "not_found")
    assert_error(api.client.put("/v1/jobs/x", headers=api.headers), 405, "method_not_allowed")


class _BrokenBackend:
    async def start(self):
        pass

    async def stop(self):
        pass

    def health(self):
        raise RuntimeError("secret detail /data/jobs/123")

    def get(self, job_id, owner):
        raise KeyError("secret detail")


@pytest.mark.mock_only  # independent of the backend
def test_unexpected_error_is_500_without_details(api):
    from fastapi.testclient import TestClient

    app = create_app(api.settings, _BrokenBackend())
    with TestClient(app, raise_server_exceptions=False) as client:
        for response in (client.get("/health"), client.get("/v1/jobs/x", headers=api.headers)):
            error = assert_error(response, 500, "internal_error")
            assert error == {"code": "internal_error", "message": "Internal server error.", "details": {}}
            assert "secret" not in response.text


def test_all_error_codes_are_known():
    assert set(API_ERROR_CODES) == {
        "invalid_options",
        "unauthorized",
        "job_not_found",
        "result_not_ready",
        "job_not_completed",
        "job_not_retryable",
        "file_too_large",
        "unsupported_file",
        "validation_error",
        "too_many_jobs",
        "not_found",
        "method_not_allowed",
        "internal_error",
    }
