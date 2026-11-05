"""RealBackend: PDF checks, external_ref reservations in Redis, rollback, DELETE, retry, owner."""

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from qgen.contract import JobOptions, OptionDefaults, resolve_options
from service.backend import CreateJobRequest, UploadedFile
from service.mock.backend import MockFixtures

from .conftest import API_KEY, PDF, assert_error, make_pdf

pytestmark = pytest.mark.real_only


def owner_of(key: str) -> str:
    return "key-" + hashlib.sha256(key.encode()).hexdigest()[:16]


OWNER = owner_of(API_KEY)


def pdf_files(*contents: bytes):
    return [("files", (f"d{i}.pdf", content, "application/pdf")) for i, content in enumerate(contents, start=1)]


def ref_key(external_ref: str, owner: str = OWNER) -> str:
    return f"qgen:extref:{owner}:{external_ref}"


def reserved(api, external_ref: str, owner: str = OWNER):
    value = api.real.redis.get(ref_key(external_ref, owner))
    return value.decode() if value is not None else None


def job_dirs(api) -> list[str]:
    return sorted(p.name for p in api.real.store.jobs_dir.iterdir())


# --- whole flow -----------------------------------------------------------------------------


def test_flow_over_job_store(api):
    response = api.create({"external_ref": "gen-real"}, files=pdf_files(make_pdf(pages=2), make_pdf(pages=3)))
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert response.json()["status"] == "queued"

    record = api.real.store.get(job_id)
    assert [(f.name, f.stored_as, f.pages) for f in record.input_files] == [("d1.pdf", "D1.pdf", 2), ("d2.pdf", "D2.pdf", 3)]
    assert record.owner == OWNER
    assert len(api.real.store.input_paths(job_id)) == 2
    assert api.real.queue.waiting == [job_id]
    assert reserved(api, "gen-real") == job_id

    assert_error(api.client.get(f"/v1/jobs/{job_id}/result", headers=api.headers), 409, "result_not_ready")
    api.start(job_id)
    assert_error(api.client.get(f"/v1/jobs/{job_id}/report", headers=api.headers), 409, "result_not_ready")
    api.finish(job_id)

    fixtures = MockFixtures.load()
    result = api.client.get(f"/v1/jobs/{job_id}/result", headers=api.headers)
    assert result.json() == fixtures.result.model_dump(mode="json")
    report = api.client.get(f"/v1/jobs/{job_id}/report", headers=api.headers)
    assert report.json() == fixtures.report.model_dump(mode="json")


def test_health_reports_redis_and_worker(api, monkeypatch):
    assert api.client.get("/health").json() == {
        "status": "ok",
        "components": {"api": "ok", "redis": "ok", "worker": "ok"},
    }

    api.real.queue.worker_alive = False
    response = api.client.get("/health")
    assert response.status_code == 503
    assert response.json()["components"]["worker"] == "error"
    api.real.queue.worker_alive = True

    def down():
        raise ConnectionError("redis down")

    monkeypatch.setattr(api.real.refs, "ping", down)
    response = api.client.get("/health")
    assert response.status_code == 503
    assert response.json()["components"]["redis"] == "error"


# --- PDF and OCR checks ---------------------------------------------------------------------


def test_password_protected_pdf_is_rejected(api):
    error = assert_error(api.create(files=pdf_files(PDF, make_pdf(password="tajne"))), 415, "unsupported_file")
    assert error["details"] == {"file": "d2.pdf", "reason": "encrypted"}
    assert api.job_count() == 0


def test_unreadable_pdf_is_rejected(api):
    error = assert_error(api.create(files=pdf_files(b"%PDF-1.7\nnot really a pdf")), 415, "unsupported_file")
    assert error["details"] == {"file": "d1.pdf", "reason": "unreadable"}
    assert api.job_count() == 0


def test_too_many_pages_in_all_files_together(api_factory):
    api = api_factory(max_pages=3)
    error = assert_error(api.create(files=pdf_files(make_pdf(pages=2), make_pdf(pages=2))), 413, "file_too_large")
    assert error["details"] == {"max_pages": 3, "pages": 4}
    assert api.job_count() == 0
    assert api.create(files=pdf_files(make_pdf(pages=1), make_pdf(pages=2))).status_code == 202


def test_ocr_requested_but_not_available(api_factory, monkeypatch):
    from qgen.extraction import text_extraction

    api = api_factory()
    monkeypatch.setattr(text_extraction, "ocr_available", lambda: False)
    error = assert_error(api.create({"extraction": {"ocr": True}}), 400, "invalid_options")
    assert error["details"]["errors"][0]["loc"] == ["extraction", "ocr"]
    assert error["details"]["errors"][0]["type"] == "ocr_unavailable"
    assert api.create({"extraction": {"ocr": False}}).status_code == 202

    # The default from QGEN_OCR_ENABLED counts too.
    with_default = api_factory(ocr_enabled=True)
    assert_error(with_default.create(), 400, "invalid_options")


def test_ocr_check_comes_before_file_checks(api, monkeypatch):
    from qgen.extraction import text_extraction

    monkeypatch.setattr(text_extraction, "ocr_available", lambda: False)
    assert_error(api.create({"extraction": {"ocr": True}}, files=pdf_files(b"hello")), 400, "invalid_options")


def test_pdf_check_comes_before_replay(api):
    """Kap. 3: PDFs are checked before the reservation, also for an existing external_ref."""
    api.create_id({"external_ref": "gen-check"})
    assert_error(api.create({"external_ref": "gen-check"}, files=pdf_files(make_pdf(password="x"))), 415, "unsupported_file")


# --- external_ref reservation ---------------------------------------------------------------


def _request(external_ref: str) -> CreateJobRequest:
    options = resolve_options(JobOptions(external_ref=external_ref), OptionDefaults())
    return CreateJobRequest(owner=OWNER, options=options, files=[UploadedFile("eko.pdf", PDF, pages=1)])


def test_concurrent_posts_with_same_external_ref_create_one_job(api):
    count = 8
    barrier = threading.Barrier(count)

    def post(_):
        barrier.wait()
        return api.backend.create(_request("gen-race"))

    with ThreadPoolExecutor(count) as pool:
        results = list(pool.map(post, range(count)))

    job_ids = {r.status.job_id for r in results}
    assert len(job_ids) == 1
    assert sorted(r.replayed for r in results) == [False] + [True] * (count - 1)
    assert api.job_count() == 1
    assert api.real.queue.waiting == list(job_ids)
    assert reserved(api, "gen-race") == job_ids.pop()


def test_concurrent_posts_over_http(api):
    count = 4
    barrier = threading.Barrier(count)

    def post(_):
        barrier.wait()
        return api.create({"external_ref": "gen-http-race"})

    with ThreadPoolExecutor(count) as pool:
        responses = list(pool.map(post, range(count)))
    assert sorted(r.status_code for r in responses) == [200] * (count - 1) + [202]
    assert len({r.json()["job_id"] for r in responses}) == 1
    assert api.job_count() == 1


def test_enqueue_failure_leaves_nothing(api, monkeypatch):
    def broken(job_id):
        raise ConnectionError("queue down")

    monkeypatch.setattr(api.real.queue, "enqueue", broken)
    error = assert_error(api.create({"external_ref": "gen-enq"}), 500, "internal_error")
    assert "queue down" not in json.dumps(error)
    assert api.job_count() == 0
    assert job_dirs(api) == []
    assert reserved(api, "gen-enq") is None

    monkeypatch.undo()
    from qgen.extraction import text_extraction

    monkeypatch.setattr(text_extraction, "ocr_available", lambda: True)
    again = api.create({"external_ref": "gen-enq"})
    assert again.status_code == 202
    assert again.json()["replayed"] is False


def test_store_failure_releases_reservation(api, monkeypatch):
    def broken(**kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(api.real.store, "create", broken)
    assert_error(api.create({"external_ref": "gen-disk"}), 500, "internal_error")
    assert reserved(api, "gen-disk") is None
    assert api.real.queue.waiting == []


def test_redis_down_creates_nothing(api, monkeypatch):
    def down(*args):
        raise ConnectionError("redis down")

    monkeypatch.setattr(api.real.refs, "lookup", down)
    assert_error(api.create({"external_ref": "gen-redis"}), 500, "internal_error")
    assert api.job_count() == 0


def test_stale_reservation_is_replaced(api):
    api.real.refs.force(OWNER, "gen-stale", "deadbeef")  # crash between SET NX and the write
    response = api.create({"external_ref": "gen-stale"})
    assert response.status_code == 202
    assert reserved(api, "gen-stale") == response.json()["job_id"]


def test_reservation_lost_in_redis_is_restored_from_job_json(api):
    job_id = api.create_id({"external_ref": "gen-lost"})
    api.real.redis.delete(ref_key("gen-lost"))
    replay = api.create({"external_ref": "gen-lost"})
    assert replay.status_code == 200
    assert replay.json()["job_id"] == job_id
    assert reserved(api, "gen-lost") == job_id
    assert api.job_count() == 1


def test_replay_does_not_write_pdfs(api):
    job_id = api.create_id({"external_ref": "gen-replay"})
    before = api.real.store.get(job_id)
    assert api.create({"external_ref": "gen-replay"}, files=pdf_files(make_pdf(pages=5))).status_code == 200
    assert api.real.store.get(job_id).input_files == before.input_files
    assert api.real.queue.waiting == [job_id]


# --- DELETE ---------------------------------------------------------------------------------


def test_delete_finished_job_releases_reservation(api):
    job_id = api.create_id({"external_ref": "gen-free"})
    api.finish(job_id)
    assert reserved(api, "gen-free") == job_id
    assert api.client.delete(f"/v1/jobs/{job_id}", headers=api.headers).status_code == 204
    assert reserved(api, "gen-free") is None
    assert job_dirs(api) == []
    assert_error(api.client.get(f"/v1/jobs/{job_id}", headers=api.headers), 404, "job_not_found")


def test_cancel_queued_job_keeps_reservation_and_deletes_pdfs(api):
    job_id = api.create_id({"external_ref": "gen-cancel-q"})
    response = api.client.delete(f"/v1/jobs/{job_id}", headers=api.headers)
    assert response.json()["status"] == "cancelled"
    record = api.real.store.get(job_id)
    assert record.cancel_requested is True
    assert record.finished_at is not None
    assert not api.real.store.has_inputs(job_id)
    assert reserved(api, "gen-cancel-q") == job_id


def test_cancel_running_job_keeps_pdfs_for_the_worker(api):
    job_id = api.create_id()
    api.start(job_id)
    assert api.client.delete(f"/v1/jobs/{job_id}", headers=api.headers).json()["status"] == "cancelled"
    assert api.real.store.get(job_id).cancel_requested is True
    assert api.real.store.has_inputs(job_id)


def test_delete_when_worker_finished_meanwhile(api, monkeypatch):
    """The job was running when DELETE read it, then the worker completed it: no deletion."""
    job_id = api.create_id()
    api.start(job_id)
    store = api.real.store
    original_update = store.update

    def finish_first(jid, change):
        monkeypatch.setattr(store, "update", original_update)
        api.finish(jid)
        return original_update(jid, change)

    monkeypatch.setattr(store, "update", finish_first)
    response = api.client.delete(f"/v1/jobs/{job_id}", headers=api.headers)
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert store.get(job_id).status == "completed"


# --- retry ----------------------------------------------------------------------------------


def test_retry_queues_the_same_job_again(api):
    job_id = api.create_id({"external_ref": "gen-retry"})
    api.fail(job_id)
    response = api.client.post(f"/v1/jobs/{job_id}/retry", headers=api.headers)
    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    record = api.real.store.get(job_id)
    assert (record.status, record.attempts, record.error, record.retryable_until) == ("queued", 0, None, None)
    assert api.real.queue.waiting == [job_id, job_id]


def test_retry_without_pdfs_is_rejected(api):
    job_id = api.create_id()
    api.fail(job_id)
    api.real.store.delete_inputs(job_id)
    assert_error(api.client.post(f"/v1/jobs/{job_id}/retry", headers=api.headers), 409, "job_not_retryable")


def test_retry_enqueue_failure_keeps_job_failed(api, monkeypatch):
    job_id = api.create_id()
    api.fail(job_id)

    def broken(jid):
        raise ConnectionError("queue down")

    monkeypatch.setattr(api.real.queue, "enqueue", broken)
    assert_error(api.client.post(f"/v1/jobs/{job_id}/retry", headers=api.headers), 500, "internal_error")
    status = api.status(job_id)
    assert status["status"] == "failed"
    assert status["error"]["code"] == "llm_error"


# --- owner ----------------------------------------------------------------------------------


def test_job_of_another_owner_does_not_exist(api_factory):
    api = api_factory(api_keys="key-a,key-b")
    a = {"X-API-Key": "key-a"}
    b = {"X-API-Key": "key-b"}
    response = api.create({"external_ref": "gen-own"}, headers=a)
    job_id = response.json()["job_id"]
    api.real.store.update(job_id, lambda r: setattr(r, "status", "running"))

    for method, suffix in [("get", ""), ("get", "/result"), ("get", "/report"), ("delete", ""), ("post", "/retry")]:
        error = assert_error(api.client.request(method, f"/v1/jobs/{job_id}{suffix}", headers=b), 404, "job_not_found")
        assert error["details"] == {"job_id": job_id}
    listed = api.client.get("/v1/jobs", params={"external_ref": "gen-own"}, headers=b)
    assert listed.json() == {"jobs": []}

    # Untouched for the owner.
    assert api.client.get(f"/v1/jobs/{job_id}", headers=a).json()["status"] == "running"
    assert reserved(api, "gen-own", owner_of("key-a")) == job_id
    assert reserved(api, "gen-own", owner_of("key-b")) is None
