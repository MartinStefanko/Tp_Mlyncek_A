"""Idempotency, DELETE, mock-fail- and retry."""

from datetime import datetime, timedelta

from qgen.contract import JobStatus

from .conftest import CALLBACK_URL, DURATION_S, PDF, assert_error

# --- idempotency ------------------------------------------------------------------------------


def test_same_external_ref_returns_original_job(api):
    first = api.create({"external_ref": "gen-01JB7Q3K"})
    assert first.status_code == 202
    job_id = first.json()["job_id"]
    api.clock.advance(DURATION_S / 2)

    # Different options and a different PDF are ignored.
    second = api.create(
        {"external_ref": "gen-01JB7Q3K", "split": {"mode": "lessons", "lesson_count": 3}},
        files=[("files", ("other.pdf", PDF + b"other", "application/pdf"))],
    )
    assert second.status_code == 200
    assert second.json() == {
        "job_id": job_id,
        "status": "running",
        "replayed": True,
        "links": {"self": f"/v1/jobs/{job_id}"},
    }
    assert len(api.backend._jobs) == 1
    assert api.backend._jobs[job_id].options.split.mode == "single"


def test_replay_of_failed_job(api):
    job_id = api.create_id({"external_ref": "mock-fail-idem"})
    api.run_to_end()
    assert api.status(job_id)["status"] == "failed"

    replay = api.create({"external_ref": "mock-fail-idem"})
    assert replay.status_code == 200
    assert replay.json()["job_id"] == job_id
    assert replay.json()["status"] == "failed"
    assert replay.json()["replayed"] is True


def test_replay_does_not_count_against_active_limit(api_factory):
    api = api_factory(max_active_jobs=1)
    job_id = api.create_id({"external_ref": "gen-limit"})
    replay = api.create({"external_ref": "gen-limit"})
    assert replay.status_code == 200
    assert replay.json()["job_id"] == job_id


def test_new_job_after_delete_of_finished_job(api):
    first = api.create_id({"external_ref": "gen-del"})
    api.run_to_end()
    assert api.client.delete(f"/v1/jobs/{first}", headers=api.headers).status_code == 204

    again = api.create({"external_ref": "gen-del"})
    assert again.status_code == 202
    assert again.json()["job_id"] != first
    assert again.json()["replayed"] is False


def test_cancelled_job_keeps_external_ref(api):
    job_id = api.create_id({"external_ref": "gen-cancel"})
    api.client.delete(f"/v1/jobs/{job_id}", headers=api.headers)
    replay = api.create({"external_ref": "gen-cancel"})
    assert replay.status_code == 200
    assert replay.json()["job_id"] == job_id
    assert replay.json()["status"] == "cancelled"


def test_external_ref_is_unique_per_api_key(api_factory):
    api = api_factory(api_keys="key-a,key-b")
    a = api.create({"external_ref": "gen-x"}, headers={"X-API-Key": "key-a"})
    b = api.create({"external_ref": "gen-x"}, headers={"X-API-Key": "key-b"})
    assert (a.status_code, b.status_code) == (202, 202)
    assert a.json()["job_id"] != b.json()["job_id"]
    # A job of another key behaves as if it did not exist.
    other = api.client.get(f"/v1/jobs/{a.json()['job_id']}", headers={"X-API-Key": "key-b"})
    assert_error(other, 404, "job_not_found")


# --- DELETE -----------------------------------------------------------------------------------


def test_delete_running_job_cancels_it(api):
    job_id = api.create_id({"external_ref": "gen-run", "callback_url": CALLBACK_URL})
    api.clock.advance(DURATION_S / 2)
    assert api.status(job_id)["status"] == "running"

    response = api.client.delete(f"/v1/jobs/{job_id}", headers=api.headers)
    assert response.status_code == 200
    status = JobStatus.model_validate(response.json())
    assert status.status == "cancelled"

    # Record stays and does not move on.
    api.clock.advance(DURATION_S * 2)
    assert api.status(job_id)["status"] == "cancelled"
    assert api.notifier.events(job_id)[-1] == "job.cancelled"
    assert api.notifier.events(job_id).count("job.cancelled") == 1
    assert "job.completed" not in api.notifier.events(job_id)

    assert_error(api.client.get(f"/v1/jobs/{job_id}/result", headers=api.headers), 409, "job_not_completed")
    assert_error(api.client.get(f"/v1/jobs/{job_id}/report", headers=api.headers), 409, "job_not_completed")


def test_delete_queued_job_cancels_it(api):
    job_id = api.create_id()
    response = api.client.delete(f"/v1/jobs/{job_id}", headers=api.headers)
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert response.json()["stage"] is None


def test_delete_finished_job_removes_everything(api):
    job_id = api.create_id({"external_ref": "gen-done"})
    api.run_to_end()
    assert api.status(job_id)["status"] == "completed"

    response = api.client.delete(f"/v1/jobs/{job_id}", headers=api.headers)
    assert response.status_code == 204
    assert response.content == b""

    for path in ("", "/result", "/report"):
        assert_error(api.client.get(f"/v1/jobs/{job_id}{path}", headers=api.headers), 404, "job_not_found")
    assert_error(api.client.delete(f"/v1/jobs/{job_id}", headers=api.headers), 404, "job_not_found")
    listed = api.client.get("/v1/jobs", params={"external_ref": "gen-done"}, headers=api.headers)
    assert listed.json() == {"jobs": []}


def test_delete_cancelled_job_removes_it(api):
    job_id = api.create_id()
    assert api.client.delete(f"/v1/jobs/{job_id}", headers=api.headers).status_code == 200
    assert api.client.delete(f"/v1/jobs/{job_id}", headers=api.headers).status_code == 204
    assert_error(api.client.get(f"/v1/jobs/{job_id}", headers=api.headers), 404, "job_not_found")


# --- mock-fail- and retry -------------------------------------------------------------------


def test_mock_fail_fails_in_questions_with_llm_error(api):
    job_id = api.create_id({"external_ref": "mock-fail-1", "callback_url": CALLBACK_URL})
    api.run_to_end()
    status = api.status(job_id)
    assert status["status"] == "failed"
    assert status["stage"] == "questions"
    assert 0 < status["progress"] < 1
    assert status["error"]["code"] == "llm_error"

    failed_at = datetime.fromisoformat(status["updated_at"])
    assert datetime.fromisoformat(status["retryable_until"]) - failed_at == timedelta(hours=24)

    events = api.notifier.events(job_id)
    assert events[-1] == "job.failed"
    assert "job.completed" not in events
    stages = [p.stage for _, p in api.notifier.sent if p.event == "job.stage_changed"]
    assert stages[-1] == "questions"
    failed_payload = api.notifier.sent[-1][1]
    assert failed_payload.error.code == "llm_error"

    assert_error(api.client.get(f"/v1/jobs/{job_id}/result", headers=api.headers), 409, "job_not_completed")
    assert_error(api.client.get(f"/v1/jobs/{job_id}/report", headers=api.headers), 409, "job_not_completed")


def test_retry_runs_failed_job_again_and_completes(api):
    job_id = api.create_id({"external_ref": "mock-fail-2", "callback_url": CALLBACK_URL})
    api.run_to_end()
    api.clock.advance(3600)

    response = api.client.post(f"/v1/jobs/{job_id}/retry", headers=api.headers)
    assert response.status_code == 202
    body = JobStatus.model_validate(response.json())
    assert body.job_id == job_id
    assert (body.status, body.stage, body.progress, body.error, body.retryable_until) == ("queued", None, 0.0, None, None)

    api.notifier.sent.clear()
    api.run_to_end()
    assert api.status(job_id)["status"] == "completed"
    assert api.client.get(f"/v1/jobs/{job_id}/result", headers=api.headers).status_code == 200
    # Stage webhooks are sent again for the new attempt, each at most once.
    assert api.notifier.events(job_id) == ["job.stage_changed"] * 8 + ["job.completed"]


def test_retry_after_24_hours_is_rejected(api):
    job_id = api.create_id({"external_ref": "mock-fail-3"})
    api.run_to_end()
    api.clock.advance(24 * 3600 + 1)
    assert_error(api.client.post(f"/v1/jobs/{job_id}/retry", headers=api.headers), 409, "job_not_retryable")


def test_retry_of_not_failed_job_is_rejected(api):
    job_id = api.create_id()
    assert_error(api.client.post(f"/v1/jobs/{job_id}/retry", headers=api.headers), 409, "job_not_retryable")
    api.run_to_end()
    assert_error(api.client.post(f"/v1/jobs/{job_id}/retry", headers=api.headers), 409, "job_not_retryable")


def test_retry_respects_active_limit(api_factory):
    api = api_factory(max_active_jobs=1)
    failed = api.create_id({"external_ref": "mock-fail-4"})
    api.run_to_end()
    api.create_id({"external_ref": "gen-busy"})
    error = assert_error(api.client.post(f"/v1/jobs/{failed}/retry", headers=api.headers), 429, "too_many_jobs")
    assert error["details"] == {"max_active_jobs": 1}
    assert api.status(failed)["status"] == "failed"


def test_api_key_is_required_for_job_routes(api):
    job_id = api.create_id()
    for method, path in [
        ("get", f"/v1/jobs/{job_id}"),
        ("get", f"/v1/jobs/{job_id}/result"),
        ("delete", f"/v1/jobs/{job_id}"),
        ("post", f"/v1/jobs/{job_id}/retry"),
    ]:
        assert_error(api.client.request(method, path), 401, "unauthorized")
    assert api.status(job_id)["status"] == "queued"
