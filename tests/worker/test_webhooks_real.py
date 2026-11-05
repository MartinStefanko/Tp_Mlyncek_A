"""Webhooks of the real service: API + RQ worker → queue "qgen-webhooks" → webhook worker.

fakeredis + SimpleWorker; Laravel is httpx.MockTransport (tests/worker/conftest.py FakeLaravel).
"""

import hashlib
import hmac
import importlib.util
import logging
import threading
import time
from pathlib import Path

import httpx
import pytest
from rq.registry import ScheduledJobRegistry

from qgen.contract import WebhookPayload
from service.jobs.recovery import recover_jobs
from service.jobs.runner import STAGE_ORDER
from service.webhooks.rq_delivery import WEBHOOK_QUEUE_NAME
from tests.fakes import FakeLLM, client_error
from tests.worker.conftest import CALLBACK_URL, HEADERS, WEBHOOK_SECRET, scripted_answer

HOOK = {"external_ref": "gen-hook", "language": "sk", "callback_url": CALLBACK_URL}
STATE_KEYS = ("job_id", "external_ref", "status", "stage", "progress", "error")


def php_signature_ok(request: httpx.Request, secret: str, now: int) -> bool:
    """Independent copy of the PHP check of Príloha A (no service code):

    $calc = 'sha256=' . hash_hmac('sha256', $ts . '.' . $request->getContent(), secret);
    abort_unless(hash_equals($calc, $sig) && abs(time() - (int) $ts) < 300, 401);
    """
    ts = request.headers["X-QGen-Timestamp"]
    calc = "sha256=" + hmac.new(secret.encode(), ts.encode() + b"." + request.content, hashlib.sha256).hexdigest()
    return hmac.compare_digest(calc, request.headers["X-QGen-Signature"]) and abs(now - int(ts)) < 300


def assert_same_state(payload: dict, status: dict) -> None:
    assert {key: payload[key] for key in STATE_KEYS} == {key: status[key] for key in STATE_KEYS}


def run_and_deliver(env, options=HOOK) -> str:
    job_id = env.create(options)
    env.work()
    env.hooks.deliver()
    return job_id


# --- the four events ----------------------------------------------------------------------------


def test_completed_job_sends_each_stage_once_then_completed(env):
    job_id = run_and_deliver(env)  # progress_interval_s=0: every callback writes job.json
    laravel = env.hooks.laravel
    events = laravel.events()
    assert events[-1] == "job.completed"
    stages = [p["stage"] for p in laravel.payloads() if p["event"] == "job.stage_changed"]
    assert len(stages) == len(set(stages))  # at most once per stage
    assert stages == [s for s in STAGE_ORDER if s in stages]  # in order
    assert {"extract", "learning_objects", "questions", "q_evaluation", "output"} <= set(stages)
    assert events == ["job.stage_changed"] * len(stages) + ["job.completed"]

    for request in laravel.requests:
        assert str(request.url) == CALLBACK_URL
        assert request.headers["Content-Type"] == "application/json"
        assert php_signature_ok(request, WEBHOOK_SECRET, now=int(env.hooks.clock.now))
        payload = WebhookPayload.model_validate_json(request.content)  # extra="forbid": no result inside
        assert payload.job_id == job_id and payload.external_ref == "gen-hook"
    assert len({r.headers["X-QGen-Delivery"] for r in laravel.requests}) == len(laravel.requests)
    assert_same_state(laravel.payloads()[-1], env.status(job_id))
    assert laravel.payloads()[-1]["progress"] == 1.0


def test_failed_job_sends_failed_with_the_error(env):
    env.use(llm=FakeLLM(default=client_error(400)))
    job_id = run_and_deliver(env)
    last = env.hooks.laravel.payloads()[-1]
    assert (last["event"], last["status"]) == ("job.failed", "failed")
    assert last["error"]["code"] == "llm_error"
    assert_same_state(last, env.status(job_id))
    assert env.hooks.laravel.events().count("job.failed") == 1


def test_cancel_of_a_queued_job_sends_cancelled(env):
    job_id = env.create(HOOK)
    response = env.client.delete(f"/v1/jobs/{job_id}", headers=HEADERS)
    assert response.json()["status"] == "cancelled"
    env.work()
    env.hooks.deliver()
    assert env.hooks.laravel.events() == ["job.cancelled"]
    payload = env.hooks.laravel.payloads()[0]
    assert payload["status"] == "cancelled"
    assert_same_state(payload, env.status(job_id))


def test_cancel_during_the_run_sends_cancelled_once_from_the_api(env):
    state = {"job_id": None, "cancelled": False}

    def answer(prompt, model):
        if not state["cancelled"]:
            state["cancelled"] = True
            assert env.client.delete(f"/v1/jobs/{state['job_id']}", headers=HEADERS).json()["status"] == "cancelled"
        return scripted_answer(prompt, model)

    env.use(llm=FakeLLM(default=answer))
    state["job_id"] = job_id = env.create(HOOK)
    env.work()
    env.hooks.deliver()
    events = env.hooks.laravel.events()
    assert events.count("job.cancelled") == 1
    assert events[-1] == "job.cancelled"
    assert "job.completed" not in events and "job.failed" not in events
    assert_same_state(env.hooks.laravel.payloads()[-1], env.status(job_id))


def test_worker_lost_from_recovery_sends_failed(env):
    job_id = env.create(HOOK)
    env.redis.delete(env.queue.queue.key)  # Redis lost the queue, the worker died twice
    env.store.update(job_id, lambda r: setattr(r, "status", "running"))
    env.store.update(job_id, lambda r: setattr(r, "attempts", 2))
    result = recover_jobs(env.store, env.queue, events=env.context.events)
    assert result.failed == [job_id]
    env.hooks.deliver()
    payload = env.hooks.laravel.payloads()[-1]
    assert (payload["event"], payload["error"]["code"]) == ("job.failed", "worker_lost")
    assert_same_state(payload, env.status(job_id))


def test_without_callback_url_nothing_is_queued(env):
    job_id = env.create({"external_ref": "no-hook", "language": "sk"})
    env.work()
    env.client.delete(f"/v1/jobs/{job_id}", headers=HEADERS)
    assert env.hooks.queue.count == 0
    env.hooks.deliver()
    assert env.hooks.laravel.requests == []


# --- retries ------------------------------------------------------------------------------------


def _one_cancelled_delivery(env) -> str:
    """A queued job cancelled by DELETE: exactly one webhook (job.cancelled) in the queue."""
    job_id = env.create(HOOK)
    env.client.delete(f"/v1/jobs/{job_id}", headers=HEADERS)
    return job_id


def test_retry_has_new_timestamp_and_signature_same_delivery_and_body(env):
    hooks = env.hooks
    hooks.laravel.statuses = [500]
    _one_cancelled_delivery(env)
    hooks.deliver()
    assert len(hooks.laravel.requests) == 1
    hooks.clock.now += 10
    assert hooks.release_scheduled() == [10]
    hooks.deliver()

    first, second = hooks.laravel.requests
    assert first.headers["X-QGen-Delivery"] == second.headers["X-QGen-Delivery"]
    assert first.content == second.content  # same occurred_at
    assert int(second.headers["X-QGen-Timestamp"]) == int(first.headers["X-QGen-Timestamp"]) + 10
    assert first.headers["X-QGen-Signature"] != second.headers["X-QGen-Signature"]
    for request in (first, second):
        assert php_signature_ok(request, WEBHOOK_SECRET, now=int(hooks.clock.now))
    assert hooks.release_scheduled() == []  # delivered: no further attempt


def test_timeout_is_retried(env):
    hooks = env.hooks
    calls = []

    def laravel(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("no answer in 5 s", request=request)
        return httpx.Response(204)

    hooks.context.client = httpx.Client(transport=httpx.MockTransport(laravel))
    _one_cancelled_delivery(env)
    hooks.deliver()
    assert hooks.release_scheduled() == [10]
    hooks.deliver()
    assert len(calls) == 2


def test_gives_up_after_three_retries_and_the_job_does_not_change(env, caplog):
    hooks = env.hooks
    hooks.laravel.statuses = [500] * 10
    job_id = _one_cancelled_delivery(env)
    before = env.status(job_id)
    delays = []
    with caplog.at_level(logging.INFO, logger="qgen.webhooks"):
        hooks.deliver()
        for _ in range(5):
            released = hooks.release_scheduled()
            delays += released
            hooks.deliver()
    assert delays == [10, 60, 300]
    assert len(hooks.laravel.requests) == 4
    assert len(ScheduledJobRegistry(queue=hooks.queue)) == 0
    gave_up = [r.getMessage() for r in caplog.records if "gave up" in r.getMessage()]
    assert len(gave_up) == 1
    assert f"job_id={job_id}" in gave_up[0] and "event=job.cancelled" in gave_up[0] and "HTTP 500" in gave_up[0]
    assert env.status(job_id) == before


def test_host_not_allowed_at_send_time_is_not_sent(env, caplog):
    hooks = env.hooks
    _one_cancelled_delivery(env)  # allowed when the job was created
    hooks.context.settings = env.settings.model_copy(update={"webhook_allowed_hosts": "other-host"})
    with caplog.at_level(logging.WARNING, logger="qgen.webhooks"):
        hooks.deliver()
    assert hooks.laravel.requests == []
    assert hooks.release_scheduled() == []  # no retry
    assert any("not in QGEN_WEBHOOK_ALLOWED_HOSTS" in r.getMessage() for r in caplog.records)


# --- delivery does not wait for processing ------------------------------------------------------


def test_delivery_does_not_wait_for_the_running_job(env):
    """While the job worker is busy with the job, the webhook worker delivers its webhooks."""
    seen = {}
    real_pipeline = env.runner.pipeline

    def pipeline(*args, **kwargs):
        job_id = seen["job_id"]
        env.hooks.deliver()  # the webhook worker, while queue "qgen" is busy with this job
        seen["delivered"] = env.hooks.laravel.payloads()
        seen["status"] = env.store.get(job_id).status
        return real_pipeline(*args, **kwargs)

    env.use(pipeline=pipeline)
    seen["job_id"] = job_id = env.create(HOOK)
    env.work()
    assert seen["status"] == "running"
    assert [(p["event"], p["stage"]) for p in seen["delivered"]] == [("job.stage_changed", "extract")]
    assert env.status(job_id)["status"] == "completed"


def test_webhooks_never_go_to_the_job_queue(env):
    job_id = env.create(HOOK)
    env.client.delete(f"/v1/jobs/{job_id}", headers=HEADERS)
    assert env.queue.queue.name != WEBHOOK_QUEUE_NAME
    assert env.hooks.queue.name == WEBHOOK_QUEUE_NAME
    assert env.hooks.queue.count == 1
    assert all(job.func_name != "service.webhooks.rq_delivery.deliver" for job in env.queue.queue.jobs)


def test_final_webhook_is_queued_at_once_without_delay(env):
    """5 s metric: the first attempt is in the queue right after the write, not scheduled."""
    job_id = env.create(HOOK)
    env.client.delete(f"/v1/jobs/{job_id}", headers=HEADERS)
    assert env.hooks.queue.count == 1
    assert len(ScheduledJobRegistry(queue=env.hooks.queue)) == 0
    assert env.hooks.context.client.timeout.read == env.settings.webhook_timeout_s


# --- tools/webhook_receiver.py ------------------------------------------------------------------


def _load_receiver():
    path = Path(__file__).resolve().parents[2] / "tools" / "webhook_receiver.py"
    spec = importlib.util.spec_from_file_location("webhook_receiver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_webhook_receiver_verifies_the_signature(env):
    receiver = _load_receiver()
    lines: list[str] = []
    server = receiver.make_server(0, WEBHOOK_SECRET, fail=1, out=lines.append, host="127.0.0.1")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    hooks = env.hooks
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/webhook"
        hooks.context.client = httpx.Client(timeout=5)
        hooks.context.settings = env.settings.model_copy(update={"webhook_allowed_hosts": "127.0.0.1"})
        hooks.clock.now = time.time()
        _one_cancelled_delivery(env)
        job = hooks.queue.jobs[0]  # send the queued job.cancelled to the local receiver
        job.args = ({**job.args[0], "url": url}, 1)
        job.save()
        hooks.deliver()  # --fail 1: 500
        assert hooks.release_scheduled() == [10]
        hooks.deliver()  # 200
        bad = httpx.post(url, content=b"{}", headers={"X-QGen-Timestamp": str(int(time.time())), "X-QGen-Signature": "sha256=00"})
    finally:
        server.shutdown()
        server.server_close()
    assert bad.status_code == 401
    assert len(lines) == 3
    assert "job.cancelled" in lines[0] and "signature=OK -> 500" in lines[0]
    assert "signature=OK -> 200" in lines[1] and "status=cancelled" in lines[1]
    assert "signature=INVALID -> 401" in lines[2]
    assert WEBHOOK_SECRET not in " ".join(lines)
