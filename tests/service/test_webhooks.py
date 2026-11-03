"""Webhooks: signature (independent implementation of the Laravel check), retries, events."""

import asyncio
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone

import httpx
from fastapi.testclient import TestClient

from qgen.contract import WebhookPayload
from service.main import create_app
from service.mock import MockBackend
from service.webhooks import RETRY_DELAYS_S, build_headers, new_delivery, next_retry_delay, send_once, sign, verify
from service.webhooks.asyncio_dispatcher import AsyncioWebhookDispatcher

from .conftest import API_KEY, CALLBACK_URL, DURATION_S, PDF, make_settings

SECRET = "test-secret"

PAYLOAD = WebhookPayload(
    event="job.completed",
    job_id="7f3c2a1e",
    external_ref="gen-01JB7Q3K",
    status="completed",
    stage="output",
    progress=1.0,
    error=None,
    occurred_at=datetime(2026, 11, 3, 10, 21, 45, tzinfo=timezone.utc),
)


def laravel_check(headers: httpx.Headers, body: bytes, secret: str, now: int) -> bool:
    """Python copy of the PHP sketch in Príloha A, written without service code:

    $calc = 'sha256=' . hash_hmac('sha256', $ts . '.' . $request->getContent(), secret);
    abort_unless(hash_equals($calc, $sig) && abs(time() - (int) $ts) < 300, 401);
    """
    ts = headers["X-QGen-Timestamp"]
    sig = headers["X-QGen-Signature"]
    calc = "sha256=" + hmac.new(secret.encode(), ts.encode() + b"." + body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(calc, sig) and abs(now - int(ts)) < 300


# --- signing ----------------------------------------------------------------------------------


def test_signature_matches_laravel_formula():
    body = PAYLOAD.model_dump_json().encode()
    headers = httpx.Headers(build_headers(new_delivery(CALLBACK_URL, PAYLOAD), SECRET, 1793701125))
    assert laravel_check(headers, body, SECRET, now=1793701125 + 10)
    assert not laravel_check(headers, body, "other-secret", now=1793701125)
    assert not laravel_check(headers, body + b" ", SECRET, now=1793701125)
    assert not laravel_check(headers, body, SECRET, now=1793701125 + 300)  # older than 5 minutes


def test_sign_known_vector():
    # Computed independently: echo -n '1793701125.{"a":1}' | openssl dgst -sha256 -hmac 'k'
    expected = "sha256=" + hmac.new(b"k", b'1793701125.{"a":1}', hashlib.sha256).hexdigest()
    assert sign("k", 1793701125, b'{"a":1}') == expected
    assert verify("k", "1793701125", b'{"a":1}', expected, now=1793701125)
    assert not verify("k", "abc", b'{"a":1}', expected, now=1793701125)


def test_headers_and_body():
    delivery = new_delivery(CALLBACK_URL, PAYLOAD)
    headers = build_headers(delivery, SECRET, 1793701125)
    assert headers["Content-Type"] == "application/json"
    assert headers["X-QGen-Event"] == "job.completed"
    assert headers["X-QGen-Delivery"] == delivery.delivery_id
    assert headers["X-QGen-Timestamp"] == "1793701125"
    assert headers["X-QGen-Signature"].startswith("sha256=")
    assert json.loads(delivery.body) == {
        "event": "job.completed",
        "job_id": "7f3c2a1e",
        "external_ref": "gen-01JB7Q3K",
        "status": "completed",
        "stage": "output",
        "progress": 1.0,
        "error": None,
        "occurred_at": "2026-11-03T10:21:45Z",
    }
    assert new_delivery(CALLBACK_URL, PAYLOAD).delivery_id != delivery.delivery_id


def test_retry_plan():
    assert RETRY_DELAYS_S == (10, 60, 300)
    assert [next_retry_delay(n) for n in range(0, 6)] == [None, 10, 60, 300, None, None]


# --- retries ----------------------------------------------------------------------------------


class Receiver:
    """Fake Laravel endpoint: answers with the given status codes in order."""

    def __init__(self, statuses, clock):
        self.statuses = list(statuses)
        self.clock = clock
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status = self.statuses.pop(0) if self.statuses else 200
        if status == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        return httpx.Response(status)


class UnixClock:
    def __init__(self, start: float = 1793701125.0):
        self.now = start

    def __call__(self) -> float:
        return self.now


def run_delivery(statuses):
    unix = UnixClock()
    receiver = Receiver(statuses, unix)
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        unix.now += seconds

    async def main():
        dispatcher = AsyncioWebhookDispatcher(
            SECRET, 5, transport=httpx.MockTransport(receiver), sleep=fake_sleep, unix_time=unix
        )
        await dispatcher.start()
        try:
            return await dispatcher.deliver(new_delivery(CALLBACK_URL, PAYLOAD))
        finally:
            await dispatcher.stop()

    delivered = asyncio.run(main())
    return delivered, receiver.requests, sleeps


def test_retry_until_success_with_new_timestamp_and_valid_signature():
    delivered, requests, sleeps = run_delivery([500, "timeout", 200])
    assert delivered is True
    assert sleeps == [10, 60]
    assert len(requests) == 3

    timestamps = [int(r.headers["X-QGen-Timestamp"]) for r in requests]
    assert timestamps == [1793701125, 1793701135, 1793701195]
    assert len({r.headers["X-QGen-Signature"] for r in requests}) == 3
    assert len({r.headers["X-QGen-Delivery"] for r in requests}) == 1
    assert len({r.content for r in requests}) == 1
    for request, ts in zip(requests, timestamps):
        assert laravel_check(request.headers, request.content, SECRET, now=ts)
        assert json.loads(request.content)["occurred_at"] == "2026-11-03T10:21:45Z"


def test_gives_up_after_three_retries():
    delivered, requests, sleeps = run_delivery([500, 502, 404, 503, 200])
    assert delivered is False
    assert len(requests) == 4
    assert sleeps == [10, 60, 300]


def test_2xx_is_success():
    delivered, requests, sleeps = run_delivery([204])
    assert delivered is True
    assert (len(requests), sleeps) == (1, [])


def test_send_once_sync_for_worker():
    received: list[httpx.Request] = []

    def handler(request):
        received.append(request)
        return httpx.Response(200)

    with httpx.Client(transport=httpx.MockTransport(handler), timeout=5) as client:
        assert send_once(client, new_delivery(CALLBACK_URL, PAYLOAD), SECRET, 1793701125)
    assert laravel_check(received[0].headers, received[0].content, SECRET, now=1793701125)


def test_timeout_is_set_on_client():
    dispatcher = AsyncioWebhookDispatcher(SECRET, 2.5)

    async def main():
        await dispatcher.start()
        timeout = dispatcher._client.timeout
        await dispatcher.stop()
        return timeout

    assert asyncio.run(main()) == httpx.Timeout(2.5)


# --- events from the mock ---------------------------------------------------------------------


def test_events_without_callback_url(api):
    api.create_id({"external_ref": "no-callback"})
    api.run_to_end()
    assert api.notifier.sent == []


def test_stage_changed_at_most_once_per_stage(api):
    job_id = api.create_id({"external_ref": "gen-hook", "callback_url": CALLBACK_URL})
    for _ in range(200):
        api.clock.advance(DURATION_S / 150)
        api.backend.tick()
        api.status(job_id)
    events = [p for _, p in api.notifier.sent]
    stages = [p.stage for p in events if p.event == "job.stage_changed"]
    assert stages == ["extract", "language", "learning_objects", "lo_evaluation", "questions", "q_evaluation", "split", "output"]
    assert [p.event for p in events][-1] == "job.completed"
    assert all(url == CALLBACK_URL for url, _ in api.notifier.sent)
    occurred = [p.occurred_at for p in events]
    assert occurred == sorted(occurred)
    assert all(p.status == "running" for p in events[:-1])
    assert [p.progress for p in events] == sorted(p.progress for p in events)


def test_last_webhook_equals_job_state(api):
    job_id = api.create_id({"external_ref": "gen-last", "callback_url": CALLBACK_URL})
    api.run_to_end()
    last = api.notifier.sent[-1][1]
    status = api.status(job_id)
    for key in ("job_id", "external_ref", "status", "stage", "progress", "error"):
        assert getattr(last, key) == status[key]


def test_webhooks_end_to_end_through_http():
    """Real dispatcher on the app loop, Laravel replaced by httpx.MockTransport."""
    received: list[httpx.Request] = []

    def laravel(request: httpx.Request) -> httpx.Response:
        received.append(request)
        # First delivery of job.completed fails once, so a retry happens.
        if request.headers["X-QGen-Event"] == "job.completed" and sum(
            r.headers["X-QGen-Event"] == "job.completed" for r in received
        ) == 1:
            return httpx.Response(500)
        return httpx.Response(200)

    settings = make_settings(mock_duration_s=0.4)
    dispatcher = AsyncioWebhookDispatcher(SECRET, 5, delays=(0.05, 0.05, 0.05), transport=httpx.MockTransport(laravel))
    backend = MockBackend(settings, dispatcher, tick_interval_s=0.02)
    with TestClient(create_app(settings, backend)) as client:
        response = client.post(
            "/v1/jobs",
            headers={"X-API-Key": API_KEY},
            files=[("files", ("eko.pdf", PDF, "application/pdf"))],
            data={"options": json.dumps({"external_ref": "gen-e2e", "callback_url": CALLBACK_URL})},
        )
        job_id = response.json()["job_id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            completed = [r for r in received if r.headers["X-QGen-Event"] == "job.completed"]
            if len(completed) >= 2:
                break
            time.sleep(0.02)

    events = [r.headers["X-QGen-Event"] for r in received]
    assert events.count("job.stage_changed") == 8
    completed = [r for r in received if r.headers["X-QGen-Event"] == "job.completed"]
    assert len(completed) == 2
    assert completed[0].headers["X-QGen-Delivery"] == completed[1].headers["X-QGen-Delivery"]
    assert completed[0].content == completed[1].content
    now = int(time.time())
    for request in received:
        assert str(request.url) == CALLBACK_URL
        assert laravel_check(request.headers, request.content, SECRET, now=now)
        payload = WebhookPayload.model_validate_json(request.content)
        assert payload.job_id == job_id
    assert WebhookPayload.model_validate_json(completed[-1].content).status == "completed"


def test_failed_event_has_error(api):
    api.create_id({"external_ref": "mock-fail-hook", "callback_url": CALLBACK_URL})
    api.run_to_end()
    last = api.notifier.sent[-1][1]
    assert (last.event, last.status, last.stage) == ("job.failed", "failed", "questions")
    assert last.error.code == "llm_error"
