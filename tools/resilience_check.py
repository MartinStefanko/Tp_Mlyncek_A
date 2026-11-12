"""Resilience check of the real service in Docker (kap. 9: Test odolnosti, Test 202, Test mazania).

Starts docker-compose.yml + dev + replay overrides (LLM and embeddings from the
recording baseline_v6/, no API key, no cost), receives the webhooks in this process
(tools/webhook_receiver.py, signature checked) and goes through the scenarios over the
HTTP API and `docker compose`:

    a   restart of api and redis right after 202            → completed
    a2  Redis loses the queue (FLUSHALL, worker stopped)    → worker restores it from job.json → completed
    b   docker kill worker during running                   → after the worker start: completed, attempts 2
    c   kill twice during running                           → failed worker_lost + webhook job.failed
    d   DELETE during running                               → cancelled, job.cancelled exactly once,
                                                              PDF deleted, /result 409 job_not_completed
    e   LLM failure (QGEN_LLM_REPLAY_FAIL=1)                → failed llm_error, PDF kept; switch off,
                                                              POST /retry → completed with the same job_id
    f   POST with the same external_ref during the run and after it → 200 with the same job_id
    g   restart of webhook-worker during the retries (receiver answers 500) → webhook delivered

At the end it prints a table OK / CHYBA and runs `docker compose down` (volumes stay).
Manual only (not part of pytest); takes about 15 minutes.

    venv/Scripts/python.exe tools/resilience_check.py
    venv/Scripts/python.exe tools/resilience_check.py --only b,d --keep-up

Needs Docker, baseline_v6/llm_calls.jsonl and eko.pdf (tests/fixtures/ or
Hlavna_cast/ucebny_material/). Free ports: 8000 (API), 9000 (receiver).
Times that cannot be waited for here (24 h input TTL, 7 day result TTL) are covered by
tests/worker/test_retention.py and tests/jobs/test_cleanup.py with a moved clock.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from webhook_receiver import make_server  # noqa: E402

COMPOSE_FILES = ("docker-compose.yml", "docker-compose.dev.yml", "docker-compose.replay.yml")
RECORDING = REPO_ROOT / "baseline_v6" / "llm_calls.jsonl"
PDF_CANDIDATES = (
    REPO_ROOT / "tests" / "fixtures" / "eko.pdf",
    REPO_ROOT / "Hlavna_cast" / "ucebny_material" / "eko.pdf",
)
API_KEY = "replay-dev-key"  # docker-compose.replay.yml defaults
WEBHOOK_SECRET = "replay-dev-secret"
ALL_SCENARIOS = ("a", "a2", "b", "c", "d", "e", "f", "g")

# Timeouts (s). A replayed run of eko.pdf takes about 1-2 min with a delay of 1 s per call;
# a job of a killed worker comes back after its Redis key expires (job_monitoring_interval
# 30 s + 60 s) and the next maintenance (QGEN_MAINTENANCE_INTERVAL_S=15 in the replay compose).
T_START = 240
T_RUNNING = 120
T_RUN = 600
T_RECOVERY = 300
T_WEBHOOK = 90
T_WEBHOOK_RETRIES = 300


class CheckFailed(Exception):
    """A scenario did not end as expected."""


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def wait_for(description: str, probe: Callable[[], Any], timeout_s: float, interval_s: float = 1.0) -> Any:
    """Call `probe` until it returns a true value; errors of `probe` count as 'not yet'."""
    deadline = time.monotonic() + timeout_s
    last_error: Optional[str] = None
    while time.monotonic() < deadline:
        try:
            value = probe()
            if value:
                return value
        except (httpx.HTTPError, subprocess.SubprocessError, OSError, ValueError, KeyError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(interval_s)
    extra = f" (last error: {last_error})" if last_error else ""
    raise CheckFailed(f"timeout {timeout_s:g} s: {description}{extra}")


# --- docker compose ----------------------------------------------------------------------------


class Compose:
    def __init__(self, env: dict[str, str]) -> None:
        self.env = env

    def run(self, *args: str, env: Optional[dict[str, str]] = None, capture: bool = False, timeout: float = 900) -> str:
        command = ["docker", "compose"]
        for name in COMPOSE_FILES:
            command += ["-f", str(REPO_ROOT / name)]
        command += list(args)
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env={**os.environ, **self.env, **(env or {})},
            capture_output=capture,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if result.returncode != 0:
            detail = (result.stderr or "").strip()[-800:] if capture else ""
            raise subprocess.CalledProcessError(result.returncode, " ".join(args), stderr=detail)
        return result.stdout if capture else ""

    def action(self, *args: str, env: Optional[dict[str, str]] = None) -> None:
        log("docker compose " + " ".join(args) + (f"  {env}" if env else ""))
        self.run(*args, env=env, capture=True)

    def sh(self, service: str, command: str) -> str:
        return self.run("exec", "-T", service, "sh", "-c", command, capture=True, timeout=60)

    def job_json(self, job_id: str) -> dict:
        return json.loads(self.sh("api", f"cat /data/jobs/{job_id}/job.json"))

    def pdf_count(self, job_id: str) -> int:
        out = self.sh("api", f"ls /data/jobs/{job_id}/input/ 2>/dev/null | grep -c '\\.pdf$' || true")
        return int(out.strip() or 0)


def ensure_port_free(port: int) -> None:
    """HTTPServer binds with SO_REUSEADDR; on Windows it then shares the port with another
    receiver (e.g. tools/webhook_receiver.py), which gets the webhooks. Bind once without it."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("0.0.0.0", port))
    except OSError:
        raise SystemExit(
            f"Port {port} is in use (another webhook receiver?). Stop it or use --receiver-port."
        ) from None
    finally:
        probe.close()


# --- webhooks ------------------------------------------------------------------------------------


class Receiver:
    """Webhook receiver in this process. Records every webhook; fails chosen ones with 500."""

    def __init__(self, port: int, secret: str) -> None:
        ensure_port_free(port)
        self.port = port
        self.events: list[dict] = []
        self._lock = threading.Lock()
        self._fail: dict[tuple[str, str], int] = {}  # (external_ref, event) → failures left
        self.server = make_server(
            port, secret, out=lambda line: None, fail_when=self._fail_when, on_webhook=self._record
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def callback_url(self) -> str:
        return f"http://host.docker.internal:{self.port}/webhook"

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def fail_next(self, external_ref: str, event: str, count: int) -> None:
        with self._lock:
            self._fail[(external_ref, event)] = count

    def _fail_when(self, payload: dict) -> bool:
        key = (payload.get("external_ref"), payload.get("event"))
        with self._lock:
            left = self._fail.get(key, 0)
            if left > 0:
                self._fail[key] = left - 1
                return True
        return False

    def _record(self, info: dict) -> None:
        payload = info.get("payload") or {}
        with self._lock:
            self.events.append(info)
        log(f"  webhook {info['event']} job={str(payload.get('job_id'))[:8]} status={payload.get('status')} "
            f"signature={'OK' if info['valid'] else 'INVALID'} -> {info['code']}")

    def of(self, job_id: str, event: Optional[str] = None, code: Optional[int] = None) -> list[dict]:
        with self._lock:
            return [
                e for e in self.events
                if (e.get("payload") or {}).get("job_id") == job_id
                and (event is None or e["event"] == event)
                and (code is None or e["code"] == code)
            ]


# --- API -----------------------------------------------------------------------------------------


class Api:
    def __init__(self, base_url: str, api_key: str, pdf: Path, callback_url: str) -> None:
        self.http = httpx.Client(base_url=base_url, headers={"X-API-Key": api_key}, timeout=30)
        self.pdf = pdf
        self.callback_url = callback_url

    def post_job(self, external_ref: str) -> httpx.Response:
        options = {"external_ref": external_ref, "callback_url": self.callback_url}
        with open(self.pdf, "rb") as f:
            return self.http.post(
                "/v1/jobs",
                files=[("files", ("eko.pdf", f, "application/pdf"))],
                data={"options": json.dumps(options)},
            )

    def create(self, external_ref: str) -> str:
        response = self.post_job(external_ref)
        expect(response.status_code == 202, f"POST /v1/jobs: expected 202, got {response.status_code} {response.text[:300]}")
        job_id = response.json()["job_id"]
        log(f"  job {job_id[:8]} created (external_ref={external_ref})")
        return job_id

    def job(self, job_id: str) -> dict:
        response = self.http.get(f"/v1/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def wait_status(self, job_id: str, statuses: tuple[str, ...], timeout_s: float, min_progress: float = 0.0) -> dict:
        def probe():
            job = self.job(job_id)
            if job["status"] in statuses and job["progress"] >= min_progress:
                return job
            final = ("completed", "failed", "cancelled")
            if job["status"] in final and job["status"] not in statuses:
                raise CheckFailed(f"job {job_id[:8]} ended as {job['status']} ({job.get('error')}), expected {statuses}")
            return None

        job = wait_for(f"job {job_id[:8]} in {statuses}", probe, timeout_s)
        log(f"  job {job_id[:8]} {job['status']} stage={job.get('stage')} progress={job.get('progress')}")
        return job

    def health(self) -> dict:
        return self.http.get("/health").json()


# --- scenarios -----------------------------------------------------------------------------------


@dataclass
class Result:
    name: str
    title: str
    ok: bool = False
    detail: str = ""
    seconds: float = 0.0


@dataclass
class Check:
    compose: Compose
    api: Api
    receiver: Receiver
    run_id: str
    notes: list[str] = field(default_factory=list)

    def ref(self, scenario: str, suffix: str = "") -> str:
        return f"rc-{self.run_id}-{scenario}{suffix}"

    def note(self, text: str) -> None:
        self.notes.append(text)

    # --- helpers ---------------------------------------------------------------------------

    def wait_healthy(self, timeout_s: float = T_START) -> None:
        def probe():
            components = self.api.health().get("components", {})
            return all(components.get(name) == "ok" for name in ("redis", "worker", "webhook_worker"))

        wait_for("/health: redis, worker and webhook_worker ok", probe, timeout_s, interval_s=2)

    def kill_worker(self) -> None:
        self.compose.action("kill", "worker")

    def start_worker(self) -> None:
        self.compose.action("start", "worker")

    def wait_webhook(self, job_id: str, event: str, timeout_s: float = T_WEBHOOK) -> dict:
        return wait_for(
            f"webhook {event} delivered (200) for {job_id[:8]}",
            lambda: (self.receiver.of(job_id, event, 200) or [None])[0],
            timeout_s,
        )

    def attempts(self, job_id: str) -> int:
        return int(self.compose.job_json(job_id)["attempts"])

    # --- a, a2: Test 202 -----------------------------------------------------------------------

    def scenario_a(self) -> str:
        job_id = self.api.create(self.ref("a"))
        self.compose.action("restart", "api", "redis")
        self.wait_healthy()
        job = self.api.wait_status(job_id, ("completed",), T_RUN)
        result = self.api.http.get(f"/v1/jobs/{job_id}/result")
        expect(result.status_code == 200, f"/result after completed: {result.status_code}")
        self.wait_webhook(job_id, "job.completed")
        return f"completed, attempts={self.attempts(job_id)}, progress={job['progress']}"

    def scenario_a2(self) -> str:
        self.compose.action("stop", "worker")
        job_id = self.api.create(self.ref("a2"))
        self.compose.action("exec", "-T", "redis", "redis-cli", "FLUSHALL")
        queued = self.compose.run("exec", "-T", "redis", "redis-cli", "LLEN", "rq:queue:qgen", capture=True).strip()
        expect(queued == "0", f"queue not empty after FLUSHALL: {queued}")
        # the webhook worker's registration is gone too: start it again so /health sees it
        self.compose.action("restart", "webhook-worker")
        self.start_worker()
        self.api.wait_status(job_id, ("completed",), T_RUN)
        attempts = self.attempts(job_id)
        expect(attempts == 1, f"attempts {attempts}, expected 1 (a queued job never ran)")
        self.wait_webhook(job_id, "job.completed")
        return "Redis lost the queue; worker restored the job from job.json, completed, attempts=1"

    # --- b, c: worker killed during the run ---------------------------------------------------

    def scenario_b(self) -> str:
        job_id = self.api.create(self.ref("b"))
        self.api.wait_status(job_id, ("running",), T_RUNNING, min_progress=0.01)
        self.kill_worker()
        killed_at = time.monotonic()
        self.start_worker()
        self.api.wait_status(job_id, ("completed",), T_RECOVERY + T_RUN)
        attempts = self.attempts(job_id)
        expect(attempts == 2, f"attempts {attempts}, expected 2")
        self.wait_webhook(job_id, "job.completed")
        return f"completed after the kill in {time.monotonic() - killed_at:.0f} s, attempts=2"

    def scenario_c(self) -> str:
        job_id = self.api.create(self.ref("c"))
        self.api.wait_status(job_id, ("running",), T_RUNNING, min_progress=0.01)
        self.kill_worker()
        self.start_worker()

        def second_run():
            job = self.api.job(job_id)
            return job["status"] == "running" and job["progress"] > 0 and self.attempts(job_id) == 2

        wait_for(f"job {job_id[:8]} running again (attempts 2)", second_run, T_RECOVERY, interval_s=2)
        log(f"  job {job_id[:8]} running again, attempts=2")
        self.kill_worker()
        self.start_worker()
        job = self.api.wait_status(job_id, ("failed",), T_RECOVERY)
        code = (job.get("error") or {}).get("code")
        expect(code == "worker_lost", f"error.code {code}, expected worker_lost")
        hook = self.wait_webhook(job_id, "job.failed")
        hook_code = ((hook["payload"] or {}).get("error") or {}).get("code")
        expect(hook_code == "worker_lost", f"webhook job.failed error.code {hook_code}")
        expect(self.compose.pdf_count(job_id) > 0, "PDF of the failed job deleted (must stay 24 h)")
        return "failed worker_lost after 2 runs, webhook job.failed, PDF kept for retry"

    # --- d: DELETE during the run -------------------------------------------------------------

    def scenario_d(self) -> str:
        job_id = self.api.create(self.ref("d"))
        self.api.wait_status(job_id, ("running",), T_RUNNING, min_progress=0.01)
        response = self.api.http.delete(f"/v1/jobs/{job_id}")
        expect(response.status_code == 200, f"DELETE: {response.status_code} {response.text[:200]}")
        expect(response.json()["status"] == "cancelled", f"DELETE status {response.json()['status']}")
        wait_for(f"PDF of {job_id[:8]} deleted", lambda: self.compose.pdf_count(job_id) == 0, 60, interval_s=2)
        result = self.api.http.get(f"/v1/jobs/{job_id}/result")
        expect(result.status_code == 409, f"/result: {result.status_code}, expected 409")
        expect(result.json()["error"]["code"] == "job_not_completed", f"/result: {result.json()['error']['code']}")
        self.wait_webhook(job_id, "job.cancelled")
        log("  waiting 20 s for duplicate or late webhooks")
        time.sleep(20)
        cancelled = self.receiver.of(job_id, "job.cancelled")
        expect(len(cancelled) == 1, f"job.cancelled received {len(cancelled)}x, expected once")
        late = self.receiver.of(job_id, "job.completed") + self.receiver.of(job_id, "job.failed")
        expect(not late, f"unexpected webhooks after cancel: {[e['event'] for e in late]}")
        expect(self.api.job(job_id)["status"] == "cancelled", "status changed after cancel")
        return "cancelled, job.cancelled once, PDF deleted, /result 409 job_not_completed"

    # --- e: LLM failure and retry ---------------------------------------------------------------

    def scenario_e(self) -> str:
        self.compose.action("up", "-d", "worker", env={"QGEN_LLM_REPLAY_FAIL": "1"})
        try:
            self.wait_healthy()
            job_id = self.api.create(self.ref("e"))
            job = self.api.wait_status(job_id, ("failed",), T_RUN)
            code = (job.get("error") or {}).get("code")
            expect(code == "llm_error", f"error.code {code}, expected llm_error")
            expect(job.get("retryable_until") is not None, "retryable_until missing")
            self.wait_webhook(job_id, "job.failed")
            pdfs = self.compose.pdf_count(job_id)
            expect(pdfs > 0, "PDF of the failed job deleted")
        finally:
            self.compose.action("up", "-d", "worker", env={"QGEN_LLM_REPLAY_FAIL": "0"})
        self.wait_healthy()
        response = self.api.http.post(f"/v1/jobs/{job_id}/retry")
        expect(response.status_code == 202, f"retry: {response.status_code} {response.text[:200]}")
        expect(response.json()["job_id"] == job_id, "retry returned another job_id")
        self.api.wait_status(job_id, ("completed",), T_RUN)
        self.wait_webhook(job_id, "job.completed")
        return f"failed llm_error (stage {job['error']['details'].get('stage')}), PDF kept; retry → completed, same job_id"

    # --- f: idempotency ------------------------------------------------------------------------

    def scenario_f(self) -> str:
        ref = self.ref("f")
        job_id = self.api.create(ref)
        self.api.wait_status(job_id, ("running",), T_RUNNING)
        for moment in ("during the run", "after completed"):
            if moment == "after completed":
                self.api.wait_status(job_id, ("completed",), T_RUN)
            response = self.api.post_job(ref)
            expect(response.status_code == 200, f"POST {moment}: {response.status_code}, expected 200")
            expect(response.json()["job_id"] == job_id, f"POST {moment}: another job_id")
            log(f"  POST {moment}: 200, same job_id, status {response.json()['status']}")
        return "repeated POST during the run and after completed: 200, same job_id"

    # --- g: webhook worker restart during retries ------------------------------------------------

    def scenario_g(self) -> str:
        ref = self.ref("g")
        self.receiver.fail_next(ref, "job.cancelled", 2)
        job_id = self.api.create(ref)
        response = self.api.http.delete(f"/v1/jobs/{job_id}")
        expect(response.status_code == 200, f"DELETE: {response.status_code}")
        wait_for("first attempt answered 500", lambda: self.receiver.of(job_id, "job.cancelled", 500), T_WEBHOOK)
        self.compose.action("restart", "webhook-worker")
        hook = self.wait_webhook(job_id, "job.cancelled", T_WEBHOOK_RETRIES)
        attempts = self.receiver.of(job_id, "job.cancelled")
        expect(len(attempts) == 3, f"{len(attempts)} attempts, expected 3 (500, 500, 200)")
        expect(len({a["delivery"] for a in attempts}) == 1, "X-QGen-Delivery changed between attempts")
        expect(all(a["valid"] for a in attempts), "invalid signature in an attempt")
        expect(len({a["timestamp"] for a in attempts}) == 3, "timestamp not new on every attempt")
        expect(len(self.receiver.of(job_id, "job.cancelled", 200)) == 1, "delivered more than once")
        delay = hook["received_at"] - attempts[0]["received_at"]
        return f"delivered on attempt 3 after {delay:.0f} s (500, restart, 500, 200), same delivery id"


TITLES = {
    "a": "reštart api a redis hneď po 202",
    "a2": "Redis stratí frontu → obnova z job.json",
    "b": "kill worker počas running",
    "c": "dvakrát kill → worker_lost",
    "d": "DELETE počas behu",
    "e": "zlyhanie LLM → failed, retry",
    "f": "opakovaný POST s external_ref",
    "g": "reštart webhook-workera počas opakovaní",
}


def print_table(results: list[Result]) -> None:
    print()
    print(f"{'scenár':<7}{'výsledok':<10}{'čas':>6}  popis / detail")
    print("-" * 100)
    for r in results:
        print(f"{r.name:<7}{'OK' if r.ok else 'CHYBA':<10}{r.seconds:>5.0f}s  {r.title}")
        print(f"{'':<25}{r.detail}")
    print("-" * 100)
    failed = [r.name for r in results if not r.ok]
    print("Všetky scenáre OK." if not failed else f"CHYBA v scenároch: {', '.join(failed)}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Resilience check of the service in Docker (replay LLM).")
    parser.add_argument("--only", default=",".join(ALL_SCENARIOS), help="comma-separated scenarios (a,a2,b,c,d,e,f,g)")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--receiver-port", type=int, default=9000)
    parser.add_argument("--delay", type=float, default=1.0, help="QGEN_LLM_REPLAY_DELAY_S (s per replayed call)")
    parser.add_argument("--no-build", action="store_true", help="do not rebuild the image")
    parser.add_argument("--keep-up", action="store_true", help="do not run docker compose down at the end")
    args = parser.parse_args(argv)

    chosen = [s.strip() for s in args.only.split(",") if s.strip()]
    unknown = [s for s in chosen if s not in ALL_SCENARIOS]
    if unknown:
        parser.error(f"unknown scenarios: {', '.join(unknown)}")
    if not RECORDING.is_file():
        print(f"Missing recording {RECORDING} (tools/baseline.py --out baseline_v6 --force).", file=sys.stderr)
        return 2
    pdf = next((p for p in PDF_CANDIDATES if p.is_file()), None)
    if pdf is None:
        print("Missing eko.pdf (tests/fixtures/ or Hlavna_cast/ucebny_material/).", file=sys.stderr)
        return 2

    env = {
        "QGEN_LLM_REPLAY_DELAY_S": f"{args.delay:g}",
        "QGEN_LLM_REPLAY_FAIL": "0",
        "QGEN_MAINTENANCE_INTERVAL_S": "15",
        "QGEN_REPLAY_API_KEY": API_KEY,
        "QGEN_REPLAY_WEBHOOK_SECRET": WEBHOOK_SECRET,
    }
    compose = Compose(env)
    receiver = Receiver(args.receiver_port, WEBHOOK_SECRET)
    api = Api(args.api, API_KEY, pdf, receiver.callback_url)
    check = Check(compose, api, receiver, run_id=uuid.uuid4().hex[:6])
    results: list[Result] = []

    receiver.start()
    log(f"receiver on port {args.receiver_port}, run id {check.run_id}, scenarios {', '.join(chosen)}")
    try:
        compose.action("up", "-d", *([] if args.no_build else ["--build"]))
        check.wait_healthy()
        info = api.http.get("/v1/info").json()["models"]
        if not info.get("replay"):
            print(f"/v1/info does not show replay: {info}", file=sys.stderr)
            return 2
        for name in chosen:
            result = Result(name, TITLES[name])
            started = time.monotonic()
            log(f"=== {name}: {TITLES[name]}")
            try:
                check.wait_healthy()
                result.detail = getattr(check, f"scenario_{name}")()
                result.ok = True
            except CheckFailed as exc:
                result.detail = str(exc)
            except Exception as exc:  # noqa: BLE001 – report and go on with the next scenario
                result.detail = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            result.seconds = time.monotonic() - started
            log(f"=== {name}: {'OK' if result.ok else 'CHYBA'} – {result.detail}")
            results.append(result)
    finally:
        receiver.stop()
        if not args.keep_up:
            try:
                compose.action("down")
            except subprocess.CalledProcessError as exc:
                log(f"docker compose down failed: {exc}")
    print_table(results)
    return 0 if results and all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
