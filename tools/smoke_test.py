"""Smoke test of the final Docker setup (step 17, success metrics of the specification).

Builds the image, starts the four containers (docker-compose.yml + docker-compose.dev.yml
+ docker-compose.replay.yml, project "qgen-smoke", port 18000), and checks:

    config      `docker compose config -q` of every compose combination
    build       docker compose build (time, image size on disk and compressed)
    health      GET /health 200 within 60 s after `up` (success metric), then status "ok"
    non-root    the containers run as a user without root rights
    ocr         image with OCR: EasyOCR models are in the image (Reader without download);
                image without OCR: POST with extraction.ocr=true → 400 invalid_options
    job         POST eko.pdf → completed (one call starts every step)
    result      /result passes the Pydantic model and schemas/lesson.schema.json (no extra
                fields), /report passes its model and schema
    latency     GET of the status and of the result answer within 200 ms
    down        docker compose down -v (the smoke project's own volumes)

Default: LLM and embeddings from the recording baseline_v5/ (replay, no key, no cost).
--real: the real providers from .env instead (costs a few cents, eko.pdf takes ~5 min).

    venv/Scripts/python.exe tools/smoke_test.py                       # image with OCR
    venv/Scripts/python.exe tools/smoke_test.py --install-ocr false   # smaller image
    venv/Scripts/python.exe tools/smoke_test.py --real                # real Gemini (.env)

Needs Docker, .env (may be a copy of .env.example), eko.pdf (tests/fixtures/ or
Hlavna_cast/ucebny_material/) and, without --real, baseline_v5/llm_calls.jsonl.
Manual only (not part of pytest). Exit code 0 = every check OK, 1 = a check failed,
2 = missing prerequisites.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from qgen.output.check import check_report, check_units  # noqa: E402

PROJECT = "qgen-smoke"
IMAGE = "qgen:latest"
BASE_FILES = ("docker-compose.yml", "docker-compose.dev.yml")
REPLAY_FILE = "docker-compose.replay.yml"
CONFIG_COMBINATIONS = (
    ("docker-compose.yml",),
    ("docker-compose.yml", "docker-compose.remote.yml"),
    ("docker-compose.yml", "docker-compose.dev.yml"),
    ("docker-compose.yml", "docker-compose.dev.yml", REPLAY_FILE),
    ("docker-compose.mock.yml",),
)
RECORDING = REPO_ROOT / "baseline_v5" / "llm_calls.jsonl"
PDF_CANDIDATES = (
    REPO_ROOT / "tests" / "fixtures" / "eko.pdf",
    REPO_ROOT / "Hlavna_cast" / "ucebny_material" / "eko.pdf",
)
LESSON_SCHEMA = REPO_ROOT / "schemas" / "lesson.schema.json"
REPLAY_API_KEY = "replay-dev-key"  # docker-compose.replay.yml defaults
HEALTH_LIMIT_S = 60
LATENCY_LIMIT_MS = 200
T_HEALTHY = 180
T_JOB_REPLAY = 900
T_JOB_REAL = 1800


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


class CheckFailed(Exception):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def wait_for(description: str, probe: Callable[[], Any], timeout_s: float, interval_s: float = 0.5) -> Any:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            value = probe()
            if value:
                return value
        except (httpx.HTTPError, OSError, ValueError, KeyError) as exc:
            last_error = f" (last error: {type(exc).__name__})"
        time.sleep(interval_s)
    raise CheckFailed(f"timeout {timeout_s:g} s: {description}{last_error}")


class Compose:
    def __init__(self, files: tuple[str, ...], env: dict[str, str]) -> None:
        self.files = files
        self.env = env

    def command(self, *args: str, files: Optional[tuple[str, ...]] = None, project: bool = True) -> list[str]:
        command = ["docker", "compose"] + (["-p", PROJECT] if project else [])
        for name in files or self.files:
            command += ["-f", str(REPO_ROOT / name)]
        return command + list(args)

    def run(self, *args: str, files: Optional[tuple[str, ...]] = None, capture: bool = True,
            timeout: float = 3600, check: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run(
            self.command(*args, files=files), cwd=REPO_ROOT, env={**os.environ, **self.env},
            capture_output=capture, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
        if check and result.returncode != 0:
            detail = (result.stderr or "").strip()[-1500:] if capture else ""
            raise CheckFailed(f"docker compose {' '.join(args)} failed (exit {result.returncode}) {detail}")
        return result

    def exec(self, service: str, *command: str) -> str:
        return self.run("exec", "-T", service, *command, timeout=300).stdout.strip()


def image_size(image: str) -> str:
    """Size on disk (unpacked, `docker image ls`) and the compressed content size
    (`docker image inspect`; with the containerd image store this is the download size)."""
    disk = subprocess.run(
        ["docker", "image", "ls", image, "--format", "{{.Size}}"], capture_output=True, text=True, check=True
    ).stdout.strip().splitlines()[0]
    packed = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Size}}"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return f"{disk} on disk, {int(packed) / 1_000_000:.0f} MB compressed"


class Smoke:
    def __init__(self, args: argparse.Namespace, compose: Compose, pdf: Path, api_key: str) -> None:
        self.args = args
        self.compose = compose
        self.pdf = pdf
        self.base = f"http://localhost:{args.port}"
        self.http = httpx.Client(base_url=self.base, headers={"X-API-Key": api_key}, timeout=30)
        self.results: list[tuple[str, bool, str]] = []
        self.facts: dict[str, str] = {}
        self.job_id: Optional[str] = None

    def step(self, name: str, action: Callable[[], str]) -> bool:
        log(f"--- {name}")
        try:
            detail = action() or ""
            self.results.append((name, True, detail))
            log(f"OK    {name} {detail}")
            return True
        except CheckFailed as exc:
            self.results.append((name, False, str(exc)))
            log(f"CHYBA {name}: {exc}")
            return False

    # --- steps ------------------------------------------------------------------------------

    def config(self) -> str:
        for files in CONFIG_COMBINATIONS:
            self.compose.run("config", "-q", files=files)
        return f"{len(CONFIG_COMBINATIONS)} combinations"

    def build(self) -> str:
        started = time.monotonic()
        self.compose.run("build", capture=False)
        size = image_size(IMAGE)
        self.facts["image"] = size
        return f"{time.monotonic() - started:.0f} s, image {IMAGE} {size} (INSTALL_OCR={self.args.install_ocr})"

    def up_and_health(self) -> str:
        started = time.monotonic()  # from `up` (containers created, Redis healthcheck waited for)
        self.compose.run("up", "-d", "--no-build", "--force-recreate")

        def health_200() -> bool:
            return self.http.get("/health", timeout=2).status_code == 200

        wait_for("/health 200", health_200, HEALTH_LIMIT_S + 30)
        health_s = time.monotonic() - started
        self.facts["health_s"] = f"{health_s:.1f}"
        expect(health_s <= HEALTH_LIMIT_S, f"/health 200 after {health_s:.1f} s (limit {HEALTH_LIMIT_S} s)")

        def all_ok() -> bool:
            return self.http.get("/health", timeout=2).json().get("status") == "ok"

        wait_for('/health status "ok" (worker and webhook worker live)', all_ok, T_HEALTHY)
        components = self.http.get("/health").json()["components"]
        return f"200 after {health_s:.1f} s, components {components}"

    def container_health(self) -> str:
        def healthy() -> bool:
            out = self.compose.run("ps", "--format", "json").stdout
            rows = [json.loads(line) for line in out.splitlines() if line.strip().startswith("{")]
            if not rows and out.strip().startswith("["):
                rows = json.loads(out)
            states = {row["Service"]: row.get("Health", "") for row in rows}
            return len(states) == 4 and all(state == "healthy" for state in states.values())

        try:
            wait_for("docker healthchecks of all 4 containers healthy", healthy, 150, interval_s=3)
        except CheckFailed:
            raise CheckFailed("not all containers healthy: " + self.compose.run("ps").stdout.strip()) from None
        return "api, worker, webhook-worker, redis healthy"

    def non_root(self) -> str:
        users = {service: self.compose.exec(service, "id", "-u") for service in ("api", "worker", "webhook-worker")}
        expect(all(uid != "0" for uid in users.values()), f"root in a container: {users}")
        return f"uid {users}"

    def ocr(self) -> str:
        if self.args.install_ocr == "true":
            code = (
                "import easyocr; easyocr.Reader(['sk', 'en'], gpu=False, verbose=False, download_enabled=False);"
                "print('models ok')"
            )
            out = self.compose.exec("worker", "python", "-c", code)
            expect("models ok" in out, f"EasyOCR models missing in the image: {out[-300:]}")
            return "EasyOCR models in the image (no download)"
        response = self.http.post(
            "/v1/jobs",
            files=[("files", (self.pdf.name, self.pdf.read_bytes(), "application/pdf"))],
            data={"options": json.dumps({"external_ref": f"smoke-ocr-{uuid.uuid4().hex[:6]}", "extraction": {"ocr": True}})},
        )
        expect(response.status_code == 400, f"ocr=true without OCR: HTTP {response.status_code}")
        body = response.json()
        expect(body["error"]["code"] == "invalid_options", f"error {body['error']['code']}")
        types = [e.get("type") for e in body["error"].get("details", {}).get("errors", [])]
        expect("ocr_unavailable" in types, f"error details {types}")
        return "400 invalid_options (ocr_unavailable)"

    def job(self) -> str:
        options = {"external_ref": f"smoke-{uuid.uuid4().hex[:8]}", "split": {"mode": "topics"}}
        started = time.monotonic()
        response = self.http.post(
            "/v1/jobs",
            files=[("files", (self.pdf.name, self.pdf.read_bytes(), "application/pdf"))],
            data={"options": json.dumps(options)},
        )
        expect(response.status_code == 202, f"POST /v1/jobs: HTTP {response.status_code} {response.text[:300]}")
        self.job_id = response.json()["job_id"]
        last = {"stage": None}

        def done() -> bool:
            status = self.http.get(f"/v1/jobs/{self.job_id}").json()
            if status.get("stage") != last["stage"]:
                last["stage"] = status.get("stage")
                log(f"      {status['status']} stage={status.get('stage')} progress={status.get('progress')}")
            if status["status"] in ("failed", "cancelled"):
                raise CheckFailed(f"job {status['status']}: {status.get('error')}")
            return status["status"] == "completed"

        wait_for("job completed", done, T_JOB_REAL if self.args.real else T_JOB_REPLAY, interval_s=2)
        return f"job {self.job_id} completed in {time.monotonic() - started:.0f} s"

    def result(self) -> str:
        expect(self.job_id is not None, "no completed job")
        units = self.http.get(f"/v1/jobs/{self.job_id}/result").json()
        report = self.http.get(f"/v1/jobs/{self.job_id}/report").json()
        problems = check_units(units) + check_report(report)
        import jsonschema

        schema = json.loads(LESSON_SCHEMA.read_text(encoding="utf-8"))
        problems += [f"lesson.schema.json: {e.message[:120]}" for e in jsonschema.Draft202012Validator(schema).iter_errors(units)]
        expect(not problems, "; ".join(problems[:5]))
        lessons = units.get("units", [])
        questions = sum(len(unit.get("questions", [])) for unit in lessons)
        expect(questions > 0, "no questions in the result")
        warnings = [w.get("code") for w in report.get("warnings", [])]
        return f"{len(lessons)} lessons, {questions} questions; warnings {sorted(set(warnings))}"

    def latency(self) -> str:
        expect(self.job_id is not None, "no completed job")
        worst: dict[str, float] = {}
        for path in (f"/v1/jobs/{self.job_id}", f"/v1/jobs/{self.job_id}/result", "/health"):
            times = []
            for _ in range(10):
                started = time.perf_counter()
                expect(self.http.get(path).status_code == 200, f"GET {path}")
                times.append((time.perf_counter() - started) * 1000)
            worst[path.replace(self.job_id, "{id}")] = max(times)
        self.facts["latency_ms"] = f"{max(worst.values()):.0f}"
        expect(all(ms < LATENCY_LIMIT_MS for ms in worst.values()), f"slower than {LATENCY_LIMIT_MS} ms: {worst}")
        return ", ".join(f"{path} max {ms:.0f} ms" for path, ms in worst.items())

    def down(self) -> str:
        self.compose.run("down", "-v", "--remove-orphans")
        return "containers and smoke volumes removed"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--install-ocr", choices=("true", "false"), default="true", help="Build arg INSTALL_OCR.")
    parser.add_argument("--real", action="store_true", help="Real LLM / embeddings from .env (costs money).")
    parser.add_argument("--pdf", type=Path, help="PDF of the job (default eko.pdf; replay works only with eko.pdf).")
    parser.add_argument("--port", type=int, default=18000, help="Published API port of the smoke project.")
    parser.add_argument("--keep-up", action="store_true", help="Do not run docker compose down at the end.")
    parser.add_argument("--no-build", action="store_true", help="Use the existing image qgen:latest.")
    args = parser.parse_args(argv)

    if not (REPO_ROOT / ".env").is_file():
        print("Missing .env (cp .env.example .env).", file=sys.stderr)
        return 2
    pdf = args.pdf or next((p for p in PDF_CANDIDATES if p.is_file()), None)
    if pdf is None or not pdf.is_file():
        print("Missing eko.pdf (tests/fixtures/ or Hlavna_cast/ucebny_material/) or --pdf.", file=sys.stderr)
        return 2
    if not args.real and not RECORDING.is_file():
        print(f"Missing recording {RECORDING}.", file=sys.stderr)
        return 2
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        print("Docker is not running.", file=sys.stderr)
        return 2

    env = {
        "QGEN_INSTALL_OCR": args.install_ocr,
        "QGEN_API_PORT": str(args.port),
        "QGEN_DEV_NETWORK": f"{PROJECT}-eceh",
        "QGEN_LLM_REPLAY_DELAY_S": "0",
        "QGEN_LLM_REPLAY_FAIL": "0",
        "QGEN_REPLAY_API_KEY": REPLAY_API_KEY,
    }
    files = BASE_FILES if args.real else BASE_FILES + (REPLAY_FILE,)
    api_key = REPLAY_API_KEY
    if args.real:
        from dotenv import dotenv_values

        keys = (dotenv_values(REPO_ROOT / ".env").get("QGEN_API_KEYS") or "").split(",")
        api_key = keys[0].strip()
    compose = Compose(files, env)
    smoke = Smoke(args, compose, pdf, api_key)
    log(f"smoke test: files {', '.join(files)}, INSTALL_OCR={args.install_ocr}, {'REAL LLM' if args.real else 'replay'}")

    try:
        if smoke.step("config", smoke.config) and (args.no_build or smoke.step("build", smoke.build)):
            if args.no_build:
                smoke.facts["image"] = image_size(IMAGE)
            if smoke.step("health", smoke.up_and_health):
                smoke.step("container healthchecks", smoke.container_health)
                smoke.step("non-root", smoke.non_root)
                smoke.step("ocr", smoke.ocr)
                if smoke.step("job", smoke.job):
                    smoke.step("result", smoke.result)
                    smoke.step("latency", smoke.latency)
    finally:
        if not args.keep_up:
            smoke.step("down", smoke.down)

    print()
    print(f"{'check':<24} {'result':<6} detail")
    for name, ok, detail in smoke.results:
        print(f"{name:<24} {'OK' if ok else 'CHYBA':<6} {detail}")
    print(f"facts: {smoke.facts}")
    return 0 if all(ok for _, ok, _ in smoke.results) else 1


if __name__ == "__main__":
    sys.exit(main())
