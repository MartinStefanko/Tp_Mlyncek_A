"""Small local receiver of QGen webhooks: try webhooks without Laravel.

Checks the signature like Laravel (Príloha A – Podpis): HMAC-SHA256 of timestamp + "." + body
with QGEN_WEBHOOK_SECRET, message not older than 5 minutes. Prints one line per webhook and
answers 200 (valid signature) or 401 (invalid).

Usage (from the repo root, while docker-compose.dev.yml runs):
    venv/Scripts/python.exe tools/webhook_receiver.py [--port 9000] [--fail N] [--secret S]

    --fail N   answer 500 to the first N webhooks (to see the retries after 10, 60, 300 s)
    --secret   default: QGEN_WEBHOOK_SECRET from the environment, else from .env

In .env: QGEN_WEBHOOK_ALLOWED_HOSTS=host.docker.internal and in the job options
"callback_url": "http://host.docker.internal:9000/webhook".
The body has only the state of the job (no document text); the secret is never printed.
"""

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from service.webhooks.signing import (  # noqa: E402
    HEADER_DELIVERY,
    HEADER_EVENT,
    HEADER_SIGNATURE,
    HEADER_TIMESTAMP,
    verify,
)

MAX_AGE_S = 300


def load_secret(explicit: Optional[str] = None, env_file: Path = REPO_ROOT / ".env") -> str:
    if explicit is not None:
        return explicit
    if "QGEN_WEBHOOK_SECRET" in os.environ:
        return os.environ["QGEN_WEBHOOK_SECRET"]
    if env_file.is_file():
        from dotenv import dotenv_values

        return dotenv_values(env_file).get("QGEN_WEBHOOK_SECRET") or ""
    return ""


def describe(body: bytes) -> str:
    try:
        payload = json.loads(body)
    except ValueError:
        return "body is not JSON"
    error = payload.get("error") or {}
    parts = [
        f"job_id={payload.get('job_id')}",
        f"external_ref={payload.get('external_ref')}",
        f"status={payload.get('status')}",
        f"stage={payload.get('stage')}",
        f"progress={payload.get('progress')}",
        f"occurred_at={payload.get('occurred_at')}",
    ]
    if error:
        parts.append(f"error.code={error.get('code')}")
    return " ".join(parts)


def make_handler(
    secret: str,
    fail: int = 0,
    out: Callable[[str], None] = print,
    *,
    fail_when: Optional[Callable[[dict], bool]] = None,
    on_webhook: Optional[Callable[[dict], None]] = None,
) -> type[BaseHTTPRequestHandler]:
    """`fail`: answer 500 to the first N valid webhooks.
    `fail_when(payload)`: answer 500 to a valid webhook when it returns True (checked after
    `fail`; used by tools/resilience_check.py to fail one job only).
    `on_webhook(info)`: called for every webhook with event, delivery, timestamp, valid,
    code and the parsed body (payload, None when not JSON)."""
    state = {"fail": fail}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 – http.server API
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            valid = verify(
                secret,
                self.headers.get(HEADER_TIMESTAMP, ""),
                body,
                self.headers.get(HEADER_SIGNATURE, ""),
                now=int(time.time()),
                max_age_s=MAX_AGE_S,
            )
            try:
                payload = json.loads(body)
            except ValueError:
                payload = None
            with lock:
                failing = valid and state["fail"] > 0
                if failing:
                    state["fail"] -= 1
                elif valid and fail_when is not None and isinstance(payload, dict):
                    failing = bool(fail_when(payload))
            code = 401 if not valid else 500 if failing else 200
            if on_webhook is not None:
                on_webhook({
                    "event": self.headers.get(HEADER_EVENT),
                    "delivery": self.headers.get(HEADER_DELIVERY),
                    "timestamp": self.headers.get(HEADER_TIMESTAMP),
                    "valid": valid,
                    "code": code,
                    "payload": payload,
                    "received_at": time.time(),
                })
            out(
                f"{datetime.now():%H:%M:%S} {self.headers.get(HEADER_EVENT)} "
                f"delivery={self.headers.get(HEADER_DELIVERY)} ts={self.headers.get(HEADER_TIMESTAMP)} "
                f"signature={'OK' if valid else 'INVALID'} -> {code}\n    {describe(body)}"
            )
            self.send_response(code)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args) -> None:  # silence the default access log
            pass

    return Handler


def make_server(
    port: int,
    secret: str,
    fail: int = 0,
    out: Callable[[str], None] = print,
    host: str = "0.0.0.0",
    **handler_options,
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(secret, fail, out, **handler_options))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Receive and verify QGen webhooks.")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--fail", type=int, default=0, help="answer 500 to the first N webhooks")
    parser.add_argument("--secret", default=None, help="default: QGEN_WEBHOOK_SECRET (environment or .env)")
    args = parser.parse_args(argv)
    secret = load_secret(args.secret)
    if not secret:
        print("warning: QGEN_WEBHOOK_SECRET is empty", file=sys.stderr)
    server = make_server(args.port, secret, args.fail)
    print(f"listening on http://0.0.0.0:{args.port}/ (from containers: http://host.docker.internal:{args.port}/webhook)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
