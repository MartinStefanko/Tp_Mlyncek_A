"""Container healthcheck (Dockerfile HEALTHCHECK, docker-compose.yml).

    python -m service.healthcheck api|worker|webhook-worker

api             GET http://localhost:8000/health answers 200 (Redis available).
worker          Redis answers and a live RQ worker of THIS container (hostname) serves the
webhook-worker  job queue "qgen" / the webhook queue "qgen-webhooks".

Exit code 0 = healthy, 1 = not healthy. Prints one short line (docker inspect shows it);
no keys, no URLs with credentials.
"""

from __future__ import annotations

import socket
import sys
import urllib.error
import urllib.request
from typing import Optional

API_URL = "http://localhost:8000/health"
ROLES = ("api", "worker", "webhook-worker")


def check_api(url: str = API_URL, timeout_s: float = 4) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return response.status == 200, f"/health {response.status}"
    except urllib.error.HTTPError as exc:
        return False, f"/health {exc.code}"
    except OSError as exc:
        return False, f"/health unreachable ({type(exc).__name__})"


def check_worker(connection, queue_name: str, hostname: Optional[str] = None) -> tuple[bool, str]:
    """A live worker of `queue_name` runs on this host (container)."""
    from rq import Queue, Worker

    from service.jobs.rq_queue import worker_alive

    hostname = hostname or socket.gethostname()
    connection.ping()
    queue = Queue(queue_name, connection=connection)
    for worker in Worker.all(queue=queue):
        if worker.hostname == hostname and worker_alive(connection, worker.name):
            return True, f"worker of {queue_name} alive"
    return False, f"no live worker of {queue_name} in this container"


def run(role: str) -> tuple[bool, str]:
    if role == "api":
        return check_api()
    import redis

    from service.jobs.rq_queue import QUEUE_NAME
    from service.settings import ServiceSettings
    from service.webhooks.rq_delivery import WEBHOOK_QUEUE_NAME

    queue_name = QUEUE_NAME if role == "worker" else WEBHOOK_QUEUE_NAME
    connection = redis.Redis.from_url(ServiceSettings().redis_url, socket_timeout=5, socket_connect_timeout=5)
    try:
        return check_worker(connection, queue_name)
    except Exception as exc:  # noqa: BLE001
        return False, f"redis error ({type(exc).__name__})"
    finally:
        connection.close()


def main(argv: Optional[list[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or args[0] not in ROLES:
        print("usage: python -m service.healthcheck " + "|".join(ROLES))
        return 2
    ok, message = run(args[0])
    print(("OK " if ok else "FAIL ") + message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
