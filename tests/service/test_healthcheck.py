"""Container healthcheck (service/healthcheck.py, step 17)."""

import fakeredis
from rq import Queue, Worker

from service import healthcheck


def _worker(connection, queue_name: str, hostname: str) -> Worker:
    worker = Worker([Queue(queue_name, connection=connection)], connection=connection)
    worker.hostname = hostname
    worker.register_birth()
    return worker


def test_worker_of_this_container_is_healthy():
    connection = fakeredis.FakeRedis()
    _worker(connection, "qgen", "container-a")
    assert healthcheck.check_worker(connection, "qgen", hostname="container-a")[0]


def test_worker_of_another_container_or_queue_does_not_count():
    connection = fakeredis.FakeRedis()
    _worker(connection, "qgen", "container-b")
    _worker(connection, "qgen-webhooks", "container-a")
    assert not healthcheck.check_worker(connection, "qgen", hostname="container-a")[0]


def test_dead_worker_is_not_healthy():
    connection = fakeredis.FakeRedis()
    _worker(connection, "qgen", "container-a").register_death()
    assert not healthcheck.check_worker(connection, "qgen", hostname="container-a")[0]


def test_api_unreachable_is_not_healthy():
    ok, message = healthcheck.check_api("http://127.0.0.1:9/health", timeout_s=1)
    assert not ok and "unreachable" in message


def test_unknown_role_is_usage_error():
    assert healthcheck.main(["scheduler"]) == 2
