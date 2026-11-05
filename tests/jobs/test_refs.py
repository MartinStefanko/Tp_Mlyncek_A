"""external_ref reservations, deletion with release, cleanup after TTL, Redis list queue."""

from datetime import timedelta

import fakeredis
import pytest

from service.jobs import JobNotFound
from service.jobs.cleanup import run_cleanup
from service.jobs.refs import ExternalRefs, delete_job_and_ref

from .conftest import drive, new_job


@pytest.fixture
def redis_client():
    return fakeredis.FakeRedis()


@pytest.fixture
def refs(redis_client) -> ExternalRefs:
    return ExternalRefs(redis_client)


def test_reserve_lookup_release(refs):
    assert refs.lookup("key-a", "gen-1") is None
    assert refs.reserve("key-a", "gen-1", "job1") is None
    assert refs.reserve("key-a", "gen-1", "job2") == "job1"
    assert refs.reserve("key-b", "gen-1", "job3") is None  # per owner
    assert refs.lookup("key-a", "gen-1") == "job1"

    assert refs.release("key-a", "gen-1", "job2") is False  # not the holder
    assert refs.lookup("key-a", "gen-1") == "job1"
    assert refs.release("key-a", "gen-1", "job1") is True
    assert refs.lookup("key-a", "gen-1") is None
    assert refs.lookup("key-b", "gen-1") == "job3"


def test_key_format(refs, redis_client):
    refs.reserve("key-a", "gen:with:colons", "job1")
    assert redis_client.get("qgen:extref:key-a:gen:with:colons") == b"job1"


def test_delete_job_and_ref(store, refs):
    record = new_job(store, external_ref="gen-1")
    refs.reserve(record.owner, "gen-1", record.job_id)
    delete_job_and_ref(store, refs, record)
    assert refs.lookup(record.owner, "gen-1") is None
    with pytest.raises(JobNotFound):
        store.get(record.job_id)


def test_delete_keeps_reservation_of_a_newer_job(store, refs):
    record = new_job(store, external_ref="gen-1")
    refs.force(record.owner, "gen-1", "newer")
    delete_job_and_ref(store, refs, record)
    assert refs.lookup(record.owner, "gen-1") == "newer"


def test_delete_with_redis_down_still_deletes_the_job(store, refs, monkeypatch):
    record = new_job(store, external_ref="gen-1")

    def down(*args):
        raise ConnectionError("redis down")

    monkeypatch.setattr(refs, "release", down)
    delete_job_and_ref(store, refs, record)
    with pytest.raises(JobNotFound):
        store.get(record.job_id)


def test_cleanup_after_ttl_releases_reservation(store, refs, clock):
    old = new_job(store, external_ref="gen-old")
    refs.reserve(old.owner, "gen-old", old.job_id)
    drive(store, old.job_id, "running", "completed")
    young = new_job(store, external_ref="gen-young")
    refs.reserve(young.owner, "gen-young", young.job_id)

    clock.advance(days=7, seconds=1)
    result = run_cleanup(store, clock(), result_ttl=timedelta(days=7), refs=refs)
    assert result.jobs_deleted == [old.job_id]
    assert refs.lookup(old.owner, "gen-old") is None
    assert refs.lookup(young.owner, "gen-young") == young.job_id

