"""external_ref reservations in Redis (kap. 3 – Opakovaná požiadavka).

external_ref is unique per API key while the job exists. POST /v1/jobs reserves it with
SET NX before any PDF is written, so two concurrent POSTs create one job. The key
`qgen:extref:{owner}:{external_ref}` holds the job_id.

The reservation must be released whenever the job is deleted (DELETE of a finished job,
cleanup after QGEN_RESULT_TTL_DAYS), else a new POST with the same external_ref would
return the deleted job. `delete_job_and_ref` is the one place that does both.

job.json stays the source of truth: a reservation without a job (crash between SET NX and
the write) is detected and replaced by RealBackend; a job whose reservation Redis lost is
found in the store and the reservation is written again.
"""

import logging
from typing import Any, Optional

import redis

from service.jobs.models import JobRecord
from service.jobs.store import JobStore

logger = logging.getLogger("qgen.jobs.refs")

KEY_PREFIX = "qgen:extref:"


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


class ExternalRefs:
    def __init__(self, client: "redis.Redis") -> None:
        self._redis = client

    @staticmethod
    def key(owner: str, external_ref: str) -> str:
        return f"{KEY_PREFIX}{owner}:{external_ref}"

    def lookup(self, owner: str, external_ref: str) -> Optional[str]:
        """job_id reserved for (owner, external_ref), or None."""
        return _text(self._redis.get(self.key(owner, external_ref)))

    def reserve(self, owner: str, external_ref: str, job_id: str) -> Optional[str]:
        """SET NX. None = reserved for `job_id`; otherwise the job_id that holds it."""
        key = self.key(owner, external_ref)
        if self._redis.set(key, job_id, nx=True):
            return None
        holder = _text(self._redis.get(key))
        if holder is None:  # released meanwhile: try once more
            return None if self._redis.set(key, job_id, nx=True) else _text(self._redis.get(key))
        return holder

    def force(self, owner: str, external_ref: str, job_id: str) -> None:
        """Point the reservation to `job_id` (repair after Redis lost the key)."""
        self._redis.set(self.key(owner, external_ref), job_id)

    def release(self, owner: str, external_ref: str, job_id: str) -> bool:
        """Delete the reservation only if it still belongs to `job_id` (compare-and-delete)."""
        key = self.key(owner, external_ref)
        with self._redis.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(key)
                    if _text(pipe.get(key)) != job_id:
                        pipe.unwatch()
                        return False
                    pipe.multi()
                    pipe.delete(key)
                    pipe.execute()
                    return True
                except redis.WatchError:
                    continue

    def ping(self) -> bool:
        return bool(self._redis.ping())


def release_quietly(refs: Optional[ExternalRefs], record: JobRecord) -> None:
    """Release the job's reservation; a Redis error is only logged (a stale reservation heals itself)."""
    if refs is None or not record.external_ref:
        return
    try:
        refs.release(record.owner, record.external_ref, record.job_id)
    except Exception as exc:  # noqa: BLE001 – Redis down must not break the deletion
        logger.error("external_ref release failed job_id=%s error=%s", record.job_id, type(exc).__name__)


def delete_job_and_ref(store: JobStore, refs: Optional[ExternalRefs], record: JobRecord) -> None:
    """Delete the job directory, then its external_ref reservation. JobNotFound if already gone."""
    store.delete_job(record.job_id)
    release_quietly(refs, record)
