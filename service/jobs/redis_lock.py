"""A simple Redis lock across workers: SET NX EX with a token, release by compare-and-delete.

Used so that recovery and the periodic cleanup never run twice at the same time when
more workers run. A lock that is held is not waited for: the caller skips its run
(`acquired` is False). The expiry frees the lock when its holder dies.
"""

import uuid
from typing import TYPE_CHECKING, Any, Optional

import redis as redis_lib

if TYPE_CHECKING:
    import redis


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


class RedisLock:
    def __init__(self, client: "redis.Redis", key: str, *, ttl_s: int) -> None:
        self._redis = client
        self.key = key
        self.ttl_s = ttl_s
        self.token = uuid.uuid4().hex
        self.acquired = False

    def acquire(self) -> bool:
        self.acquired = bool(self._redis.set(self.key, self.token, nx=True, ex=self.ttl_s))
        return self.acquired

    def release(self) -> bool:
        """Delete the key only if it still holds our token."""
        if not self.acquired:
            return False
        self.acquired = False
        with self._redis.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(self.key)
                    if _text(pipe.get(self.key)) != self.token:
                        pipe.unwatch()
                        return False
                    pipe.multi()
                    pipe.delete(self.key)
                    pipe.execute()
                    return True
                except redis_lib.WatchError:
                    continue

    def __enter__(self) -> "RedisLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
