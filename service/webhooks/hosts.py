"""Allowed webhook hosts (QGEN_WEBHOOK_ALLOWED_HOSTS): "host" or "host:port", lower case.

The API checks callback_url when a job is created (400 invalid_options); the webhook worker
checks it again before every attempt, because the configuration may have changed since.
"""

from typing import Union

from pydantic import AnyHttpUrl, TypeAdapter, ValidationError

_URL = TypeAdapter(AnyHttpUrl)


def callback_host_allowed(url: Union[str, AnyHttpUrl], allowed: set[str]) -> bool:
    if isinstance(url, str):
        try:
            url = _URL.validate_python(url)
        except ValidationError:
            return False
    host = (url.host or "").lower()
    return host in allowed or f"{host}:{url.port}" in allowed
