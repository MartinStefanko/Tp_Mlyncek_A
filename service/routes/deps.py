"""Shared route dependencies: settings, backend, API key."""

import hashlib
import hmac
from typing import Annotated, Optional

from fastapi import Depends, Header, Request

from service.backend import JobBackend
from service.errors import ApiError
from service.settings import ServiceSettings

ANONYMOUS_OWNER = "anonymous"


def get_settings(request: Request) -> ServiceSettings:
    return request.app.state.settings


def get_backend(request: Request) -> JobBackend:
    return request.app.state.backend


def require_api_key(
    settings: Annotated[ServiceSettings, Depends(get_settings)],
    x_api_key: Annotated[Optional[str], Header(description="Required when QGEN_API_KEYS is set.")] = None,
) -> str:
    """Check X-API-Key; return the owner id of the key (never the key itself)."""
    keys = settings.api_key_list
    if not keys:
        return ANONYMOUS_OWNER
    if x_api_key:
        given = x_api_key.encode("utf-8")
        for key in keys:
            if hmac.compare_digest(given, key.encode("utf-8")):
                return "key-" + hashlib.sha256(given).hexdigest()[:16]
    raise ApiError(401, "unauthorized", "Missing or invalid X-API-Key.")


Settings = Annotated[ServiceSettings, Depends(get_settings)]
Backend = Annotated[JobBackend, Depends(get_backend)]
Owner = Annotated[str, Depends(require_api_key)]
