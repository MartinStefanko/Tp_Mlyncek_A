"""GET /health, GET /v1/info, GET /v1/schema/lesson, GET /v1/schema/report."""

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from qgen.contract import ErrorResponse
from qgen.contract.schemas import build_schema
from service import SERVICE_VERSION
from service.routes.deps import Backend, Owner, Settings

router = APIRouter()

_AUTH_ERRORS: dict[int | str, dict[str, Any]] = {401: {"model": ErrorResponse, "description": "Missing or invalid X-API-Key."}}


@router.get("/health", tags=["service"], summary="Service state (API, Redis, worker, webhook worker). No API key.")
def health(backend: Backend) -> JSONResponse:
    components = backend.health()
    healthy = all(state in ("ok", "mock") for state in components.values())
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "ok" if healthy else "error", "components": components},
    )


@router.get("/v1/info", tags=["service"], summary="Service version, models and limits.", responses=_AUTH_ERRORS)
def info(settings: Settings, _owner: Owner) -> dict[str, Any]:
    return {
        "service": "qgen",
        "version": SERVICE_VERSION,
        "backend": settings.backend,
        "models": {
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "embed_provider": settings.embed_provider,
            "embed_model": settings.embed_model,
        },
        "limits": {
            "max_files": settings.max_files,
            "max_upload_mb": settings.max_upload_mb,
            "max_pages": settings.max_pages,
            "max_active_jobs": settings.max_active_jobs,
            "result_ttl_days": settings.result_ttl_days,
            "failed_input_ttl_hours": settings.failed_input_ttl_hours,
        },
        "option_defaults": asdict(settings.option_defaults()),
        "webhooks_enabled": bool(settings.allowed_webhook_hosts),
    }


@router.get("/v1/schema/lesson", tags=["schema"], summary="JSON Schema of the units format.", responses=_AUTH_ERRORS)
def lesson_schema(_owner: Owner) -> dict[str, Any]:
    return build_schema("lesson")


@router.get("/v1/schema/report", tags=["schema"], summary="JSON Schema of the report.", responses=_AUTH_ERRORS)
def report_schema(_owner: Owner) -> dict[str, Any]:
    return build_schema("report")
