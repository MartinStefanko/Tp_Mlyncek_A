"""FastAPI application.

    uvicorn service.main:app --host 0.0.0.0 --port 8000

OpenAPI documentation: /docs.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI

from service import SERVICE_VERSION
from service.backend import JobBackend
from service.errors import install_error_handlers
from service.routes import health, jobs
from service.settings import ServiceSettings
from service.webhooks.asyncio_dispatcher import AsyncioWebhookDispatcher

logger = logging.getLogger("qgen.api")


def build_backend(settings: ServiceSettings) -> JobBackend:
    if settings.backend == "mock":
        from service.mock import MockBackend

        if settings.allowed_webhook_hosts and not settings.webhook_secret.get_secret_value():
            logger.warning("QGEN_WEBHOOK_SECRET is empty: webhooks are signed with an empty key")
        dispatcher = AsyncioWebhookDispatcher(
            secret=settings.webhook_secret.get_secret_value(),
            timeout_s=settings.webhook_timeout_s,
        )
        return MockBackend(settings, dispatcher)
    if settings.backend == "real":
        return build_real_backend(settings)
    raise ValueError(f"Unknown backend: {settings.backend}")


def build_real_backend(settings: ServiceSettings) -> JobBackend:
    """job.json in QGEN_DATA_DIR, external_ref reservations and the RQ queue in Redis (QGEN_REDIS_URL)."""
    import redis

    from service.jobs import JobStore
    from service.jobs.refs import ExternalRefs
    from service.jobs.rq_queue import RqJobQueue
    from service.real_backend import RealBackend

    client = redis.Redis.from_url(settings.redis_url)
    return RealBackend(
        settings,
        JobStore.from_settings(settings),
        RqJobQueue(client, job_timeout_s=settings.job_timeout_s),
        ExternalRefs(client),
        on_stop=client.close,
    )


def create_app(settings: Optional[ServiceSettings] = None, backend: Optional[JobBackend] = None) -> FastAPI:
    settings = settings or ServiceSettings()
    backend = backend or build_backend(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await backend.start()
        try:
            yield
        finally:
            await backend.stop()

    mock_note = " (MOCK: simulated jobs, fixed result)" if settings.backend == "mock" else ""
    app = FastAPI(
        title="QGen API" + mock_note,
        version=SERVICE_VERSION,
        description="Question generation service for ECEH. Specification: Príloha A.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.backend = backend
    install_error_handlers(app)
    app.include_router(health.router)
    app.include_router(jobs.router)
    install_file_upload_schema(app)
    return app


def install_file_upload_schema(app: FastAPI) -> None:
    """Show upload fields as file pickers in /docs.

    FastAPI (OpenAPI 3.1) describes the optional file lists as
    anyOf[array of {type: string, contentMediaType: application/octet-stream} (or an anyOf
    with plain string), null]. Swagger UI renders that as text inputs. The fields `files` and `files[]` become
    {type: array, items: {type: string, format: binary, example: ""}}; the endpoint does not change.
    Without the empty example Swagger UI keeps its sample value "string" for a file field
    left untouched and sends it as text (422); with it, it sends an empty part, which the
    route ignores (service.routes.jobs._uploads).
    """
    build = app.openapi

    def openapi() -> dict:
        if app.openapi_schema is None:
            schema = build()
            for component in schema.get("components", {}).get("schemas", {}).values():
                for name, prop in component.get("properties", {}).items():
                    if _is_file_list(prop):
                        component["properties"][name] = {
                            key: value for key, value in prop.items() if key in ("title", "description")
                        } | {"type": "array", "items": {"type": "string", "format": "binary", "example": ""}}
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]


def _is_file_list(prop: dict) -> bool:
    """An array whose items are (or may be) a file. The route types the items as
    UploadFile | str (empty text parts from Swagger UI), so items may be an anyOf."""
    return any(
        variant.get("type") == "array" and any(_is_file(item) for item in variant.get("items", {}).get("anyOf", [variant.get("items", {})]))
        for variant in prop.get("anyOf", [prop])
    )


def _is_file(item: dict) -> bool:
    return item.get("type") == "string" and (
        item.get("contentMediaType") == "application/octet-stream" or item.get("format") == "binary"
    )


_default_app: Optional[FastAPI] = None


def __getattr__(name: str) -> FastAPI:
    """`service.main:app` is built on first access, so importing this module in tests has no side effects."""
    global _default_app
    if name != "app":
        raise AttributeError(name)
    if _default_app is None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
        _default_app = create_app()
    return _default_app
