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
    raise ValueError(f"Unknown backend: {settings.backend}")


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
    return app


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
