"""Start-up security check of the service (step 17).

Called only by the entry points (`service.main:app`, `service.worker`,
`service.webhook_worker`), never by `create_app()` in tests.

Problems:
    QGEN_API_KEYS is empty                         the API accepts requests without a key
    webhooks on (QGEN_WEBHOOK_ALLOWED_HOSTS set)   Laravel cannot trust the signature
      and QGEN_WEBHOOK_SECRET is empty

    QGEN_LLM_PROVIDER or QGEN_EMBED_PROVIDER     answers from a recording, not from a model
      is replay (production only)

QGEN_ENV=production (default): any problem stops the start (StartupConfigError; the worker
exits with code 1, uvicorn fails to load the app). QGEN_ENV=development (docker-compose.dev.yml,
replay, mock): the key / secret problems are only a warning in the log, and replay is allowed
(qgen.llm.factory.check_config still logs its loud REPLAY warning).
"""

import logging

from qgen.config import Settings

logger = logging.getLogger("qgen.startup")


class StartupConfigError(RuntimeError):
    """The service must not start with this configuration (QGEN_ENV=production)."""


def security_problems(settings: Settings) -> list[str]:
    problems = []
    if not settings.api_key_list:
        problems.append("QGEN_API_KEYS is empty: the API accepts requests without X-API-Key")
    if settings.allowed_webhook_hosts and not settings.webhook_secret.get_secret_value():
        problems.append(
            "QGEN_WEBHOOK_SECRET is empty while webhooks are on (QGEN_WEBHOOK_ALLOWED_HOSTS): "
            "webhooks would be signed with an empty key"
        )
    return problems


def replay_problems(settings: Settings) -> list[str]:
    """Replay providers; a problem only with QGEN_ENV=production."""
    used = [
        name for name, provider in (("QGEN_LLM_PROVIDER", settings.llm_provider), ("QGEN_EMBED_PROVIDER", settings.embed_provider))
        if provider == "replay"
    ]
    if not used:
        return []
    return [" and ".join(used) + "=replay answers from a recording (development and tests only)"]


def check_service_config(settings: Settings, component: str) -> None:
    """Refuse (production) or warn (development) about every security problem."""
    problems = security_problems(settings)
    if settings.env == "production":
        problems += replay_problems(settings)
    if not problems:
        return
    if settings.env == "production":
        message = (
            f"{component} not started (QGEN_ENV=production): " + "; ".join(problems)
            + ". Set the variables in .env, or QGEN_ENV=development for local use only."
        )
        logger.error(message)
        raise StartupConfigError(message)
    for problem in problems:
        logger.warning("%s (QGEN_ENV=development, allowed only locally)", problem)
