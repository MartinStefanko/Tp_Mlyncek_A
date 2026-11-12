"""Start-up security check (step 17): QGEN_ENV=production refuses, development warns."""

import logging
import subprocess
import sys
from pathlib import Path

import pytest

from service import webhook_worker as webhook_worker_main
from service import worker as worker_main
from service.settings import ServiceSettings
from service.startup import StartupConfigError, check_service_config, security_problems

ROOT = Path(__file__).resolve().parents[2]


def settings(**values) -> ServiceSettings:
    base = {"api_keys": "", "webhook_allowed_hosts": "", "webhook_secret": "", "env": "production"}
    return ServiceSettings(_env_file=None, **(base | values))


def test_default_env_is_production(monkeypatch):
    monkeypatch.delenv("QGEN_ENV", raising=False)
    assert ServiceSettings(_env_file=None).env == "production"


@pytest.mark.parametrize(
    "values, count",
    [
        ({"api_keys": "k"}, 0),
        ({"api_keys": "k", "webhook_allowed_hosts": "eceh", "webhook_secret": "s"}, 0),
        ({"api_keys": "k", "webhook_secret": ""}, 0),  # webhooks off: no secret needed
        ({}, 1),  # no API key
        ({"api_keys": "k", "webhook_allowed_hosts": "eceh"}, 1),  # webhooks on, no secret
        ({"webhook_allowed_hosts": "eceh"}, 2),
    ],
)
def test_security_problems(values, count):
    assert len(security_problems(settings(**values))) == count


def test_production_refuses():
    with pytest.raises(StartupConfigError) as exc:
        check_service_config(settings(webhook_allowed_hosts="eceh"), "api")
    assert "QGEN_API_KEYS" in str(exc.value) and "QGEN_WEBHOOK_SECRET" in str(exc.value)


def test_development_only_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="qgen.startup"):
        check_service_config(settings(env="development", webhook_allowed_hosts="eceh"), "api")
    assert len(caplog.records) == 2


def test_secret_value_is_never_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="qgen.startup"):
        check_service_config(settings(env="development", webhook_secret="very-secret"), "api")
    assert "very-secret" not in caplog.text


def test_complete_production_config_passes():
    check_service_config(settings(api_keys="k", webhook_allowed_hosts="eceh", webhook_secret="s"), "api")


@pytest.mark.parametrize(
    "providers, name",
    [
        ({"llm_provider": "replay"}, "QGEN_LLM_PROVIDER"),
        ({"embed_provider": "replay"}, "QGEN_EMBED_PROVIDER"),
        ({"llm_provider": "replay", "embed_provider": "replay"}, "QGEN_LLM_PROVIDER and QGEN_EMBED_PROVIDER"),
    ],
)
def test_production_refuses_replay(providers, name):
    complete = {"api_keys": "k", "webhook_allowed_hosts": "eceh", "webhook_secret": "s"}
    with pytest.raises(StartupConfigError) as exc:
        check_service_config(settings(**complete, **providers), "worker")
    assert f"{name}=replay" in str(exc.value)


def test_development_allows_replay(caplog):
    complete = {"api_keys": "k", "webhook_allowed_hosts": "eceh", "webhook_secret": "s"}
    with caplog.at_level(logging.WARNING, logger="qgen.startup"):
        check_service_config(settings(env="development", llm_provider="replay", embed_provider="replay", **complete), "api")
    assert caplog.records == []  # the loud REPLAY warning comes from qgen.llm.factory.check_config


def test_worker_does_not_start_with_replay_in_production(monkeypatch):
    """LLM configuration is fine (check_config passes), but production with replay → exit 1."""
    bad = settings(api_keys="k", llm_provider="replay", embed_provider="replay")
    monkeypatch.setattr(worker_main, "check_config", lambda s: None)
    monkeypatch.setattr(worker_main, "ServiceSettings", lambda: bad)
    assert worker_main.main() == 1


def test_worker_does_not_start_without_api_key(monkeypatch):
    """LLM configuration is fine, but production without QGEN_API_KEYS → exit 1."""
    bad = settings()
    monkeypatch.setattr(worker_main, "check_config", lambda s: None)
    monkeypatch.setattr(worker_main, "ServiceSettings", lambda: bad)
    assert worker_main.main() == 1


def test_webhook_worker_does_not_start_without_secret(monkeypatch):
    bad = settings(api_keys="k", webhook_allowed_hosts="eceh")
    monkeypatch.setattr(webhook_worker_main, "ServiceSettings", lambda: bad)
    assert webhook_worker_main.main() == 1


def _import_app(env: dict, cwd: Path) -> subprocess.CompletedProcess:
    import os

    clean = {k: v for k, v in os.environ.items() if not k.startswith("QGEN_")}
    clean.update({"QGEN_BACKEND": "mock", "PYTHONPATH": str(ROOT)} | env)
    return subprocess.run(
        [sys.executable, "-c", "import service.main as m; print(m.app.title)"],
        cwd=cwd, env=clean, capture_output=True, text=True, timeout=60,
    )


def test_uvicorn_app_refuses_in_production(tmp_path):
    result = _import_app({"QGEN_API_KEYS": ""}, tmp_path)
    assert result.returncode != 0
    assert "StartupConfigError" in result.stderr


def test_uvicorn_app_starts_in_development(tmp_path):
    result = _import_app({"QGEN_API_KEYS": "", "QGEN_ENV": "development"}, tmp_path)
    assert result.returncode == 0, result.stderr


def test_text_extraction_has_no_deprecated_fitz_import():
    """`import fitz` gives a FutureWarning in pymupdf; the module imports pymupdf."""
    result = subprocess.run(
        [sys.executable, "-W", "error::FutureWarning", "-c", "import qgen.extraction.text_extraction"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
