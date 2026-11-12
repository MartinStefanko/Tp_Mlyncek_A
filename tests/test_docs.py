"""Documentation and compose files do not drift from the code (step 17)."""

import re
from pathlib import Path

from service.settings import ServiceSettings

ROOT = Path(__file__).resolve().parent.parent
DEPLOYMENT = (ROOT / "docs" / "deployment.md").read_text(encoding="utf-8")
ENV_EXAMPLE = (ROOT / ".env.example").read_text(encoding="utf-8")
VENDOR_KEYS = {"gemini_api_key": "GEMINI_API_KEY", "anthropic_api_key": "ANTHROPIC_API_KEY"}


def _variable(field: str) -> str:
    return VENDOR_KEYS.get(field, f"QGEN_{field.upper()}")


def _in_table(variable: str) -> bool:
    if f"`{variable}`" in DEPLOYMENT:
        return True
    # paired rows, e.g. `QGEN_HINT_NAME_SK` / `_EN`
    if variable.endswith("_EN"):
        return re.search(rf"`{re.escape(variable[:-3])}_SK` / `_EN`", DEPLOYMENT) is not None
    return False


def test_deployment_doc_lists_every_setting():
    missing = [_variable(name) for name in ServiceSettings.model_fields if not _in_table(_variable(name))]
    assert missing == []


def test_compose_variables_are_documented():
    """Every ${QGEN_…} of the compose files is in .env.example and in docs/deployment.md."""
    used = set()
    for path in ROOT.glob("docker-compose*.yml"):
        used |= set(re.findall(r"\$\{(QGEN_[A-Z_]+)", path.read_text(encoding="utf-8")))
    internal = {"QGEN_DEV_NETWORK", "QGEN_REMOTE_NETWORK", "QGEN_REPLAY_API_KEY", "QGEN_REPLAY_WEBHOOK_SECRET"}
    used -= internal
    replay_only = {"QGEN_LLM_REPLAY_DELAY_S", "QGEN_LLM_REPLAY_FAIL", "QGEN_MAINTENANCE_INTERVAL_S"}
    assert [v for v in sorted(used) if f"{v}=" not in ENV_EXAMPLE] == []
    assert [v for v in sorted(used - replay_only) if f"`{v}`" not in DEPLOYMENT] == []


def test_production_compose_publishes_no_port():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert not re.search(r"^\s+ports:", text, flags=re.MULTILINE)
    assert "external: true" in text


def test_dockerignore_is_a_whitelist_without_secrets():
    lines = [l.strip() for l in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()]
    lines = [l for l in lines if l and not l.startswith("#")]
    assert lines[0] == "*"
    assert {l for l in lines if l.startswith("!")} == {"!qgen/", "!service/", "!requirements-*.txt"}


def test_requirements_are_pinned():
    for name in ("requirements-mock.txt", "requirements-service.txt", "requirements-ocr.txt", "requirements.txt"):
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if line and not line.startswith("-r"):
                assert "==" in line, f"{name}: {line} is not pinned"
