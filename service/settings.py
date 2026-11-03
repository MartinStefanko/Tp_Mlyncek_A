"""Service settings from the environment (Príloha D – Konfigurácia).

Only what the API and the mock need. Pipeline settings (LLM, embeddings, thresholds)
will live in qgen/config.py.
"""

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from qgen.contract import OptionDefaults


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QGEN_", env_file=".env", extra="ignore")

    backend: Literal["mock"] = Field(default="mock", description="Job backend; only 'mock' exists now.")

    # Access
    api_keys: SecretStr = Field(default=SecretStr(""), description="Comma-separated keys; empty = no check.")

    # Input limits
    max_files: int = Field(default=10, ge=1)
    max_upload_mb: float = Field(default=50, gt=0)
    max_pages: int = Field(default=300, ge=1)
    max_active_jobs: int = Field(default=10, ge=1)

    # Retention
    result_ttl_days: float = Field(default=7, gt=0)
    failed_input_ttl_hours: float = Field(default=24, gt=0)

    # Option defaults
    default_lesson_minutes: int = 45
    ocr_enabled: bool = False

    # Webhooks
    webhook_secret: SecretStr = SecretStr("")
    webhook_allowed_hosts: str = Field(default="", description="Comma-separated hosts; empty = webhooks off.")
    webhook_timeout_s: float = Field(default=5, gt=0)

    # Models (shown in /v1/info)
    llm_provider: str = "gemini"
    llm_model: str = "gemini-4-argon"
    embed_provider: str = "gemini"
    embed_model: str = "gemini-embedding-001"

    # Mock only
    mock_duration_s: float = Field(default=30, gt=0, description="Simulated duration of one job.")

    @property
    def api_key_list(self) -> list[str]:
        return _split_list(self.api_keys.get_secret_value())

    @property
    def allowed_webhook_hosts(self) -> set[str]:
        return {host.lower() for host in _split_list(self.webhook_allowed_hosts)}

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    def option_defaults(self) -> OptionDefaults:
        return OptionDefaults(ocr=self.ocr_enabled, lesson_minutes=self.default_lesson_minutes)
