"""Service settings: the shared `qgen.config.Settings` plus API-only fields.

All variables of Príloha D (limits, webhooks, TTL, models, option defaults) are defined
once in qgen/config.py. Here are only the fields that exist just for the API / mock.
"""

from typing import Literal

from pydantic import Field

from qgen.config import Settings


class ServiceSettings(Settings):
    backend: Literal["mock"] = Field(default="mock", description="Job backend; only 'mock' exists now.")

    # Mock only
    mock_duration_s: float = Field(default=30, gt=0, description="Simulated duration of one job.")
