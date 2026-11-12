"""Published JSON Schemas of the contract (schemas/*.json, GET /v1/schema/...)."""

from typing import Any

from pydantic import BaseModel

from qgen.contract.options import JobOptions
from qgen.contract.report import Report
from qgen.contract.units import UnitsResult

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

# name -> (file name, model, description)
SCHEMAS: dict[str, tuple[str, type[BaseModel], str]] = {
    "lesson": ("lesson.schema.json", UnitsResult, "Formát units (GET /v1/jobs/{id}/result)"),
    "report": ("report.schema.json", Report, "Report (GET /v1/jobs/{id}/report)"),
    "options": ("options.schema.json", JobOptions, "Vstupné parametre options (POST /v1/jobs)"),
}


def build_schema(name: str) -> dict[str, Any]:
    file_name, model, description = SCHEMAS[name]
    schema = model.model_json_schema(mode="validation")
    return {
        "$schema": JSON_SCHEMA_DIALECT,
        "$id": f"https://qgen/schemas/{file_name}",
        "description": description,
        **schema,
    }
