"""Check of the output before the worker saves it (step 14).

    1. units and report pass their Pydantic models (validated again from the JSON form)
    2. units and report pass the published JSON Schemas (lesson, report)
    3. the report matches the questions in units: every question has exactly one entry in
       question_meta (no extra or missing one, indexes in range) and its section_id is one
       of report.sections

OutputCheckError lists only paths and kinds of the errors, never a value (it could be text
of the document).
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Any

import jsonschema
from pydantic import ValidationError

from qgen.contract.report import Report
from qgen.contract.schemas import build_schema
from qgen.contract.units import UnitsResult

MAX_LISTED_PROBLEMS = 5

__all__ = ["OutputCheckError", "check_output", "check_report", "check_units"]


class OutputCheckError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        listed = "; ".join(problems[:MAX_LISTED_PROBLEMS])
        more = f" (+{len(problems) - MAX_LISTED_PROBLEMS} more)" if len(problems) > MAX_LISTED_PROBLEMS else ""
        super().__init__(f"output check failed: {listed}{more}")


@lru_cache(maxsize=None)
def _validator(name: str) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(build_schema(name))


def _model_problems(model: type, data: Any, name: str) -> list[str]:
    try:
        model.model_validate(data)
    except ValidationError as exc:
        return [f"{name} model: {'.'.join(map(str, error['loc'])) or '<root>'} {error['type']}" for error in exc.errors()]
    return []


def _schema_problems(data: Any, name: str) -> list[str]:
    return [f"{name} schema: {error.json_path} {error.validator}" for error in _validator(name).iter_errors(data)]


def check_units(data: Any) -> list[str]:
    """Problems of units in the JSON form ([] = valid)."""
    return _model_problems(UnitsResult, data, "units") + _schema_problems(data, "lesson")


def check_report(data: Any) -> list[str]:
    return _model_problems(Report, data, "report") + _schema_problems(data, "report")


def _consistency_problems(result: UnitsResult, report: Report) -> list[str]:
    expected = {(u, q) for u, unit in enumerate(result.units) for q in range(len(unit.questions))}
    pairs = Counter((meta.unit, meta.question) for meta in report.question_meta)
    problems = [f"question_meta: no entry for unit {u} question {q}" for u, q in sorted(expected - set(pairs))]
    problems += [f"question_meta: unit {u} question {q} not in units" for u, q in sorted(set(pairs) - expected)]
    problems += [f"question_meta: unit {u} question {q} listed {n} times" for (u, q), n in sorted(pairs.items()) if n > 1]
    section_ids = {section.id for section in report.sections}
    problems += [
        f"question_meta: unit {meta.unit} question {meta.question} has unknown section_id {meta.section_id}"
        for meta in report.question_meta
        if meta.section_id not in section_ids
    ]
    return problems


def check_output(result: UnitsResult, report: Report) -> None:
    """Raises OutputCheckError when the output must not be saved."""
    units_data = result.model_dump(mode="json")
    report_data = report.model_dump(mode="json")
    problems = check_units(units_data) + check_report(report_data)
    if not problems:
        problems = _consistency_problems(result, report)
    if problems:
        raise OutputCheckError(problems)
