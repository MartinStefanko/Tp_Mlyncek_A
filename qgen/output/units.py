"""Mapping of the pipeline questions to the format units (kap. 6).

Since step 13 the pipeline returns the questions in the contract shape (qgen.questions.types):

    units           one lesson (the split into lessons is step 15–16); title = names of the
                    input PDFs, description = list of section titles
    name            name from the LLM (empty: title of the question's section, else the
                    description); length limits in qgen.output.limits
    description, type, time_*, choices   as returned by the pipeline
    hints           as returned by the pipeline (one hint with the document name and pages,
                    qgen.questions.hints); url cleaned by nh3 (qgen.output.sanitize), a hint
                    without text after cleaning is dropped (invalid_hint)
"""

from __future__ import annotations

from pathlib import PurePath
from typing import Optional

from qgen.contract.report import ReportWarning
from qgen.contract.units import UNIT_DESCRIPTION_MAX, UNIT_TITLE_MAX
from qgen.output.limits import cut_at_word
from qgen.output.sanitize import clean_hint_html, has_visible_text
from qgen.questions.types import contract_part

DEFAULT_UNIT_TITLE = "Lekcia"

__all__ = ["DEFAULT_UNIT_TITLE", "question_draft", "unit_description", "unit_title"]


def question_draft(item: dict, section: Optional[dict], position: int) -> tuple[dict, list[ReportWarning]]:
    """Contract fields of the question before the length limits, and its hint warnings."""
    data = contract_part(item)
    name_source = data.get("name") or (section or {}).get("vzdelavaci_objekt") or str(data.get("description") or "")
    data["name"] = " ".join(str(name_source).split())
    data["choices"] = [dict(choice) if isinstance(choice, dict) else choice for choice in data.get("choices") or []]
    warnings: list[ReportWarning] = []
    hints = []
    for number, hint in enumerate(data.get("hints") or [], start=1):
        if not isinstance(hint, dict):
            continue
        name = str(hint.get("name", "")).strip()
        url = clean_hint_html(str(hint.get("url", "")))
        if not name or not has_visible_text(url):
            warnings.append(
                ReportWarning(
                    code="invalid_hint",
                    message=f"Question {position}: hint {number} was removed: no text after HTML cleaning.",
                )
            )
            continue
        hints.append({"name": name, "url": url})
    data["hints"] = hints
    return data, warnings


def unit_title(input_names: list[str]) -> str:
    names = [PurePath(name).stem or name for name in input_names]
    return cut_at_word(", ".join(names) or DEFAULT_UNIT_TITLE, UNIT_TITLE_MAX) or DEFAULT_UNIT_TITLE


def unit_description(sections: list[dict]) -> str:
    """`sections`: normalized sections (qgen.cli.normalize_lo_for_export)."""
    if not sections:
        return ""
    titles = ", ".join(lo["vzdelavaci_objekt"] for lo in sections if lo["vzdelavaci_objekt"])
    return cut_at_word("Tematické celky: " + titles, UNIT_DESCRIPTION_MAX)
