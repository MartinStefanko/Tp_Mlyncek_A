"""Input of the split into lessons (kap. 8): sections (tematické celky) and their questions.

A section knows only what the algorithm needs: id, prerequisites, cited pages (order of the
document and the syllabus page ranges), time and topic. `build_split_sections` makes them
from the pipeline sections (LO dicts) and questions; nothing here calls an LLM or embeddings.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional

from qgen.context_builder import parse_source_refs
from qgen.contract.units import QuestionType

PageRef = tuple[Optional[str], int]  # ("D1", 14); None = the only document (D1)

_DOC_NUMBER_RE = re.compile(r"^D(\d+)$", re.I)

__all__ = [
    "PageRef",
    "SplitQuestion",
    "SplitSection",
    "build_split_sections",
    "document_number",
    "page_sort_key",
    "section_label",
]


@dataclass(frozen=True)
class SplitSection:
    """One section (tematický celok); `seconds` = Σ time_to_handle + time_to_explain of its questions."""

    id: int
    prerequisites: tuple[int, ...] = ()
    pages: tuple[PageRef, ...] = ()
    seconds: int = 0
    topic: Optional[str] = None


@dataclass(frozen=True)
class SplitQuestion:
    """A question for the order inside a lesson; `key` is the caller's identifier (e.g. an index)."""

    key: Any
    section_id: int
    type: QuestionType


def section_label(section_id: int) -> str:
    """Id as in the report ("S-12")."""
    return f"S-{section_id}"


def document_number(source_id: Optional[str]) -> Optional[int]:
    """'D3' → 3; None (a reference without a document) → 1; another text → None."""
    if source_id is None:
        return 1
    match = _DOC_NUMBER_RE.match(str(source_id).strip())
    return int(match.group(1)) if match else None


def page_sort_key(section: SplitSection) -> tuple:
    """Place in the document like qgen.lo.pipeline.lo_page_sort_key (first cited page, then
    id), with the document number compared as a number (D10 after D9). A reference without
    a document sorts first (as "" there); a section without pages goes to the end."""
    if not section.pages:
        return (math.inf, math.inf, section.id)
    source_id, page = section.pages[0]
    if source_id is None:
        return (0, page, section.id)
    number = document_number(source_id)
    return (number if number is not None else math.inf, page, section.id)


def build_split_sections(
    sections: list[dict],
    items: list[dict],
    topics: Optional[dict[Any, str]] = None,
) -> list[SplitSection]:
    """Sections of the pipeline (id, prerekvizity, citovane_zdroje) with the time of their
    questions (lo_id, time_to_handle, time_to_explain).

    A section without questions is left out (it would give an empty lesson). Prerequisites
    that are not among the returned sections (dropped by a filter) are left out too.
    """
    seconds: dict[Any, int] = {}
    for item in items:
        lo_id = item.get("lo_id")
        seconds[lo_id] = seconds.get(lo_id, 0) + _int(item.get("time_to_handle")) + _int(item.get("time_to_explain"))

    kept = [lo for lo in sections if lo.get("id") in seconds]
    kept_ids = {lo["id"] for lo in kept}
    result = []
    for lo in kept:
        prerequisites = []
        for p in lo.get("prerekvizity") or []:
            if isinstance(p, int) and p in kept_ids and p not in prerequisites:
                prerequisites.append(p)
        result.append(
            SplitSection(
                id=lo["id"],
                prerequisites=tuple(prerequisites),
                pages=tuple(parse_source_refs(lo.get("citovane_zdroje", []))),
                seconds=seconds[lo["id"]],
                topic=(topics or {}).get(lo["id"]),
            )
        )
    return result


def _int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
