"""Order of the sections (kap. 8, step a): topological by prerequisites (Kahn's algorithm),
ties by the place in the document.

A cycle is broken on the edge towards a later page: in a cyclic group of sections the one
that comes first in the document stops requiring the members of the group, so the result
keeps the order of the document. Every removed edge is a warning `prerequisite_cycle`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qgen.contract.report import ReportWarning
from qgen.split.sections import SplitSection, page_sort_key, section_label

__all__ = ["SectionOrder", "order_sections"]


@dataclass
class SectionOrder:
    sections: list[SplitSection]
    warnings: list[ReportWarning] = field(default_factory=list)
    # (section, prerequisite) edges removed to break cycles; not checked again later.
    broken: set[tuple[int, int]] = field(default_factory=set)


def order_sections(sections: list[SplitSection]) -> SectionOrder:
    by_id: dict[int, SplitSection] = {}
    for section in sections:
        if section.id in by_id:
            raise ValueError(f"duplicate section id {section.id}")
        by_id[section.id] = section
    keys = {sid: page_sort_key(s) for sid, s in by_id.items()}
    pending = {sid: {p for p in s.prerequisites if p in by_id} for sid, s in by_id.items()}

    result = SectionOrder(sections=[])
    for sid in sorted(by_id, key=lambda x: keys[x]):
        if sid in pending[sid]:  # a section that requires itself: a cycle of one
            _remove_edge(sid, sid, pending, result)
    done: set[int] = set()
    while len(done) < len(by_id):
        ready = [sid for sid in by_id if sid not in done and not pending[sid] - done]
        if not ready:
            _break_cycle(by_id, keys, pending, done, result)
            continue
        chosen = min(ready, key=lambda sid: keys[sid])
        done.add(chosen)
        result.sections.append(by_id[chosen])
    return result


def _break_cycle(by_id, keys, pending, done, result: SectionOrder) -> None:
    remaining = [sid for sid in by_id if sid not in done]
    edges = {sid: pending[sid] - done for sid in remaining}
    reach = {sid: _reachable(sid, edges) for sid in remaining}
    cyclic = [sid for sid in remaining if sid in reach[sid]]
    first = min(cyclic, key=lambda sid: keys[sid])
    group = {sid for sid in reach[first] if first in reach[sid]}
    for prerequisite in sorted(edges[first] & group, key=lambda sid: keys[sid]):
        _remove_edge(first, prerequisite, pending, result)


def _remove_edge(sid: int, prerequisite: int, pending, result: SectionOrder) -> None:
    pending[sid].discard(prerequisite)
    result.broken.add((sid, prerequisite))
    where = "itself" if sid == prerequisite else f"{section_label(prerequisite)} (later in the document)"
    result.warnings.append(
        ReportWarning(
            code="prerequisite_cycle",
            message=f"Prerequisite cycle broken: {section_label(sid)} no longer requires {where}.",
        )
    )


def _reachable(start: int, edges: dict[int, set[int]]) -> set[int]:
    """Sections reachable from `start` over at least one edge."""
    seen: set[int] = set()
    stack = list(edges.get(start, ()))
    while stack:
        sid = stack.pop()
        if sid in seen:
            continue
        seen.add(sid)
        stack.extend(edges.get(sid, ()))
    return seen
