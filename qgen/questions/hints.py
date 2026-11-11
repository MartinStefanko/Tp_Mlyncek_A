"""Hint of a question (krok 13c): a reference to the source document and pages.

The LLM writes no hints. After the generation the code builds one hint per question from
its own sources (`source`, e.g. ["D1:12", "D1:13"]); when the question has no valid own
source, the pages of its section (`citovane_zdroje`) are used. Format (templates in the
configuration, QGEN_HINT_*):

    name  "Zdroj" (English document: "Source")
    url   "<p>eko.pdf, strana 14</p>", "<p>eko.pdf, strany 14 – 15</p>",
          "<p>eko.pdf, strany 14, 16</p>",
          "<p>eko.pdf, strana 4; uvod.pdf, strany 2 – 3</p>"; English "page 14" / "pages 14 – 15"

The document name is the original name of the uploaded file (`source_name` of the
segments = InputFile.name), never D1.pdf. Documents in upload order, pages ascending.
ECEH limits: name <= 50, url <= 2000 (the HTML); a too long page list is shortened with "…".
options.questions.hints: 0 = no hint, >= 1 = one hint with the source.
"""

from __future__ import annotations

import html
from typing import Iterable, Optional

from qgen.config import Settings
from qgen.context_builder import make_source_ref, parse_source_refs
from qgen.contract.units import HINT_NAME_MAX, HINT_URL_MAX

RANGE_SEPARATOR = " – "
PAGE_SEPARATOR = ", "
DOCUMENT_SEPARATOR = "; "
ELLIPSIS = "…"

__all__ = ["build_source_hints", "format_page_runs", "page_runs"]


def page_runs(pages: Iterable[int]) -> list[tuple[int, int]]:
    """Sorted unique pages as runs of consecutive pages: [3, 4, 5, 9] -> [(3, 5), (9, 9)]."""
    runs: list[tuple[int, int]] = []
    for page in sorted(set(pages)):
        if runs and page == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], page)
        else:
            runs.append((page, page))
    return runs


def format_page_runs(runs: list[tuple[int, int]]) -> str:
    return PAGE_SEPARATOR.join(str(a) if a == b else f"{a}{RANGE_SEPARATOR}{b}" for a, b in runs)


def _valid_refs(cited, source_name_map: dict, allowed_refs: set) -> list[tuple[Optional[str], int]]:
    refs = []
    for source_id, page in parse_source_refs(cited):
        if page < 1:
            continue
        if allowed_refs and make_source_ref(source_id, page) not in allowed_refs:
            continue
        if not source_name_map.get(source_id):
            continue
        refs.append((source_id, page))
    return refs


def _templates(language: str, settings: Settings) -> tuple[str, str, str]:
    if language == "en":
        return settings.hint_name_en, settings.hint_page_en, settings.hint_pages_en
    return settings.hint_name_sk, settings.hint_page_sk, settings.hint_pages_sk


def _document_text(name: str, runs: list[tuple[int, int]], cut: bool, one: str, many: str) -> str:
    single = len(runs) == 1 and runs[0][0] == runs[0][1] and not cut
    pages = format_page_runs(runs) + (PAGE_SEPARATOR + ELLIPSIS if cut else "")
    return (one if single else many).replace("{pages}", pages).replace("{file}", name)


def _html(text: str) -> str:
    return f"<p>{html.escape(text, quote=False)}</p>"


def _fit_url(documents: list[tuple[str, list[tuple[int, int]]]], one: str, many: str) -> str:
    """HTML of the hint within HINT_URL_MAX: page runs are dropped from the end of the
    document with the most runs, then whole documents, then characters (marked with "…")."""
    docs = [[name, list(runs), False] for name, runs in documents]
    dropped_docs = False

    def text() -> str:
        joined = DOCUMENT_SEPARATOR.join(_document_text(n, r, c, one, many) for n, r, c in docs)
        return joined + (DOCUMENT_SEPARATOR + ELLIPSIS if dropped_docs else "")

    while len(_html(text())) > HINT_URL_MAX:
        longest = max(docs, key=lambda doc: len(doc[1]))
        if len(longest[1]) > 1:
            longest[1].pop()
            longest[2] = True
        elif len(docs) > 1:
            docs.pop()
            dropped_docs = True
        else:
            break
    plain = text()
    if len(_html(plain)) > HINT_URL_MAX:  # a single file name longer than the limit
        while len(_html(plain + ELLIPSIS)) > HINT_URL_MAX:
            plain = plain[:-1]
        plain = plain.rstrip() + ELLIPSIS
    return _html(plain)


def _cut_name(name: str) -> str:
    name = " ".join(name.split())
    if len(name) <= HINT_NAME_MAX:
        return name
    head = name[: HINT_NAME_MAX + 1]
    cut = head.rsplit(" ", 1)[0] if " " in head else name[:HINT_NAME_MAX]
    return cut[:HINT_NAME_MAX].rstrip()


def build_source_hints(
    item: dict,
    section: Optional[dict],
    *,
    source_name_map: dict,
    allowed_refs: set,
    language: str,
    count: int,
    settings: Settings,
) -> list[dict]:
    """[{"name", "url"}] with the source of the question, or [] (count 0, no valid pages).

    source_name_map: qgen.context_builder.build_source_name_map (source id -> file name,
    None -> name of the first file); allowed_refs: build_allowed_source_refs (pages that
    exist; empty = no check).
    """
    if count <= 0:
        return []
    refs = _valid_refs(item.get("source", []), source_name_map, allowed_refs)
    if not refs:
        refs = _valid_refs((section or {}).get("citovane_zdroje", []), source_name_map, allowed_refs)
    if not refs:
        return []

    order = {source_id: position for position, source_id in enumerate(source_name_map)}
    # A bare page number (one document) belongs to the first document: "D1:4" and "4" are one page.
    first_id = next((source_id for source_id in source_name_map if source_id is not None), None)
    pages: dict[Optional[str], set[int]] = {}
    for source_id, page in refs:
        pages.setdefault(first_id if source_id is None else source_id, set()).add(page)
    documents = [
        (source_name_map[source_id], page_runs(found))
        for source_id, found in sorted(pages.items(), key=lambda entry: order.get(entry[0], len(order)))
    ]
    name, one, many = _templates(language, settings)
    return [{"name": _cut_name(name), "url": _fit_url(documents, one, many)}]
