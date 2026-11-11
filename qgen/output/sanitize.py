"""Cleaning of the hint HTML (hints[].url, kap. 6 – Dĺžkové limity a čistenie HTML).

Allowed tags only (HINT_TAGS), every attribute removed (also onclick, style); the content of
<script> and <style> is dropped, not only the tags. Other tags are removed, their text stays.
A hint without visible text after cleaning is dropped by qgen.output.units (invalid_hint).
"""

import html
import re

import nh3

HINT_TAGS = frozenset({"p", "br", "strong", "em", "ul", "ol", "li", "code", "pre", "sub", "sup"})
DROPPED_CONTENT_TAGS = frozenset({"script", "style"})

_TAG = re.compile(r"<[^>]*>")

__all__ = ["DROPPED_CONTENT_TAGS", "HINT_TAGS", "clean_hint_html", "has_visible_text"]


def clean_hint_html(text: str) -> str:
    return nh3.clean(
        text,
        tags=set(HINT_TAGS),
        clean_content_tags=set(DROPPED_CONTENT_TAGS),
        attributes={},
        link_rel=None,
    )


def has_visible_text(cleaned_html: str) -> bool:
    """True if the (already cleaned) HTML has some text besides tags and whitespace."""
    return bool(html.unescape(_TAG.sub(" ", cleaned_html)).strip())
