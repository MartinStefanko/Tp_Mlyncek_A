"""HTML of the hints (qgen.output.sanitize, nh3): allowed tags only, no attributes."""

import pytest

from qgen.output import clean_hint_html
from qgen.output.sanitize import HINT_TAGS, has_visible_text
from tests.output.data import build, question, section


@pytest.mark.parametrize(
    ("dirty", "clean"),
    [
        ("<p>a<script>alert('x')</script>b</p>", "<p>ab</p>"),
        ('<p onclick="steal()">a</p>', "<p>a</p>"),
        ('<p style="color:red">a</p>', "<p>a</p>"),
        ("<style>p{display:none}</style><p>a</p>", "<p>a</p>"),
        ('<p>a<img src="x" onerror="steal()"></p>', "<p>a</p>"),
        ('<a href="javascript:steal()">odkaz</a>', "odkaz"),
        ('<div class="x"><p>a</p></div>', "<p>a</p>"),
    ],
)
def test_dangerous_html_is_removed(dirty, clean):
    assert clean_hint_html(dirty) == clean


def test_allowed_tags_stay():
    allowed = ("<p>a<br><strong>b</strong><em>c</em><code>d</code><sub>1</sub><sup>2</sup></p>"
               "<ul><li>e</li></ul><ol><li>f</li></ol><pre>g</pre>")
    assert clean_hint_html(allowed) == allowed
    assert HINT_TAGS == {"p", "br", "strong", "em", "ul", "ol", "li", "code", "pre", "sub", "sup"}


def test_visible_text():
    assert has_visible_text("<p>a</p>")
    assert not has_visible_text("<p> <br> </p>")
    assert not has_visible_text("")
    assert not has_visible_text("<p>&nbsp;</p>")


def test_hint_empty_after_cleaning_is_dropped_with_invalid_hint():
    questions = [
        question(1, hints=("<script>alert(1)</script>",)),
        question(1, hints=('<p onclick="x">eko.pdf, strana 1</p><img src=x>',)),
        question(1, hints=("<style>p{}</style><p> </p>",)),
    ]
    out = build([section(1, "A")], questions)
    first, second, third = out.result.units[0].questions  # the questions stay
    assert first.hints == [] and third.hints == []
    assert [h.model_dump() for h in second.hints] == [{"name": "Zdroj", "url": "<p>eko.pdf, strana 1</p>"}]
    assert [(w.code, w.message) for w in out.report.warnings] == [
        ("invalid_hint", "Question 1: hint 1 was removed: no text after HTML cleaning."),
        ("invalid_hint", "Question 3: hint 1 was removed: no text after HTML cleaning."),
    ]
    assert out.dropped == 0
