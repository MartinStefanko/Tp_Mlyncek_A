"""Step a: topological order by prerequisites (Kahn), ties by pages, cycles broken."""

import pytest

from qgen.split import build_split_sections, order_sections, page_sort_key

from .helpers import section


def _ids(order):
    return [s.id for s in order.sections]


def test_ties_follow_the_pages():
    sections = [section(1, 5, page=9), section(2, 5, page=3), section(3, 5, page=6)]
    order = order_sections(sections)
    assert _ids(order) == [2, 3, 1]
    assert order.warnings == []


def test_prerequisite_comes_first_even_on_a_later_page():
    sections = [section(1, 5, page=1, prerequisites=(2,)), section(2, 5, page=8), section(3, 5, page=4)]
    assert _ids(order_sections(sections)) == [3, 2, 1]


def test_document_number_compares_as_number():
    sections = [section(1, 5, pages=(("D10", 1),)), section(2, 5, pages=(("D9", 50),)), section(3, 5, pages=())]
    assert _ids(order_sections(sections)) == [2, 1, 3]
    assert page_sort_key(sections[2])[0] == float("inf")


def test_two_section_cycle_broken_towards_later_page():
    sections = [section(1, 5, page=2, prerequisites=(2,)), section(2, 5, page=7, prerequisites=(1,))]
    order = order_sections(sections)
    assert _ids(order) == [1, 2]
    assert order.broken == {(1, 2)}
    assert [w.code for w in order.warnings] == ["prerequisite_cycle"]
    assert "S-1 no longer requires S-2" in order.warnings[0].message


def test_three_section_cycle_keeps_document_order():
    sections = [
        section(1, 5, page=1, prerequisites=(3,)),
        section(2, 5, page=2, prerequisites=(1,)),
        section(3, 5, page=3, prerequisites=(2,)),
    ]
    order = order_sections(sections)
    assert _ids(order) == [1, 2, 3]
    assert order.broken == {(1, 3)}


def test_section_waiting_for_a_cycle_keeps_its_edge():
    # 5 (page 1) needs 2; 2 and 3 form a cycle on pages 10, 11.
    sections = [
        section(5, 5, page=1, prerequisites=(2,)),
        section(2, 5, page=10, prerequisites=(3,)),
        section(3, 5, page=11, prerequisites=(2,)),
    ]
    order = order_sections(sections)
    assert _ids(order) == [2, 5, 3]
    assert order.broken == {(2, 3)}


def test_self_prerequisite_is_a_cycle():
    order = order_sections([section(1, 5, prerequisites=(1,)), section(2, 5)])
    assert _ids(order) == [1, 2]
    assert order.broken == {(1, 1)}


def test_unknown_prerequisites_are_ignored():
    order = order_sections([section(1, 5, prerequisites=(99,)), section(2, 5, prerequisites=(1,))])
    assert _ids(order) == [1, 2]
    assert order.warnings == []


def test_duplicate_ids_are_a_caller_error():
    with pytest.raises(ValueError):
        order_sections([section(1, 5), section(1, 6)])


def test_build_split_sections_sums_times_and_drops_sections_without_questions():
    los = [
        {"id": 1, "prerekvizity": [], "citovane_zdroje": ["D1:3"]},
        {"id": 2, "prerekvizity": [1, 3, "x"], "citovane_zdroje": ["D1:5", "D1:6"]},
        {"id": 3, "prerekvizity": [1], "citovane_zdroje": ["D1:8"]},
    ]
    items = [
        {"lo_id": 1, "time_to_handle": 60, "time_to_explain": 30},
        {"lo_id": 2, "time_to_handle": 300, "time_to_explain": 0},
        {"lo_id": 1, "time_to_handle": 120, "time_to_explain": None},
    ]
    found = build_split_sections(los, items)
    assert [(s.id, s.seconds, s.prerequisites, s.pages) for s in found] == [
        (1, 210, (), (("D1", 3),)),
        (2, 300, (1,), (("D1", 5), ("D1", 6))),
    ]
