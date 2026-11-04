"""Named changes of the baseline recording for the refactor equivalence test (step 4).

Step 4 removed the testing of Python tasks. This changed two prompts on purpose. The
baseline recording (baseline/llm_calls.jsonl) still has the old prompts, so the replay
client applies these transforms to the RECORDED inputs before it matches calls. Every
other part of every prompt must stay byte-identical.

All changes of the recording live in this module. Each transform counts how often it
changed something, so the test can check that none of them silently matches nothing.
"""

import re

from tests.replay_client import RecordingTransform

# --- 1. Fixed reason in the "document classification" block of the item generation prompt.
# Before: the reason came from the LLM call classify_document_for_python_items (now removed).
# After: the block is fixed text with is_python_document = false and the existing fallback
# reason of each language ("bez zdovodnenia" / "no justification").
_CLASSIFICATION_REASON = [
    (re.compile(r"(Klasifikácia dokumentu:\n- is_python_document: false\n- reason: )[^\n]*"), "bez zdovodnenia"),
    (re.compile(r"(Document classification:\n- is_python_document: false\n- reason: )[^\n]*"), "no justification"),
]


def _fixed_classification_reason(method, contents):
    if method != "generate_content" or not isinstance(contents, str):
        return contents, 0
    total = 0
    for pattern, reason in _CLASSIFICATION_REASON:
        contents, n = pattern.subn(lambda m: m.group(1) + reason, contents)
        total += n
    return contents, total


FIXED_CLASSIFICATION_REASON = RecordingTransform(
    name="fixed_classification_reason",
    description=(
        "Item generation prompt: the LLM-written 'reason' line of the document classification "
        "block is replaced by the fixed fallback text used after step 4."
    ),
    apply=_fixed_classification_reason,
)

# --- 2. Python fields removed from every item in the item evaluation prompt.
# Before: each item block listed jazyk, execution_mode, function_name, kod_riesenia and
# test_cases between "- odpoved:" and "- napoveda:". After: these 5 lines are gone.
# In the baseline all 5 values were empty, so each line fits on one line.
_ITEM_PYTHON_FIELD_LINES = re.compile(
    r"- jazyk: [^\n]*\n"
    r"- execution_mode: [^\n]*\n"
    r"- function_name: [^\n]*\n"
    r"- kod_riesenia: [^\n]*\n"
    r"- test_cases: [^\n]*\n"
)


def _drop_item_python_fields(method, contents):
    if method != "generate_content" or not isinstance(contents, str):
        return contents, 0
    return _ITEM_PYTHON_FIELD_LINES.subn("", contents)


DROP_ITEM_PYTHON_FIELDS = RecordingTransform(
    name="drop_item_python_fields",
    description=(
        "Item evaluation prompt: the 5 lines jazyk / execution_mode / function_name / "
        "kod_riesenia / test_cases are removed from each item block."
    ),
    apply=_drop_item_python_fields,
)


def step4_transforms():
    """Fresh transform objects (hit counters start at 0) for one replay client."""
    return [
        RecordingTransform(t.name, t.description, t.apply)
        for t in (FIXED_CLASSIFICATION_REASON, DROP_ITEM_PYTHON_FIELDS)
    ]


# Recorded calls that no longer exist after step 4 (removed Python document classification).
REMOVED_CALL_MARKERS = (
    "rozhodni, ci je to dokument o programovacom jazyku Python",
)

# Item fields removed in step 4; they are dropped from the baseline before comparison.
REMOVED_ITEM_FIELDS = (
    "jazyk",
    "kod_riesenia",
    "execution_mode",
    "function_name",
    "automaticky_testovatelna",
    "test_cases",
)
