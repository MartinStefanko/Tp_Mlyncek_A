"""Length limits of ECEH (kap. 6 – Dĺžkové limity a čistenie HTML), counted in characters.

    units.title 255, units.description 4000, questions.name 255, questions.description 4000,
    choices.text 4000, hints.name 50, hints.url 2000      (qgen.contract.units)

A question with a field over its limit:
    1. one LLM call per question (role "questions", JSON mode, language of the document):
       only the too long fields among name, description and choices[i], each with its limit;
       the answer {"items": [{"field", "text"}]} replaces only the text of the same field,
       so the order of the choices and is_correct never change. Hints are not sent: the
       code builds them (document + pages, qgen.questions.hints) within the limits.
    2. a name still too long is cut at a word end (cut_at_word)
    3. a description, choice or hint still too long: the question is left out with the
       warning field_too_long (field, length, limit; never the text)
    4. a failed call (LLMError) does not stop the job: warning shortening_failed, then 2–3.
The calls of all questions go through qgen.llm.parallel.run_batches; the drafts change
only in the calling thread.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from qgen.config import Settings
from qgen.contract.units import (
    CHOICE_TEXT_MAX,
    HINT_NAME_MAX,
    HINT_URL_MAX,
    QUESTION_DESCRIPTION_MAX,
    QUESTION_NAME_MAX,
)
from qgen.llm.base import LLMClient
from qgen.llm.errors import LLMError
from qgen.llm.parallel import run_batches
from qgen.llm.retry import generate_json

SHORTEN_ROLE = "questions"
SHORTENED_FIELD = re.compile(r"^(?:name|description|choices\[(\d+)\])$")

__all__ = [
    "SHORTEN_EN",
    "SHORTEN_ROLE",
    "SHORTEN_SK",
    "TooLongField",
    "build_shorten_prompt",
    "cut_at_word",
    "enforce_limits",
    "shorten_questions",
    "too_long_fields",
]


@dataclass(frozen=True)
class TooLongField:
    field: str
    length: int
    limit: int


def cut_at_word(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit + 1].rsplit(" ", 1)[0] if " " in text[: limit + 1] else text[:limit]
    return cut[:limit].rstrip(" ,;:-–")


def _fields(data: dict) -> list[tuple[str, Any, int]]:
    """(field, value, limit) of every limited text field of a question draft."""
    fields: list[tuple[str, Any, int]] = [
        ("name", data.get("name"), QUESTION_NAME_MAX),
        ("description", data.get("description"), QUESTION_DESCRIPTION_MAX),
    ]
    for index, choice in enumerate(data.get("choices") or []):
        if isinstance(choice, dict):
            fields.append((f"choices[{index}]", choice.get("text"), CHOICE_TEXT_MAX))
    for index, hint in enumerate(data.get("hints") or []):
        if isinstance(hint, dict):
            fields.append((f"hints[{index}].name", hint.get("name"), HINT_NAME_MAX))
            fields.append((f"hints[{index}].url", hint.get("url"), HINT_URL_MAX))
    return fields


def too_long_fields(data: dict) -> list[TooLongField]:
    return [
        TooLongField(field, len(value), limit)
        for field, value, limit in _fields(data)
        if isinstance(value, str) and len(value) > limit
    ]


def _shortenable(data: dict) -> list[dict]:
    """The too long fields the LLM may shorten (no hints), with their limit and text."""
    texts = {field: value for field, value, _ in _fields(data)}
    return [
        {"field": found.field, "limit": found.limit, "text": texts[found.field]}
        for found in too_long_fields(data)
        if SHORTENED_FIELD.match(found.field)
    ]


# --- prompt -----------------------------------------------------------------------------------

SHORTEN_SK = """
Nasledujúce texty jednej otázky sú príliš dlhé pre databázu. Skráť každý text tak, aby mal najviac
"limit" znakov (počítajú sa všetky znaky vrátane medzier).

Pravidlá:
- zachovaj význam, odbornú správnosť a jazyk textu (slovenčina),
- nepridávaj nové informácie,
- pri možnosti odpovede (choices[...]) skráť iba jej text; to, či je možnosť správna, sa nesmie zmeniť,
- "field" vráť presne tak, ako je na vstupe.

Vráť iba JSON objekt {"items": [{"field": "...", "text": "..."}]}, jednu položku pre každý text zo vstupu.

Texty (JSON):
{fields_json}
"""

SHORTEN_EN = """
The following texts of one question are too long for the database. Shorten every text so that it
has at most "limit" characters (all characters count, including spaces).

Rules:
- keep the meaning, the factual correctness and the language of the text (English),
- do not add new information,
- for an answer option (choices[...]) shorten only its text; whether the option is correct must not change,
- return "field" exactly as in the input.

Return only a JSON object {"items": [{"field": "...", "text": "..."}]}, one item for every input text.

Texts (JSON):
{fields_json}
"""


def build_shorten_prompt(fields: list[dict], language: str) -> str:
    template = SHORTEN_EN if language == "en" else SHORTEN_SK
    return template.replace("{fields_json}", json.dumps(fields, ensure_ascii=False, indent=2)).strip()


# --- shortening -------------------------------------------------------------------------------


def _apply(data: dict, requested: list[dict], answer: list) -> None:
    """Replaces the text of the requested fields by the answer (non-empty strings only)."""
    wanted = {field["field"] for field in requested}
    for row in answer:
        if not isinstance(row, dict):
            continue
        field, text = row.get("field"), row.get("text")
        if field not in wanted or not isinstance(text, str) or not text.strip():
            continue
        choice_index = SHORTENED_FIELD.match(field).group(1)
        if choice_index is not None:
            data["choices"][int(choice_index)]["text"] = text.strip()
        else:
            data[field] = text.strip()


def shorten_questions(
    drafts: list[dict],
    *,
    llm: Optional[LLMClient],
    settings: Settings,
    language: str,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> list[int]:
    """Shortens the too long fields of the drafts in place; returns the indexes (in `drafts`)
    whose call failed. One call per draft with a too long name, description or choice."""
    requests = [(index, _shortenable(data)) for index, data in enumerate(drafts)]
    requests = [(index, fields) for index, fields in requests if fields]
    if not requests:
        return []

    def task(fields: list[dict]) -> Callable[[], Optional[list]]:
        def call() -> Optional[list]:
            try:
                return generate_json(
                    llm,
                    build_shorten_prompt(fields, language),
                    role=SHORTEN_ROLE,
                    settings=settings,
                    items_key="items",
                )
            except LLMError:  # the job goes on with the original texts
                return None

        return call

    done = 0

    def on_done(_index: int, _answer: Optional[list]) -> None:
        nonlocal done
        done += 1
        if progress_callback is not None:
            progress_callback("output", done, len(requests))

    answers = run_batches([task(fields) for _, fields in requests], settings=settings, on_done=on_done)
    failed = []
    for (index, fields), answer in zip(requests, answers):
        if answer is None:
            failed.append(index)
        else:
            _apply(drafts[index], fields, answer)
    return failed


def enforce_limits(data: dict) -> Optional[TooLongField]:
    """After the shortening: the name is cut at a word end; returns the first other field
    still over its limit (the question is left out), or None."""
    if isinstance(data.get("name"), str):
        data["name"] = cut_at_word(data["name"], QUESTION_NAME_MAX)
    remaining = too_long_fields(data)
    return remaining[0] if remaining else None
