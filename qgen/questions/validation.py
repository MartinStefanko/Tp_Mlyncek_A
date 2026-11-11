"""Validation of the questions of the pipeline (internal format, qgen.questions.types).

Errors (the question is dropped, reason "validation"): missing fields, empty description,
unknown type, the type rules of the contract (qgen.contract.units.type_rule_errors: choice
>= 2 options and >= 1 correct, no duplicate texts; open exactly 1 correct choice; no_answer
without choices), is_correct that is not a bool, an "all / none of the above" choice, bad
`source`. Error texts start with "Polozka {id}" (qgen.questions.pipeline reads the ids).

Warnings (the question stays), with a code in `warning_counts` / `warning_items`:
    choice_count_mismatch   number of choices != options.questions.choice_options
Hints are not checked here: the LLM writes none, the pipeline builds them from the sources
at the end (qgen.questions.hints, krok 13c); only "hints" must be a list.
Length limits are the output step (krok 14).
"""

import json
from collections import Counter
from typing import Optional

from qgen.context_builder import parse_source_ref_strings
from qgen.contract.units import type_rule_errors
from qgen.questions.types import FORBIDDEN_CHOICE_RE, QUESTION_TYPES, QuestionSettings

REQUIRED_ITEM_FIELDS = {
    "id",
    "lo_id",
    "name",
    "description",
    "type",
    "time_to_handle",
    "time_to_explain",
    "hints",
    "choices",
    "source",
}

ALLOWED_ITEM_TYPES = set(QUESTION_TYPES)

WARNING_CHOICE_COUNT = "choice_count_mismatch"


def validate_items_json_text(text, allowed_pages=None, valid_lo_ids=None, question_settings=None):
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return {
            "is_valid_json": False,
            "is_valid": False,
            "errors": [f"Nevalidny JSON: {e}"],
            "warnings": [],
            "warning_counts": {},
            "warning_items": [],
            "stats": {"total": 0, "valid": 0, "invalid": 0},
        }
    return validate_items(data, allowed_pages=allowed_pages, valid_lo_ids=valid_lo_ids,
                          question_settings=question_settings)


def validate_items(data, allowed_pages=None, valid_lo_ids=None, question_settings: Optional[QuestionSettings] = None):
    qs = question_settings or QuestionSettings()
    report = {
        "is_valid_json": True,
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "warning_counts": {},
        "warning_items": [],
        "stats": {"total": 0, "valid": 0, "invalid": 0},
    }
    warning_counts: Counter = Counter()

    if not isinstance(data, list):
        report["is_valid"] = False
        report["errors"].append("Vystup poloziek musi byt JSON pole objektov.")
        report["stats"]["invalid"] = 1
        return report

    report["stats"]["total"] = len(data)
    known_sources = {str(p).strip() for p in (allowed_pages or []) if str(p).strip()}
    known_lo_ids = set(valid_lo_ids or [])
    item_ids = []

    def warn(code, item_id, prefix, message):
        warning_counts[code] += 1
        report["warnings"].append(f"{prefix} {message}")
        report["warning_items"].append({"code": code, "item_id": item_id, "message": message})

    for index, item in enumerate(data, start=1):
        prefix = f"Polozka na indexe {index}"
        item_valid = True

        if not isinstance(item, dict):
            report["errors"].append(f"{prefix} nie je objekt.")
            report["stats"]["invalid"] += 1
            report["is_valid"] = False
            continue

        missing_fields = sorted(REQUIRED_ITEM_FIELDS - set(item.keys()))
        if missing_fields:
            report["errors"].append(f"{prefix} nema povinne polia: {', '.join(missing_fields)}.")
            item_valid = False

        item_id = item.get("id")
        if not isinstance(item_id, int) or item_id <= 0:
            report["errors"].append(f"{prefix} ma neplatne pole 'id' (ocakavane kladne cele cislo).")
            item_valid = False
        else:
            item_ids.append(item_id)
            prefix = f"Polozka {item_id}"

        lo_id = item.get("lo_id")
        if not isinstance(lo_id, int) or isinstance(lo_id, bool) or lo_id <= 0:
            report["errors"].append(f"{prefix} ma neplatne pole 'lo_id' (ocakavane kladne cele cislo).")
            item_valid = False
        elif known_lo_ids and lo_id not in known_lo_ids:
            report["errors"].append(f"{prefix} odkazuje na neexistujuce LO id {lo_id}.")
            item_valid = False

        if not validate_non_empty_text(item.get("description"), prefix, "description", report):
            item_valid = False

        item_type = item.get("type")
        type_known = isinstance(item_type, str) and item_type in ALLOWED_ITEM_TYPES
        if not isinstance(item_type, str) or not item_type.strip():
            report["errors"].append(f"{prefix} ma prazdne alebo nespravne pole 'type'.")
            item_valid = False
        elif not type_known:
            report["errors"].append(f"{prefix} ma neplatnu hodnotu pola 'type' ({item_type}).")
            item_valid = False

        choices = item.get("choices")
        if not isinstance(choices, list):
            report["errors"].append(f"{prefix} ma nespravny typ pola 'choices' (ocakavany zoznam).")
            item_valid = False
            choices = []
        pairs = []
        for position, choice in enumerate(choices, start=1):
            text = choice.get("text") if isinstance(choice, dict) else None
            correct = choice.get("is_correct") if isinstance(choice, dict) else None
            if not isinstance(text, str) or not text.strip():
                report["errors"].append(f"{prefix} ma prazdnu moznost na pozicii {position}.")
                item_valid = False
            if not isinstance(correct, bool):
                report["errors"].append(f"{prefix} ma pri moznosti {position} 'is_correct', ktore nie je true/false.")
                item_valid = False
            pairs.append((str(text or ""), correct is True))

        if type_known:
            for error in type_rule_errors(item_type, pairs):
                report["errors"].append(f"{prefix}: {error}.")
                item_valid = False
            if item_type == "choice":
                forbidden = [text for text, _ in pairs if FORBIDDEN_CHOICE_RE.search(text)]
                if forbidden:
                    report["errors"].append(f"{prefix} ma nepovolenu moznost typu 'vsetky/ziadna z uvedenych'.")
                    item_valid = False
                if len(pairs) != qs.choice_options:
                    warn(WARNING_CHOICE_COUNT, item_id, prefix,
                         f"ma {len(pairs)} moznosti namiesto {qs.choice_options}.")

        if not isinstance(item.get("hints"), list):
            report["errors"].append(f"{prefix} ma nespravny typ pola 'hints' (ocakavany zoznam).")
            item_valid = False

        if not validate_sources(item.get("source"), prefix, report, known_sources):
            item_valid = False

        if item_valid:
            report["stats"]["valid"] += 1
        else:
            report["stats"]["invalid"] += 1
            report["is_valid"] = False

    duplicates = sorted({item_id for item_id in item_ids if item_ids.count(item_id) > 1})
    if duplicates:
        report["errors"].append(f"Duplicne item id: {', '.join(str(item_id) for item_id in duplicates)}.")
        report["is_valid"] = False

    report["stats"]["invalid"] = report["stats"]["total"] - report["stats"]["valid"]
    report["warning_counts"] = dict(warning_counts)
    return report


def validate_non_empty_text(value, prefix, field_name, report):
    if not isinstance(value, str):
        report["errors"].append(f"{prefix} ma nespravny typ pola '{field_name}' (ocakavany string).")
        return False
    if not value.strip():
        report["errors"].append(f"{prefix} ma prazdne pole '{field_name}'.")
        return False
    return True


def validate_sources(value, prefix, report, known_sources):
    if not isinstance(value, list):
        report["errors"].append(f"{prefix} ma nespravny typ pola 'source' (ocakavany zoznam).")
        return False
    if not value:
        report["errors"].append(f"{prefix} ma prazdne pole 'source'.")
        return False

    for idx, item in enumerate(value, start=1):
        if isinstance(item, str):
            if not item.strip():
                report["errors"].append(f"{prefix} ma v 'source' prazdnu hodnotu na pozicii {idx}.")
                return False
            continue
        if isinstance(item, int):
            continue
        report["errors"].append(
            f"{prefix} ma v 'source' hodnotu na pozicii {idx} s nespravnym typom (ocakavany int alebo string)."
        )
        return False

    refs = parse_source_ref_strings(value)
    if not refs:
        report["errors"].append(
            f"{prefix} ma v poli 'source' neplatny format. Ocakava sa format D1:1 alebo cisla stran."
        )
        return False

    if known_sources:
        invalid_refs = sorted(set(refs) - known_sources)
        if invalid_refs:
            report["errors"].append(
                f"{prefix} odkazuje na neexistujuce zdroje: {', '.join(str(p) for p in invalid_refs)}."
            )
            return False

    return True
