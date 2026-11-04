import json

from context_builder import parse_source_ref_strings


BLOOM_LEVELS = {
    "Zapamätať si",
    "Pochopiť",
    "Aplikovať",
    "Analyzovať",
    "Hodnotiť",
    "Vytvoriť",
}

REQUIRED_FIELDS = {
    "id",
    "vzdelávací_objekt",
    "bloom_level",
    "odporúčané_aktivity",
    "odporúčané_zadania",
    "citovane_zdroje",
    "prerekvizity",
}


def validate_lo_json_text(text, allowed_pages=None):
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return {
            "is_valid_json": False,
            "is_valid": False,
            "errors": [f"Nevalidny JSON: {e}"],
            "warnings": [],
            "stats": {"total": 0, "valid": 0, "invalid": 0},
        }
    return validate_learning_objects(data, allowed_pages=allowed_pages)


def validate_learning_objects(data, allowed_pages=None):
    report = {
        "is_valid_json": True,
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "stats": {"total": 0, "valid": 0, "invalid": 0},
    }

    if not isinstance(data, list):
        report["is_valid"] = False
        report["errors"].append("LO vystup musi byt JSON pole objektov.")
        report["stats"]["invalid"] = 1
        return report

    report["stats"]["total"] = len(data)
    known_sources = {str(p).strip() for p in (allowed_pages or []) if str(p).strip()}
    all_ids = []

    for index, item in enumerate(data, start=1):
        prefix = f"LO na indexe {index}"
        item_valid = True

        if not isinstance(item, dict):
            report["errors"].append(f"{prefix} nie je objekt.")
            report["stats"]["invalid"] += 1
            report["is_valid"] = False
            continue

        missing_fields = sorted(REQUIRED_FIELDS - set(item.keys()))
        if missing_fields:
            report["errors"].append(f"{prefix} nema povinne polia: {', '.join(missing_fields)}.")
            item_valid = False

        lo_id = item.get("id")
        if not isinstance(lo_id, int) or lo_id <= 0:
            report["errors"].append(f"{prefix} ma neplatne pole 'id' (ocakavane kladne cele cislo).")
            item_valid = False
        else:
            all_ids.append(lo_id)
            prefix = f"LO {lo_id}"

        lo_name = item.get("vzdelávací_objekt")
        if not isinstance(lo_name, str):
            report["errors"].append(f"{prefix} ma nespravny typ pola 'vzdelávací_objekt' (ocakavany string).")
            item_valid = False
        elif not lo_name.strip():
            report["errors"].append(f"{prefix} ma prazdne pole 'vzdelávací_objekt'.")
            item_valid = False

        bloom = item.get("bloom_level")
        if not isinstance(bloom, str):
            report["errors"].append(f"{prefix} ma nespravny typ pola 'bloom_level' (ocakavany string).")
            item_valid = False
        elif not bloom.strip():
            report["errors"].append(f"{prefix} ma prazdne pole 'bloom_level'.")
            item_valid = False
        elif bloom not in BLOOM_LEVELS:
            allowed = ", ".join(sorted(BLOOM_LEVELS))
            report["errors"].append(
                f"{prefix} ma neplatnu Bloomovu uroven '{bloom}'. Povolené: {allowed}."
            )
            item_valid = False

        if not validate_non_empty_string_or_list(
            item.get("odporúčané_aktivity"),
            prefix,
            "odporúčané_aktivity",
            report
        ):
            item_valid = False

        if not validate_non_empty_string_or_list(
            item.get("odporúčané_zadania"),
            prefix,
            "odporúčané_zadania",
            report
        ):
            item_valid = False

        if not validate_sources(item.get("citovane_zdroje"), prefix, report, known_sources):
            item_valid = False

        if not validate_prerequisites(item.get("prerekvizity"), prefix, report):
            item_valid = False

        if item_valid:
            report["stats"]["valid"] += 1
        else:
            report["stats"]["invalid"] += 1
            report["is_valid"] = False

    duplicates = sorted({lo_id for lo_id in all_ids if all_ids.count(lo_id) > 1})
    if duplicates:
        report["errors"].append(f"Duplicne LO id: {', '.join(str(lo_id) for lo_id in duplicates)}.")
        report["is_valid"] = False

    known_ids = set(all_ids)
    for item in data:
        if not isinstance(item, dict):
            continue
        lo_id = item.get("id")
        prereq = item.get("prerekvizity")
        if not isinstance(lo_id, int) or not isinstance(prereq, list):
            continue

        invalid_refs = sorted({p for p in prereq if isinstance(p, int) and p not in known_ids})
        if invalid_refs:
            report["errors"].append(
                f"LO {lo_id} odkazuje v 'prerekvizity' na neexistujuce id: {', '.join(str(p) for p in invalid_refs)}."
            )
            report["is_valid"] = False

        if lo_id in prereq:
            report["errors"].append(f"LO {lo_id} obsahuje samo seba v 'prerekvizity'.")
            report["is_valid"] = False

    report["stats"]["invalid"] = report["stats"]["total"] - report["stats"]["valid"]
    return report


def validate_non_empty_string_list(value, prefix, field_name, report):
    if not isinstance(value, list):
        report["errors"].append(f"{prefix} ma nespravny typ pola '{field_name}' (ocakavany zoznam).")
        return False
    if not value:
        report["errors"].append(f"{prefix} ma prazdne pole '{field_name}'.")
        return False

    ok = True
    for idx, item in enumerate(value, start=1):
        if not isinstance(item, str):
            report["errors"].append(
                f"{prefix} ma v poli '{field_name}' hodnotu na pozicii {idx} s nespravnym typom (ocakavany string)."
            )
            ok = False
            continue
        if not item.strip():
            report["errors"].append(f"{prefix} ma v poli '{field_name}' prazdnu hodnotu na pozicii {idx}.")
            ok = False
    return ok


def validate_non_empty_string_or_list(value, prefix, field_name, report):
    if isinstance(value, str):
        if not value.strip():
            report["errors"].append(f"{prefix} ma prazdne pole '{field_name}'.")
            return False
        return True
    return validate_non_empty_string_list(value, prefix, field_name, report)


def validate_sources(value, prefix, report, known_sources):
    if isinstance(value, str):
        if not value.strip():
            report["errors"].append(f"{prefix} ma prazdne pole 'citovane_zdroje'.")
            return False
        normalized = [value]
    elif isinstance(value, list):
        if not value:
            report["errors"].append(f"{prefix} ma prazdne pole 'citovane_zdroje'.")
            return False
        normalized = value
    else:
        report["errors"].append(
            f"{prefix} ma nespravny typ pola 'citovane_zdroje' (ocakavany string alebo zoznam)."
        )
        return False

    for idx, item in enumerate(normalized, start=1):
        if isinstance(item, str):
            if not item.strip():
                report["errors"].append(f"{prefix} ma v 'citovane_zdroje' prazdnu hodnotu na pozicii {idx}.")
                return False
            continue
        if isinstance(item, int):
            continue
        report["errors"].append(
            f"{prefix} ma v 'citovane_zdroje' hodnotu na pozicii {idx} s nespravnym typom (ocakavany int alebo string)."
        )
        return False

    refs = parse_source_ref_strings(normalized)
    if not refs:
        report["errors"].append(
            f"{prefix} ma v poli 'citovane_zdroje' neplatny format. Ocakava sa format D1:1 alebo cisla stran."
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


def validate_prerequisites(value, prefix, report):
    if not isinstance(value, list):
        report["errors"].append(f"{prefix} ma nespravny typ pola 'prerekvizity' (ocakavany zoznam).")
        return False

    ok = True
    for idx, item in enumerate(value, start=1):
        if not isinstance(item, int) or item <= 0:
            report["errors"].append(
                f"{prefix} ma v poli 'prerekvizity' neplatnu hodnotu na pozicii {idx} (ocakavane kladne cele cislo)."
            )
            ok = False
    return ok
