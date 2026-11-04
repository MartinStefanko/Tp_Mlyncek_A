import json

from context_builder import parse_source_ref_strings


REQUIRED_ITEM_FIELDS = {
    "id",
    "lo_id",
    "typ",
    "otazka",
    "odpoved",
    "napoveda",
    "citovane_zdroje",
}

ALLOWED_ITEM_TYPES = {"teoreticka_otazka", "prakticka_uloha"}


def validate_items_json_text(text, allowed_pages=None, valid_lo_ids=None):
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
    return validate_items(data, allowed_pages=allowed_pages, valid_lo_ids=valid_lo_ids)


def validate_items(data, allowed_pages=None, valid_lo_ids=None):
    report = {
        "is_valid_json": True,
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "stats": {"total": 0, "valid": 0, "invalid": 0},
    }

    if not isinstance(data, list):
        report["is_valid"] = False
        report["errors"].append("Vystup poloziek musi byt JSON pole objektov.")
        report["stats"]["invalid"] = 1
        return report

    report["stats"]["total"] = len(data)
    known_sources = {str(p).strip() for p in (allowed_pages or []) if str(p).strip()}
    known_lo_ids = set(valid_lo_ids or [])
    item_ids = []

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
        if not isinstance(lo_id, int) or lo_id <= 0:
            report["errors"].append(f"{prefix} ma neplatne pole 'lo_id' (ocakavane kladne cele cislo).")
            item_valid = False
        elif known_lo_ids and lo_id not in known_lo_ids:
            report["errors"].append(f"{prefix} odkazuje na neexistujuce LO id {lo_id}.")
            item_valid = False

        item_type = item.get("typ")
        if not isinstance(item_type, str):
            report["errors"].append(f"{prefix} ma nespravny typ pola 'typ' (ocakavany string).")
            item_valid = False
        elif not item_type.strip():
            report["errors"].append(f"{prefix} ma prazdne pole 'typ'.")
            item_valid = False
        elif item_type not in ALLOWED_ITEM_TYPES:
            report["errors"].append(
                f"{prefix} ma neplatnu hodnotu pola 'typ' ({item_type})."
            )
            item_valid = False

        if not validate_non_empty_text(item.get("otazka"), prefix, "otazka", report):
            item_valid = False
        if not validate_non_empty_value(item.get("odpoved"), prefix, "odpoved", report):
            item_valid = False
        if not validate_non_empty_value(item.get("napoveda"), prefix, "napoveda", report):
            item_valid = False
        if not validate_sources(item.get("citovane_zdroje"), prefix, report, known_sources):
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
    return report


def validate_non_empty_text(value, prefix, field_name, report):
    if not isinstance(value, str):
        report["errors"].append(f"{prefix} ma nespravny typ pola '{field_name}' (ocakavany string).")
        return False
    if not value.strip():
        report["errors"].append(f"{prefix} ma prazdne pole '{field_name}'.")
        return False
    return True


def validate_non_empty_value(value, prefix, field_name, report):
    if value is None:
        report["errors"].append(f"{prefix} ma prazdne pole '{field_name}'.")
        return False
    if isinstance(value, str):
        if not value.strip():
            report["errors"].append(f"{prefix} ma prazdne pole '{field_name}'.")
            return False
        return True
    if isinstance(value, list):
        if not value:
            report["errors"].append(f"{prefix} ma prazdne pole '{field_name}'.")
            return False
        return True
    text = str(value).strip()
    if not text:
        report["errors"].append(f"{prefix} ma prazdne pole '{field_name}'.")
        return False
    return True


def validate_sources(value, prefix, report, known_sources):
    if not isinstance(value, list):
        report["errors"].append(f"{prefix} ma nespravny typ pola 'citovane_zdroje' (ocakavany zoznam).")
        return False
    if not value:
        report["errors"].append(f"{prefix} ma prazdne pole 'citovane_zdroje'.")
        return False

    for idx, item in enumerate(value, start=1):
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

    refs = parse_source_ref_strings(value)
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
