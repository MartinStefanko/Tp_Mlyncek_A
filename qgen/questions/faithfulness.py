from context_builder import build_context_for_sources, build_page_map
from json_load import safe_load_json
from llm_client import generate_with_retry


FAITHFULNESS_MODEL = "gemini-2.5-flash-lite"


def analyze_item_faithfulness(
    segmenty,
    items,
    client=None,
    model: str = FAITHFULNESS_MODEL,
    verbose: bool = True,
    batch_size: int = 10,
    max_batch_attempts: int = 2,
    document_language: str = "sk",
):
    report = {
        "stats": {
            "items_total": len(items),
            "items_evaluated": 0,
            "average_faithfulness_score": 0.0,
            "faithful_items": 0,
            "faithful_items_percent": 0.0,
        },
        "items": [],
    }

    if not items:
        return report

    page_map = build_page_map(segmenty)
    scores = []
    faithful_items = 0
    comparable_items = []

    for item in items:
        row = {
            "item_id": item.get("id"),
            "lo_id": item.get("lo_id"),
            "source_pages": item.get("citovane_zdroje", []),
            "faithfulness_score": None,
            "faithful": False,
            "reason": "",
        }
        report["items"].append(row)

        source_text = build_context_for_item(item, page_map)
        if not source_text.strip():
            continue

        comparable_items.append({
            "row": row,
            "item_id": item.get("id"),
            "item_text": item_to_text(item),
            "source_text": source_text,
        })

    for start in range(0, len(comparable_items), batch_size):
        batch = comparable_items[start:start + batch_size]
        evaluations = {}
        for attempt in range(1, max_batch_attempts + 1):
            evaluations = evaluate_item_faithfulness_batch(
                batch,
                client=client,
                model=model,
                verbose=verbose,
                document_language=document_language,
            )
            if evaluations:
                break
            if verbose and max_batch_attempts > 1:
                print(f"Item faithfulness batch prazdny, opakujem ({attempt}/{max_batch_attempts})")

        for item in batch:
            evaluation = evaluations.get(item["item_id"])
            if not evaluation:
                continue
            row = item["row"]
            row["faithfulness_score"] = evaluation.get("skore")
            row["reason"] = evaluation.get("zdovodnenie", "")
            if isinstance(row["faithfulness_score"], int):
                scores.append(row["faithfulness_score"])
                row["faithful"] = row["faithfulness_score"] >= 4
                if row["faithful"]:
                    faithful_items += 1

    report["stats"]["items_evaluated"] = len(scores)
    if scores:
        report["stats"]["average_faithfulness_score"] = round(sum(scores) / len(scores), 4)
        report["stats"]["faithful_items"] = faithful_items
        report["stats"]["faithful_items_percent"] = round((faithful_items / len(scores)) * 100, 2)
    return report


def evaluate_item_faithfulness_batch(batch, client=None, model: str = FAITHFULNESS_MODEL, verbose: bool = True, document_language: str = "sk"):
    parts = []
    for item in batch:
        parts.append(
            f"ITEM ID: {item['item_id']}\n"
            f"Otazka a odpoved:\n\"\"\"{item['item_text']}\"\"\"\n\n"
            f"Zdroj:\n\"\"\"{item['source_text']}\"\"\""
        )
    joined = "\n\n-----\n\n".join(parts)

    if document_language == "en":
        prompt = f"""
Assess the faithfulness of each answer against its source.

Score 1-5:
1 = the answer contains unsupported claims
5 = the answer is fully supported by the source

Records:
{joined}

Return ONLY valid JSON as an array of objects:
[
  {{"item_id": 1, "skore": 1, "zdovodnenie": "short justification in English"}}
]
"""
    else:
        prompt = f"""
Posud faithfulness odpovede voci zdroju pre kazdy zaznam.

Skore 1-5:
1 = odpoved obsahuje nepodlozene tvrdenia
5 = odpoved je plne podlozena zdrojom

Zaznamy:
{joined}

Vrat LEN validny JSON ako pole objektov:
[
  {{"item_id": 1, "skore": 1, "zdovodnenie": "kratke zdovodnenie"}}
]
"""

    try:
        response = generate_with_retry(prompt, client=client, model=model, verbose=verbose)
        parsed = safe_load_json(response.text if response else "")
    except Exception as e:
        if verbose:
            print(f"Faithfulness hodnotenie odpovede zlyhalo: {e}")
        return None

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return {}

    evaluations = {}
    for row in parsed:
        if not isinstance(row, dict):
            continue
        item_id = row.get("item_id")
        try:
            score = int(row.get("skore"))
        except (TypeError, ValueError):
            continue
        reason = str(row.get("zdovodnenie", "")).strip()
        evaluations[item_id] = {"skore": max(1, min(5, score)), "zdovodnenie": reason}
    return evaluations


def item_to_text(item):
    parts = [
        f"otazka: {item.get('otazka', '')}",
        f"odpoved: {item.get('odpoved', '')}",
    ]
    return "\n".join(parts)


def build_context_for_item(item, page_map, max_chars=8000):
    return build_context_for_sources(item.get("citovane_zdroje", []), page_map, max_chars=max_chars)
