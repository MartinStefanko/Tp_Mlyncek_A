from context_builder import build_context_for_lo, build_page_map
from json_load import safe_load_json
from llm_client import generate_with_retry


FAITHFULNESS_MODEL = "gemini-2.5-flash-lite"


def analyze_lo_faithfulness(
    segmenty,
    los,
    client=None,
    model: str = FAITHFULNESS_MODEL,
    verbose: bool = True,
    batch_size: int = 20,
    max_batch_attempts: int = 2,
    document_language: str = "sk",
):
    report = {
        "stats": {
            "los_total": len(los),
            "los_evaluated": 0,
            "average_faithfulness_score": 0.0,
        },
        "items": [],
    }

    if not los:
        return report

    page_map = build_page_map(segmenty)
    scores = []
    comparable_items = []

    for lo in los:
        item = {
            "lo_id": lo.get("id"),
            "lo_name": lo.get("vzdelávací_objekt", ""),
            "source_pages": lo.get("citovane_zdroje", []),
            "faithfulness_score": None,
            "reason": "",
        }
        report["items"].append(item)

        source_text = build_context_for_lo(lo, page_map, max_chars=8000)
        if not source_text.strip():
            continue

        comparable_items.append({
            "row": item,
            "lo_id": lo.get("id"),
            "lo_text": lo_to_text(lo),
            "source_text": source_text,
        })

    for start in range(0, len(comparable_items), batch_size):
        batch = comparable_items[start:start + batch_size]
        evaluations = {}
        for attempt in range(1, max_batch_attempts + 1):
            evaluations = evaluate_lo_faithfulness_batch(
                batch,
                client=client,
                model=model,
                verbose=verbose,
                document_language=document_language,
            )
            if evaluations:
                break
            if verbose and max_batch_attempts > 1:
                print(f"LO faithfulness batch prazdny, opakujem ({attempt}/{max_batch_attempts})")

        for item in batch:
            evaluation = evaluations.get(item["lo_id"])
            if not evaluation:
                continue
            row = item["row"]
            row["faithfulness_score"] = evaluation.get("skore")
            row["reason"] = evaluation.get("zdovodnenie", "")
            if isinstance(row["faithfulness_score"], int):
                scores.append(row["faithfulness_score"])

    report["stats"]["los_evaluated"] = len(scores)
    if scores:
        report["stats"]["average_faithfulness_score"] = round(sum(scores) / len(scores), 4)
    return report


def evaluate_lo_faithfulness_batch(batch, client=None, model: str = FAITHFULNESS_MODEL, verbose: bool = True, document_language: str = "sk"):
    parts = []
    for item in batch:
        parts.append(
            f"LO ID: {item['lo_id']}\n"
            f"LO:\n\"\"\"{item['lo_text']}\"\"\"\n\n"
            f"Zdroj:\n\"\"\"{item['source_text']}\"\"\""
        )
    joined = "\n\n-----\n\n".join(parts)

    if document_language == "en":
        prompt = f"""
Assess the faithfulness of each learning objective against its source.

Score 1-5:
1 = the learning objective is not supported by the source
5 = the learning objective is fully supported by the source

Records:
{joined}

Return ONLY valid JSON as an array of objects:
[
  {{"lo_id": 1, "skore": 1, "zdovodnenie": "short justification in English"}}
]
"""
    else:
        prompt = f"""
Posud faithfulness LO voci zdroju pre kazdy zaznam.

Skore 1-5:
1 = tvrdenia LO nie su podlozene zdrojom
5 = LO je plne podlozene zdrojom

Zaznamy:
{joined}

Vrat LEN validny JSON ako pole objektov:
[
  {{"lo_id": 1, "skore": 1, "zdovodnenie": "kratke zdovodnenie"}}
]
"""

    try:
        response = generate_with_retry(prompt, client=client, model=model, verbose=verbose)
        parsed = safe_load_json(response.text if response else "")
    except Exception as e:
        if verbose:
            print(f"Faithfulness hodnotenie LO zlyhalo: {e}")
        return None

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return {}

    evaluations = {}
    for row in parsed:
        if not isinstance(row, dict):
            continue
        lo_id = row.get("lo_id")
        try:
            score = int(row.get("skore"))
        except (TypeError, ValueError):
            continue
        reason = str(row.get("zdovodnenie", "")).strip()
        evaluations[lo_id] = {"skore": max(1, min(5, score)), "zdovodnenie": reason}
    return evaluations


def lo_to_text(lo):
    parts = [
        f"vzdelavaci_objekt: {lo.get('vzdelávací_objekt', '')}",
        f"bloom_level: {lo.get('bloom_level', '')}",
        f"odporucane_aktivity: {lo.get('odporúčané_aktivity', [])}",
        f"odporucane_zadania: {lo.get('odporúčané_zadania', [])}",
    ]
    return "\n".join(parts)
