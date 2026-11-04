from qgen.context_builder import build_context_for_lo, build_page_map
from qgen.config import setting_or_default
from qgen.llm.retry import generate_json
from qgen.progress import STAGE_LO_EVALUATION, batch_count, report_progress


def analyze_lo_faithfulness(
    segmenty,
    los,
    llm=None,
    model=None,
    verbose: bool = True,
    batch_size=None,
    max_batch_attempts=None,
    document_language: str = "sk",
    context_max_chars=None,
    progress_callback=None,
):
    model = setting_or_default(model, "llm_model")
    batch_size = setting_or_default(batch_size, "lo_batch_size")
    max_batch_attempts = setting_or_default(max_batch_attempts, "check_batch_attempts")
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

        source_text = build_context_for_lo(lo, page_map, max_chars=context_max_chars)
        if not source_text.strip():
            continue

        comparable_items.append({
            "row": item,
            "lo_id": lo.get("id"),
            "lo_text": lo_to_text(lo),
            "source_text": source_text,
        })

    total_batches = batch_count(len(comparable_items), batch_size)
    report_progress(progress_callback, STAGE_LO_EVALUATION, 0, total_batches)
    for start in range(0, len(comparable_items), batch_size):
        batch = comparable_items[start:start + batch_size]
        evaluations = {}
        for attempt in range(1, max_batch_attempts + 1):
            evaluations = evaluate_lo_faithfulness_batch(
                batch,
                llm=llm,
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
        report_progress(progress_callback, STAGE_LO_EVALUATION, start // batch_size + 1, total_batches)

    report["stats"]["los_evaluated"] = len(scores)
    if scores:
        report["stats"]["average_faithfulness_score"] = round(sum(scores) / len(scores), 4)
    return report


def evaluate_lo_faithfulness_batch(batch, llm=None, model=None, verbose: bool = True, document_language: str = "sk"):
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

    # Step 6: no `except Exception: return None` (the LO then lost their score and were
    # filtered out); LLMError propagates.
    parsed = generate_json(llm, prompt, role="check", model=model, verbose=verbose)

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
